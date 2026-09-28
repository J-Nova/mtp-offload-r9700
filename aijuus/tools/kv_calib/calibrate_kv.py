"""--kvcalibration — capture FP8 KV-cache scales for a checkpoint.

    docker run ... tcclaviger/vllm:dev --kvcalibration \
        --modeldir /models/MyModel [--outputdir /out]

Without calibrated scales vLLM defaults k_scale/v_scale to 1.0
(``quantization/kv_cache.py:111-115``). On a model whose K activations run an
amax of ~24, that is not a small error, it is the wrong scale entirely. This
walks a calibration corpus through the model and records what the real range is.

How the values are captured
---------------------------
vLLM already exposes q/k/v before the cache write through
``Attention.calc_kv_scales``. The worker extension replaces that method with an
accumulating version (see kv_calib_worker.py). This half builds the prompt set,
drives the engine one sequence at a time, and writes the result out.

**One sequence per request, never concatenated.** The activation-aware quantizer
packs the corpus into one flat token stream because it only needs aggregate
second moments. Here every capture must be attributable to a single prompt, so
each row is its own request and each fold into the accumulator is one sequence's
amax. That is also what makes the spread metrics meaningful.

The emitted value
-----------------
    scale = (observed_max * PAD) / range_constant

The divisors are NOT the FP8 max. ``envs.Q_SCALE_CONSTANT`` / ``K_`` / ``V_``
are 200 / 200 / 100 (``vllm/envs.py:145-147``), and ``calc_kv_scales`` divides
by exactly those. They are read from ``envs`` at run time rather than hardcoded,
so an overridden constant cannot silently desync checkpoint from runtime.

PAD is headroom for outliers this corpus never saw. It is deliberately modest:
10% over the observed max barely moves accuracy, while the difference between a
calibrated scale and the 1.0 default is large. The report's spread column is the
evidence for whether the corpus explored the range at all.

Capture runs at ``kv_cache_dtype="auto"``, NOT fp8. ``attention.py:271`` sets
``calculate_kv_scales`` from the cache config with no dependency on the dtype, so
the capture fires either way — and at ``auto`` it measures unquantized q/k/v,
which is the truer amax.

--mtp: calibrating the MTP draft
--------------------------------
An MTP draft has its OWN attention layer and its own KV cache; without scales it
writes fp8 at 1.0 while the target serves calibrated. ``--mtp`` runs the pass
WITH speculative decoding enabled (normally deliberately off, see the LLM() call)
and generates a few tokens per prompt, because the speculator only runs on decode
steps — at ``max_tokens=1`` the request finishes at the first sampled token and
the draft would fold zero captures. The draft's inputs are target hidden states,
so this measures the distribution that actually serves, quantized target and all.

Only the draft's scales are EMITTED: the run's target captures include decode
steps and spec-decode batching, a differently-conditioned run than the documented
prefill-only methodology, so the published target scales are left untouched and
the new ``mtp.*`` entries are merged into the existing shard. The report lands
separately in ``kv_calibration_report_mtp.json``; its spread is per draft step,
not per sequence.

Tensor naming: probed, never guessed
------------------------------------
Every emitted tensor name is verified in the worker against the model's OWN
``load_weights`` before it is written: a candidate name is accepted only when a
probe scalar loaded through it lands on that layer's scale parameter
(``kv_calib_worker._probe_checkpoint_names``). Candidate spellings come from the
checkpoint's real index (so wrapper inversions like ``language_model.model.X``
vs ``model.language_model.X`` are read, not encoded) crossed with the
``.attn.`` / ``.<q|k|v>_proj.`` / ``.qkv_proj.`` conventions the loader's remap
understands. The result: for any model vLLM can load — target or MTP draft —
the shard needs no renaming afterwards, and a layer no candidate reaches fails
loudly at install instead of serving at 1.0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "expert_map"))

WORKER_EXT = "kv_calib_worker.KVCalibWorkerExtension"

# Corpora and subset are fixed, not parameters: calibration must measure the
# same corpus every run for numbers to compare across models and days.
TEXT_CORPUS = (
    Path(__file__).resolve().parent.parent
    / "quant_engine"
    / "aa"
    / "corpus"
    / "aa_calib_2m.jsonl.gz"
)
VISION_CORPUS = (
    Path(__file__).resolve().parent.parent
    / "expert_map"
    / "corpus"
    / "expert_map_calib_v4.jsonl.gz"
)
# The vision rows reference absolute build-host paths (/mnt/raid/datasets/
# calib_images/img_*.jpg), which exist nowhere inside a container on someone
# else's machine. The 100 images ship in the image next to the corpus (see the
# corpus COPY in the Dockerfile), and passing this dir as image_root makes
# calibration.resolve_image arm 1 (image_root/name) hit for every row.
VISION_IMAGES_DIR = VISION_CORPUS.parent / "calib_images"
N_VISION = 120
# Tokens per domain kept by the token-parity subset. 25000 against the
# corpus's ~74k/domain keeps a fraction of the 3438 text rows. What this pass measures
# is a per-layer max over per-sequence maxima, so its accuracy is governed by
# whether the corpus reached the extremes, not by how many rows restate the
# middle of the distribution: every domain's longest row is taken first (the
# only rows exercising high RoPE positions, where K amax peaks post-rotary),
# then the rest are sampled across the length distribution. A quarter of the
# whole-corpus wall clock for the same measured range.
SUBSET_TOKENS = 25000
PAD = 1.10
SCALE_SHARD = "model-kvscales.safetensors"
REPORT = "kv_calibration_report.json"
REPORT_MTP = "kv_calibration_report_mtp.json"
# --mtp decode length: enough decode steps for the draft to propose across a
# spread of positions, cheap enough to be negligible per prompt.
DEFAULT_MTP_DECODE_TOKENS = 32

log = logging.getLogger("kvcalib")


class ScaleKind(Enum):
    """The three scales, their projection module, and vLLM's divisor.

    ``proj`` is the module the tensor is written on. vLLM's loader maps
    ``<block>.<proj>.<suffix>`` back to ``<block>.attn.<suffix>``
    (``weight_utils.py:1417``), so writing on the projection is what a
    checkpoint like Step-3.7 already does and needs no loader change.
    """

    Q = ("q", "q_scale", "q_proj", "Q_SCALE_CONSTANT")
    K = ("k", "k_scale", "k_proj", "K_SCALE_CONSTANT")
    V = ("v", "v_scale", "v_proj", "V_SCALE_CONSTANT")

    def __init__(self, tag: str, suffix: str, proj: str, env_name: str):
        self.tag = tag
        self.suffix = suffix
        self.proj = proj
        self.env_name = env_name

    @property
    def divisor(self) -> float:
        import os

        from vllm import envs

        # v27 dropped the *_SCALE_CONSTANT entries from envs' lazy lookup
        # (their runtime consumer, calc_kv_scales, was removed) while keeping
        # the annotations. Honor a real env override first, then the envs
        # attr if it still resolves, then the documented defaults.
        defaults = {"Q_SCALE_CONSTANT": 200, "K_SCALE_CONSTANT": 200,
                    "V_SCALE_CONSTANT": 100}
        if self.env_name in os.environ:
            return float(os.environ[self.env_name])
        try:
            return float(getattr(envs, self.env_name))
        except AttributeError:
            return float(defaults[self.env_name])


# ── model inspection ────────────────────────────────────────────────────────


def is_vision_capable(model_dir: Path) -> bool:
    """A vision tower declared in config.json, by structure not by name."""
    cfg = json.loads((model_dir / "config.json").read_text())
    if "vision_config" in cfg:
        return True
    tc = cfg.get("text_config")
    return bool(isinstance(tc, dict) and "vision_config" in tc)


def assert_no_kv_scheme(model_dir: Path) -> None:
    """attention.py:285 disables the capture for such checkpoints."""
    cfg = json.loads((model_dir / "config.json").read_text())
    q = cfg.get("quantization_config") or {}
    scheme = q.get("kv_cache_scheme")
    if scheme is not None:
        raise SystemExit(
            f"{model_dir}/config.json declares quantization_config."
            f"kv_cache_scheme={scheme!r}.\n"
            f"  attention.py:285 forces kv_cache_dtype=fp8 and disables "
            f"calculate_kv_scales for such checkpoints, so nothing would be "
            f"captured. This checkpoint already carries KV scales."
        )


# ── prompts ─────────────────────────────────────────────────────────────────


def subset_by_token_parity(rows, target: int):
    """Trim the corpus to ~``target`` tokens per domain, preserving parity.

    The baked corpus is balanced by TOKENS (~74k/domain), not rows, so a
    row-count cap would destroy that balance — a domain of 879 short rows and one
    of 2 long rows carry the same weight by design. This keeps the same property
    in the subset.

    Two rules beyond the budget:

    * **Each domain's longest row is always taken first.** Those are the only
      rows exercising high RoPE positions, and ``calc_kv_scales`` sees K
      post-rotary, so they are where K amax is most likely to peak.
    * **The rest are sampled evenly across the length distribution**, not taken
      longest-first or in corpus order, so a domain's sample spans its real range
      of prompt lengths instead of skewing to one end.

    Some domains cannot reach parity and that is a property of the data, not a
    bug: ``books_xlong`` has 2 rows whose shortest is ~39k tokens, so it
    overshoots any smaller budget. Those are the long-context domains, where
    overshooting is the harmless direction.
    """
    import collections

    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("domain", "?")].append(r)

    out, report = [], []
    for domain, rs in by.items():
        rs = sorted(rs, key=lambda r: -int(r.get("n_tokens", 0) or 0))
        take = [rs[0]]
        total = int(rs[0].get("n_tokens", 0) or 0)
        rest = rs[1:]
        if rest and total < target:
            # Walk the remaining rows by length RANK with a stride, so the
            # sample spans short..long rather than clustering at one end.
            stride = max(1, len(rest) // 64)
            order = [
                rest[i]
                for start in range(stride)
                for i in range(start, len(rest), stride)
            ]
            seen = set()
            for r in order:
                if total >= target:
                    break
                if id(r) in seen:
                    continue
                seen.add(id(r))
                take.append(r)
                total += int(r.get("n_tokens", 0) or 0)
        out.extend(take)
        report.append((domain, len(take), total))

    tot = sum(t for _, _, t in report)
    per = [t for _, _, t in report]
    log.info(
        "token-parity subset: %d rows, %s tokens over %d domains "
        "(target %s/domain, parity %.1fx)",
        len(out),
        f"{tot:,}",
        len(report),
        f"{target:,}",
        max(per) / min(per) if min(per) else 0,
    )
    for d, n, t in sorted(report, key=lambda x: -x[2])[:5]:
        log.info("   %-16s %4d rows %9s tok", d, n, f"{t:,}")
    return out


def build_prompts(
    model_dir: Path,
    want_vision: bool,
    n_vision: int,
    text_corpus: Path,
    vision_corpus: Path,
    image_root,
    seq_len: int,
    tokens_per_domain: int = 0,
):
    """Corpus -> engine prompts.

    ``seq_len`` is the per-row truncation cap and tracks --max-model-len, NOT a
    fixed number. The baked corpus runs to 41,383 tokens on its longest row
    (median 203, p99 3193, 10 rows over 8k), and those long rows are the only
    ones exercising high RoPE positions — where K amax is most likely to peak,
    since calc_kv_scales sees K post-rotary. Truncating them silently discards
    the part of the range this calibration exists to find.
    """
    import calibration as calib
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    rows = calib.read_corpus(text_corpus, skip_images=True, log=log)
    if tokens_per_domain:
        rows = subset_by_token_parity(rows, tokens_per_domain)
    prompts = calib.to_prompts(tok, rows, seq_len, log)
    capped = sum(1 for p in prompts if len(p.get("token_ids", ())) >= seq_len)
    log.info(
        "text prompts: %d (%d hit the %d-token cap)", len(prompts), capped, seq_len
    )
    if capped:
        log.warning(
            "%d row(s) were truncated; raise --max-model-len to keep "
            "their high-position activations",
            capped,
        )

    if want_vision:
        try:
            vrows = [
                r
                for r in calib.read_corpus(
                    vision_corpus, Path(image_root) if image_root else None, log=log
                )
                if r.get("images")
            ]
            vrows = vrows[:n_vision]
            vprompts = calib.to_prompts(tok, vrows, seq_len, log)
            prompts += vprompts
            log.info("vision prompts: %d", len(vprompts))
        except Exception as e:  # noqa: BLE001
            # Missing image mount is a degraded run, not a failed one.
            log.warning("vision samples unavailable (%s); text only", e)
    return prompts


# ── output ──────────────────────────────────────────────────────────────────


def write_scales(
    outdir: Path,
    model_dir: Path,
    scales: dict,
    report: dict,
    merge: bool = False,
    report_name: str = REPORT,
):
    """A scalars-only safetensors plus the index entries that reach it.

    A new shard, never a rewrite of an existing one: a failure here cannot
    corrupt the checkpoint. The index is copied from the model, extended, and
    written temp-file + rename.

    ``merge=True`` (--mtp): existing entries in the shard are preserved and the
    new ones added, so the draft's scales join the target's instead of
    replacing the whole shard.
    """
    import torch
    from safetensors.torch import save_file

    outdir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    if merge:
        for src in (outdir / SCALE_SHARD, model_dir / SCALE_SHARD):
            if src.exists():
                from safetensors import safe_open

                with safe_open(str(src), framework="pt") as f:
                    tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 - safe_open has no __contains__/__iter__
                log.info("merging into %d existing scalars from %s", len(tensors), src)
                break
    tensors.update(
        {
            name: torch.tensor(float(val), dtype=torch.float32)
            for name, val in scales.items()
        }
    )
    shard = outdir / SCALE_SHARD
    tmp = shard.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(shard)
    log.info("wrote %d scalars -> %s", len(tensors), shard)

    idx_src = model_dir / "model.safetensors.index.json"
    if idx_src.exists():
        idx = json.loads(idx_src.read_text())
        wmap = idx.setdefault("weight_map", {})
        # The shard just written holds exactly ``tensors``. Any index entry
        # still pointing OTHER names at it (a prior calibration whose probed
        # spelling has since drifted with the loader) would dangle and fail
        # weight load, so drop them.
        stale = [n for n, s in wmap.items() if s == SCALE_SHARD and n not in tensors]
        for name in stale:
            del wmap[name]
        if stale:
            log.info(
                "dropped %d stale scale entries from a prior calibration", len(stale)
            )
        new = [n for n in tensors if wmap.get(n) != SCALE_SHARD]
        for name in tensors:
            wmap[name] = SCALE_SHARD
        meta = idx.setdefault("metadata", {})
        if "total_size" in meta:
            meta["total_size"] = int(meta["total_size"]) + 4 * (len(new) - len(stale))
        idx_out = outdir / "model.safetensors.index.json"
        t = idx_out.with_suffix(".tmp")
        t.write_text(json.dumps(idx, indent=2))
        t.replace(idx_out)
        log.info("index updated (+%d new entries) -> %s", len(new), idx_out)
    else:
        log.warning("no index at %s; wrote the shard only", idx_src)

    (outdir / report_name).write_text(json.dumps(report, indent=2))
    log.info("report -> %s", outdir / report_name)


# ── main ────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="--kvcalibration",
        description="Capture FP8 KV-cache q/k/v scales from a calibration corpus.",
    )
    ap.add_argument("--kvcalibration", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--modeldir", required=True, type=Path)
    ap.add_argument(
        "--outputdir",
        default=None,
        type=Path,
        help="where the scalars land (default: --modeldir)",
    )
    ap.add_argument("-tp", "--tensor-parallel-size", dest="tp", type=int, default=4)
    ap.add_argument(
        "--pad",
        type=float,
        default=PAD,
        help=f"headroom over the observed max (default {PAD})",
    )
    ap.add_argument(
        "--mtp",
        action="store_true",
        help="calibrate the MTP draft's KV scales: run WITH "
        "speculative decoding, decode a few tokens per prompt "
        "so the draft actually proposes, and MERGE only the "
        "draft's mtp.* scales into the existing shard. Target "
        "scales must already exist (run without --mtp first).",
    )
    ap.add_argument(
        "--mtp-method",
        default="mtp",
        help="speculative-config method (default mtp; "
        "any name in MTPModelTypes normalizes to 'mtp')",
    )
    ap.add_argument(
        "--mtp-num-tokens",
        type=int,
        default=4,
        help="num_speculative_tokens; match the serve compose",
    )
    ap.add_argument(
        "--mtp-decode-tokens",
        type=int,
        default=DEFAULT_MTP_DECODE_TOKENS,
        help="tokens generated per prompt in --mtp mode; the "
        "speculator only runs on decode steps, so 1 would "
        "capture nothing from the draft",
    )
    ap.add_argument(
        "--dryRunLimit", type=int, default=0, help="smoke test: first N prompts"
    )
    ap.add_argument(
        "--textOnly",
        action="store_true",
        help="calibrate on the text corpus only, even when the model "
        "declares a vision tower",
    )
    # 0.75, not something greedier: this is a calibration pass with
    # max_num_seqs=1 and max_tokens=1, so it needs weights plus a token of KV,
    # not a serving-sized cache — and anything else resident (a TTS model, say)
    # eats the difference. Too high fails at startup with "Free memory on device
    # cuda:0 is less than desired GPU memory utilization".
    ap.add_argument(
        "--gpu-memory-utilization", dest="gpu_util", type=float, default=0.75
    )
    ap.add_argument(
        "--max-model-len",
        dest="max_len",
        type=int,
        default=16384,
        help="16384 fits a whole-sequence eager forward on 32 GiB cards at "
        "TP4 (chunked prefill is off, so max_model_len IS the forward size); "
        "corpus rows beyond it are truncated with a warning. Raise on bigger "
        "GPUs to keep the tail rows' high-position activations (longest "
        "baked row: 41,383).",
    )
    ap.add_argument(
        "--max-num-batched-tokens", dest="max_batched", type=int, default=4096
    )
    ap.add_argument("--mm-processor-cache-gb", dest="mm_cache", type=float, default=0.5)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    model_dir = args.modeldir
    outdir = args.outputdir or model_dir

    assert_no_kv_scheme(model_dir)
    vision = is_vision_capable(model_dir) and not args.textOnly
    log.info("model %s | vision=%s | out %s", model_dir, vision, outdir)

    # Truncation cap leaves headroom below max_model_len: a row truncated to
    # exactly max_len would fail engine validation (prompt + >=1 output token).
    prompts = build_prompts(
        model_dir,
        vision,
        N_VISION,
        TEXT_CORPUS,
        VISION_CORPUS,
        VISION_IMAGES_DIR,
        max(1, args.max_len - 64),
        SUBSET_TOKENS,
    )
    if args.dryRunLimit:
        prompts = prompts[: args.dryRunLimit]
    log.info("calibrating on %d sequences", len(prompts))

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Chunked prefill is off by design (a split sequence breaks "one amax per
    # sequence"), so every prompt must fit in ONE batch and vLLM rejects a
    # max_num_batched_tokens below max_model_len rather than silently capping
    # sequence length. Raise it to match rather than fail.
    batched = args.max_batched
    if batched < args.max_len:
        log.info(
            "raising --max-num-batched-tokens %d -> %d to match "
            "--max-model-len (chunked prefill is off)",
            batched,
            args.max_len,
        )
        batched = args.max_len

    # kv_cache_dtype stays "auto": attention.py:271 arms the capture from the
    # cache config alone, and at auto we measure UNQUANTIZED q/k/v.
    # enforce_eager / max_num_seqs=1 / no chunked prefill / no prefix caching:
    # one prefill per sequence, so one amax per sequence, unambiguously.
    llm_kwargs = {}
    if args.mtp:
        llm_kwargs["speculative_config"] = {
            "method": args.mtp_method,
            "num_speculative_tokens": args.mtp_num_tokens,
        }
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=args.tp,
        worker_extension_cls=WORKER_EXT,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # calculate_kv_scales was removed in v27 along with the runtime
        # calc path; the worker extension owns the capture seam entirely
        # (it wraps Attention.forward at kv_install time).
        kv_cache_dtype="auto",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_len,
        max_num_batched_tokens=batched,
        mm_processor_cache_gb=args.mm_cache,
        limit_mm_per_prompt={"image": 20, "video": 1},
        trust_remote_code=True,
        **llm_kwargs,
    )
    # Deliberately NOT set, and each for a reason that would corrupt the result:
    #   speculative_config   - the draft model has its own attention layers; they
    #                          land in the same registry and would fold draft
    #                          activations into the accumulator. --mtp inverts
    #                          this on purpose: it enables the draft precisely
    #                          to capture it, and emits ONLY the mtp.* scales.
    #   prefix caching       - a cached prefix is never re-forwarded, so those
    #                          sequences contribute nothing and vanish silently.
    #   chunked prefill      - splits one sequence across forwards, so "one amax
    #                          per sequence" stops being true.
    # Serving flags that never reach q/k/v (tool/reasoning parsers, generation
    # config, compilation config) are omitted because they only touch generated
    # text, and this run emits one token and discards it.
    # No rope override. Calibration runs well inside the model's native context,
    # so YaRN would only change the positions the scales are measured at.

    info = llm.collective_rpc("kv_install", args=({},))[0]
    log.info("armed %d attention layers", info["n_armed"])
    for cand, why in sorted((info.get("probe_debug") or {}).items()):
        log.warning("probe: %s -> %s", cand, why)
    draft_names = set(info.get("draft_layer_names") or [])
    if args.mtp and not draft_names:
        raise SystemExit(
            "--mtp requested but no draft attention layer was armed; the "
            "speculative config did not produce a draft with its own KV cache."
        )
    if args.mtp:
        log.info("draft layers: %s", sorted(draft_names))
    llm.collective_rpc("kv_reset")

    # --mtp needs decode steps: the speculator never runs on a request that
    # finishes at its first sampled token.
    max_tokens = args.mtp_decode_tokens if args.mtp else 1
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, detokenize=False)
    t0 = time.time()
    for i, p in enumerate(prompts):
        if p["kind"] == "vision":
            from map_experts import _vision_messages

            # preserve_thinking matches the serve compose: it changes template
            # rendering, so it changes prompt tokens and therefore the
            # activations. It is a chat() kwarg, not an EngineArgs one. Only
            # reaches vision rows; text rows are pre-tokenized by calibration.py
            # upstream of vLLM.
            llm.chat(
                _vision_messages(p),
                sampling,
                use_tqdm=False,
                add_generation_prompt=True,
                chat_template_kwargs={"preserve_thinking": True},
            )
        else:
            llm.generate(
                TokensPrompt(prompt_token_ids=p["token_ids"]), sampling, use_tqdm=False
            )
        if (i + 1) % 100 == 0 or i + 1 == len(prompts):
            el = time.time() - t0
            log.info(
                "  %d/%d  %.2fs/prompt  eta %.0fs",
                i + 1,
                len(prompts),
                el / (i + 1),
                el / (i + 1) * (len(prompts) - i - 1),
            )

    acc = llm.collective_rpc("kv_collect")[0]
    names = info["names"]

    if args.mtp:
        # Only the draft's scales are emitted; the target's captures in this
        # run include decode steps and spec-decode batching, so the published
        # prefill-calibrated target scales stay as they are.
        missing = draft_names - set(acc)
        if missing:
            raise SystemExit(
                f"draft layer(s) folded no captures: {sorted(missing)}. "
                f"The speculator never ran; raise --mtp-decode-tokens."
            )
        acc = {k: v for k, v in acc.items() if k in draft_names}

    unverified = set(info.get("unverified") or [])
    scales, unverified_scales, report = (
        {},
        {},
        {
            "pad": args.pad,
            "sequences": len(prompts),
            "vision": vision,
            "mtp": args.mtp,
            "layers": {},
        },
    )
    for lname in sorted(acc):
        entry = {}
        for kind in ScaleKind:
            s = acc[lname].get(kind.tag)
            if not s or s["max"] is None:
                continue
            value = (s["max"] * args.pad) / kind.divisor
            dest = unverified_scales if lname in unverified else scales
            dest[names[lname][kind.tag]] = value
            entry[kind.tag] = {
                "observed_max": s["max"],
                "observed_min": s["min"],
                "mean": s["mean"],
                "n": s["n"],
                "spread": (s["max"] / s["min"]) if s["min"] else None,
                "divisor": kind.divisor,
                "scale": value,
                "tensor": names[lname][kind.tag],
                "name_unverified": lname in unverified,
            }
        report["layers"][lname] = entry

    llm.collective_rpc("kv_uninstall")
    write_scales(
        outdir,
        model_dir,
        scales,
        report,
        merge=args.mtp,
        report_name=REPORT_MTP if args.mtp else REPORT,
    )
    if unverified_scales:
        # Captures whose tensor names could NOT be verified against the
        # model's own load_weights. The data is real; only the naming is
        # unproven — so it goes in a side file the serve never reads, with a
        # filename nobody can miss, and NO index entries. Rename/merge into
        # the checkpoint once the correct spellings are known.
        import torch as _torch
        from safetensors.torch import save_file as _save_file

        side = (
            outdir
            / "WARNING-UNVERIFIED-TENSOR-NAMES-DO-NOT-SERVE-kv_scales.safetensors"
        )
        _save_file(
            {
                k: _torch.tensor(v, dtype=_torch.float32)
                for k, v in unverified_scales.items()
            },
            str(side),
        )
        log.warning(
            "%d scale(s) across %d layer(s) had NO load-probe-verified name; "
            "written with GENERIC names to %s — NOT added to the index. Fix "
            "the names before serving.",
            len(unverified_scales),
            len(unverified),
            side,
        )

    sp = [
        e[k]["spread"] for e in report["layers"].values() for k in e if e[k]["spread"]
    ]
    if sp:
        log.info(
            "per-sequence amax spread (max/min): median %.1fx  worst %.1fx",
            sorted(sp)[len(sp) // 2],
            max(sp),
        )
    log.info("DONE")


if __name__ == "__main__":
    main()
