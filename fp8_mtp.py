#!/usr/bin/env python3
"""MXFP4 body + FP8 drafter, in one checkpoint.

Run this once against a Quark MXFP4 snapshot to produce the checkpoint
serve-mxfp4.sh serves (./setup-mxfp4.sh drives it for you, or
convert-blend-mxfp4.sh calls it right after its Quark pass).

It is not optional: AMD ships the MTP head bf16 but names it in neither `exclude`
nor `layer_quant_config`, so vLLM's quark config falls through to
`global_quant_config` (mxfp4) for `mtp.*`, builds a packed uint8 weight of half
the input width, and dies loading the full-width bf16 tensor into it --

    AssertionError: Attempted to load weight (torch.Size([5120, 10240]))
                    into parameter (torch.Size([5120, 5120]))

-- before any of the quality argument below comes into play. Writing an explicit
`layer_quant_config` for the eight MTP projections is what makes the head
loadable at all.

MXFP4 on the drafter failed twice: plain RTN cost acceptance 2.5 -> 2.21, and AWQ
calibration did not rescue it (0-5% error improvement; the alpha search chose
a=0.1-0.2, and a=0.0 for mtp.fc, because MXFP4's per-32 E8M0 block exponent
already does most of what per-channel scaling would). The error is intrinsic to
4 bits, and for a drafter accuracy IS throughput.

FP8 trades a smaller bandwidth win for a much smaller error: e4m3 per-channel is
~2-3% relative versus MXFP4's ~11.6%. The drafter is 34% of decode weight traffic
at n=8, so fp8 removes ~17% of total decode traffic instead of 25% -- but should
actually hold acceptance.

vLLM's quark config supports this natively: `layer_quant_config` is matched with
fnmatch, and QuarkW8A8Fp8 wants weight fp8_e4m3 static per_channel + input_tensors
fp8_e4m3 dynamic (so no input_scale is stored). Explicit layer names are used
rather than a `*q_proj` glob, which would also match the body's 64 layers.

Memory: the input may be a single `model.safetensors` or a sharded export with an
index. All tensor metadata is read up front; payloads are streamed in bounded
chunks, so this never holds more than one chunk plus the eight MTP weights in RAM.
The output is always a single `model.safetensors` (any stale index is removed).
"""
import json
import os
import pathlib
import shutil
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:  # quant_defaults.py lives beside fp8_mtp.py in some trees ...
    from quant_defaults import DEFAULT_MTP_EXCLUDE, DEFAULT_MTP_KEEP_IN_EXCLUDE  # noqa: E402
except ImportError:  # ... and under the aijuus package in this repo (repo root is on sys.path)
    from aijuus.quant_defaults import DEFAULT_MTP_EXCLUDE, DEFAULT_MTP_KEEP_IN_EXCLUDE  # noqa: E402

# Checked before importing torch, so a bare invocation on a host without it still
# explains itself.
if len(sys.argv) != 3:
    sys.exit(f"usage: {pathlib.Path(sys.argv[0]).name} <src-checkpoint> <dst-checkpoint>\n"
             "  src  a Quark AWQ MXFP4 snapshot, e.g. the directory under\n"
             "       ~/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/\n"
             "  dst  where to write it, e.g. $MODELS/Qwen3.8-27B-MXFP4-mtpfp8\n"
             "\n"
             "Needs torch. If the host has none, run it inside the image -- see the README.")

import torch

# Plan-driven MTP policy (see quant_plan.py). With no plan this is the historical
# behavior: rewrite the eight mtp.* linears to fp8 and drop them from exclude.
_PLAN = {}
_PLAN_PATH = os.environ.get("PLAN_QUANT_PLAN_RESOLVED")
if _PLAN_PATH and os.path.isfile(_PLAN_PATH):
    _PLAN = json.loads(pathlib.Path(_PLAN_PATH).read_text()).get("resolved", {})

MTP = _PLAN.get("plan_mtp_module_list") or [
    "mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj"]
MTP_REWRITE = bool(_PLAN.get("plan_mtp_rewrite", True))
if _PLAN:
    # A plan is authoritative, including an explicit empty final-MTP list.
    KEEP_MTP_IN_EXCLUDE = bool(_PLAN.get("plan_mtp_keep_in_exclude", False))
    FINAL_MTP_ENTRIES = list(_PLAN.get("plan_final_mtp_entries") or [])
else:
    # Plan-free default matches AMD's Standard checkpoint: the final config keeps
    # the full weight-scoped mtp.* set even though it is load-inert.
    KEEP_MTP_IN_EXCLUDE = DEFAULT_MTP_KEEP_IN_EXCLUDE
    FINAL_MTP_ENTRIES = list(DEFAULT_MTP_EXCLUDE)
RETAIN_ALGO_CONFIG = bool(_PLAN.get("plan_retain_algo_config", False))
PLAN_VERSION = _PLAN.get("plan_version")
FP8_MAX = 448.0
TDT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "F64": torch.float64, "U8": torch.uint8}
SDT = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16",
       torch.float16: "F16", torch.float8_e4m3fn: "F8_E4M3"}
COPY_CHUNK = 32 << 20


def spec(dtype, dynamic, qscheme, ch_axis):
    return {"block_size": None, "ch_axis": ch_axis, "dtype": dtype, "enable_buffer_reuse": False,
            "group_size": None, "is_dynamic": dynamic, "is_scale_quant": False,
            "max_input_numel": 4194304, "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver" if qscheme == "per_channel"
                            else "PerTensorMinMaxObserver",
            "qscheme": qscheme, "round_method": "half_even", "scale_calculation_mode": None,
            "scale_format": None, "scale_type": "float", "symmetric": True}


FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
           "weight": spec("fp8_e4m3", False, "per_channel", 0),
           "input_tensors": spec("fp8_e4m3", True, "per_tensor", -1)}


def read_header(p):
    """Return (header_json, base_offset) for a safetensors file."""
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def load_entry(src, entry):
    name, shard, base, dtype, shape, s, e = entry
    if dtype not in TDT:
        sys.exit(f"ERROR: {name}: cannot decode dtype {dtype!r} (expected a bf16/fp16 MTP weight)")
    with open(src / shard, "rb") as f:
        f.seek(base + s)
        buf = bytearray(f.read(e - s))
    return torch.frombuffer(buf, dtype=TDT[dtype]).reshape(shape)


def discover(src):
    """Return (ordered entry list, set of names, shard filename list)."""
    idx = src / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text())["weight_map"]
        shards = sorted(set(wm.values()))
    else:
        shards = ["model.safetensors"] if (src / "model.safetensors").is_file() else []
    if not shards:
        sys.exit(f"ERROR: {src}: no model.safetensors or model.safetensors.index.json found")

    entries, names = [], set()
    for shard in shards:
        if not (src / shard).is_file():
            sys.exit(f"ERROR: {src}: index references missing shard {shard}")
        hdr, base = read_header(src / shard)
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            s, e = m["data_offsets"]
            entries.append((name, shard, base, m["dtype"], m["shape"], s, e))
            names.add(name)
    return entries, names, shards


def main(src_dir, dst_dir):
    src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    entries, names, shards = discover(src)
    print(f"source: {src}  ({len(entries)} tensors, {len(shards)} shard(s))")

    by_name = {e[0]: e for e in entries}
    new = {}
    if MTP_REWRITE:
        missing = [m + ".weight" for m in MTP if m + ".weight" not in names]
        if missing:
            sys.exit("ERROR: source is missing MTP weight(s):\n  " + "\n  ".join(missing))
        print(f"{'tensor':34s} {'shape':>18} {'rel err':>9}   (MXFP4 was ~0.116)")
        for name in MTP:
            key = name + ".weight"
            dtype = by_name[key][3]
            if dtype not in ("BF16", "F16", "F32"):
                sys.exit(f"ERROR: {key} is {dtype}, expected bf16/fp16/fp32. This looks like an "
                         "already-rewritten (fp8) or body-quantized checkpoint; refusing to requantize.")
            w = load_entry(src, by_name[key]).float()
            amax = w.abs().amax(dim=1).clamp(min=1e-12)              # per output channel
            s = (amax / FP8_MAX).float()
            q = (w / s.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            rel = ((q.float() * s.unsqueeze(1) - w).norm() / w.norm()).item()
            print(f"{name:34s} {str(tuple(w.shape)):>18} {rel:9.4f}")
            new[key] = q
            new[name + ".weight_scale"] = s
    else:
        print("MTP policy: bf16 -- no rewrite, MTP is copied unchanged and stays excluded")

    # ---- build the output plan (single file, deterministic order) ----
    out_hdr, plan, off = {}, [], 0
    for name, shard, base, dtype, shape, s, e in entries:
        if name in new:
            t = new[name]
            nb = t.numel() * t.element_size()
            out_hdr[name] = {"dtype": SDT[t.dtype], "shape": list(t.shape),
                             "data_offsets": [off, off + nb]}
            plan.append(("new", raw(t), None, 0, 0, nb))
        else:
            nb = e - s
            out_hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nb]}
            plan.append(("copy", None, shard, base + s, nb, nb))
        off += nb
    for name, t in new.items():
        if name not in out_hdr:
            nb = t.numel() * t.element_size()
            out_hdr[name] = {"dtype": SDT[t.dtype], "shape": list(t.shape),
                             "data_offsets": [off, off + nb]}
            plan.append(("new", raw(t), None, 0, 0, nb))
            off += nb
    out_hdr["__metadata__"] = {"format": "pt"}

    blob = json.dumps(out_hdr).encode()
    blob += b" " * ((8 - (len(blob) % 8)) % 8)

    # A stale sharded index must not survive beside the single output file.
    (dst / "model.safetensors.index.json").unlink(missing_ok=True)
    outf = dst / "model.safetensors"
    handles = {sh: open(src / sh, "rb") for sh in shards}
    try:
        with open(outf, "wb") as fout:
            fout.write(struct.pack("<Q", len(blob)))
            fout.write(blob)
            for kind, data, shard, pos, nb, _ in plan:
                if kind == "new":
                    fout.write(data)
                else:
                    fin = handles[shard]
                    fin.seek(pos)
                    left = nb
                    while left:
                        chunk = fin.read(min(left, COPY_CHUNK))
                        if not chunk:
                            sys.exit(f"ERROR: short read from {shard}")
                        fout.write(chunk)
                        left -= len(chunk)
    finally:
        for f in handles.values():
            f.close()
    print(f"\nwrote {outf} ({outf.stat().st_size / 2**30:.2f} GiB)")

    cfg = json.loads((src / "config.json").read_text())
    qc = cfg["quantization_config"]
    if qc.get("quant_method") != "quark":
        print(f"WARNING: source quant_method={qc.get('quant_method')!r}, expected 'quark'")
    # vLLM's Quark loader maps EVERY quantization_config value through its
    # WeightsMapper. A list value is assumed to be a list of strings (apply_list
    # -> _map_name -> key.endswith), so a list of dicts crashes with "'dict'
    # object has no attribute 'endswith'". Quark's AWQ export leaves algo_config
    # as a list of dicts; it is calibration metadata only and AMD's own export
    # ships it as null, so null it here.
    if RETAIN_ALGO_CONFIG:
        print("WARNING: retaining algo_config; vLLM's WeightsMapper may crash on a "
              "list-of-dicts value (see the comment this replaced)")
    else:
        # vLLM's Quark loader maps EVERY quantization_config value through its
        # WeightsMapper. A list value is assumed to be a list of strings (apply_list
        # -> _map_name -> key.endswith), so a list of dicts crashes with "'dict'
        # object has no attribute 'endswith'". Quark's AWQ export leaves algo_config
        # as a list of dicts; it is calibration metadata only and AMD's own export
        # ships it as null, so null it here.
        qc["algo_config"] = None

    if MTP_REWRITE:
        # The Quark pass left MTP unquantized (excluded); it is now fp8. Drop any
        # MTP entries from exclude so it loads through layer_quant_config, then
        # optionally re-add the reference checkpoint's own (load-inert) entries so
        # an imported config can be matched exactly.
        exclude = [e for e in qc.get("exclude", []) if e not in MTP and "mtp" not in e]
        if KEEP_MTP_IN_EXCLUDE:
            for e in FINAL_MTP_ENTRIES:
                if e not in exclude:
                    exclude.append(e)
        qc["exclude"] = exclude
        # Merge, do not replace: a KV-cache-calibrated Quark export carries
        # "*k_proj"/"*v_proj" entries here (written by LLMTemplate._set_kv_cache_config)
        # that vLLM's QuarkConfig.from_config cross-checks against export.kv_cache_group.
        # Replacing them would leave kv_cache_group non-empty with no matching
        # layer_quant_config and vLLM would refuse to load the checkpoint.
        qc["layer_quant_config"] = {
            **qc.get("layer_quant_config", {}),
            **{name: FP8_CFG for name in MTP},
        }
    else:
        # MTP stays bf16, so it must be excluded by exact module name (vLLM's
        # exclude matcher is exact or `re:`, never a glob; quant_plan.py kept the
        # module names out of the Quark pass only).
        exclude = list(qc.get("exclude", []))
        for name in MTP:
            if name not in exclude:
                exclude.append(name)
        qc["exclude"] = exclude

    if PLAN_VERSION:
        qc["version"] = PLAN_VERSION

    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"exclude {len(qc['exclude'])} entries; "
          f"layer_quant_config {sum('mtp' in k for k in qc.get('layer_quant_config', {}))} mtp layers")
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json", "chat_template.jinja"):
        src_n = src / n
        if not src_n.exists():
            continue
        dest_n = dst / n
        # The upstream tokenizer.json is mode 444; a re-run would fail to
        # overwrite the existing read-only copy, so drop it first.
        if dest_n.exists():
            try:
                dest_n.unlink()
            except OSError:
                os.chmod(dest_n, 0o644)
                dest_n.unlink()
        shutil.copy(src_n, dest_n)


main(sys.argv[1], sys.argv[2])
