#!/usr/bin/env python3
"""RADIANCE dynamic draft depth + n-gram tail for the V2 model runner (vLLM 0.29). v2.

DEPTH (RADIANCE_DYNAMIC_DEPTH=1): the V2 runner ignores SchedulerOutput.num_spec_tokens_to_schedule,
so the speculator derives K from num_speculative_tokens_per_batch_size indexed by the scheduled
request count, forces the non-fused per-step decode loop, and bounds the loop to K. Measured optimum:
bs 1-2 -> 5, bs 3-8 -> 4 (+10-17% over static k=8).

NGRAM (RADIANCE_DRAFT_NGRAM=1): after the MTP drafts, append verbatim continuations from the
request's own context where a long suffix repeats (lossless -- the target verifies). This is the
additive half of the legacy controller; the per-slot tau gate is NOT ported (the graphed decode loop
means its early-stop is subsumed by the batch-level K).

Both gated at runtime. Idempotent (marker patch_dynamic_depth); every edited file must ast.parse."""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_dynamic_depth"
SP = Path(sysconfig.get_paths()["purelib"])
RUNNER = SP / "vllm/v1/worker/gpu/model_runner.py"
SPEC = SP / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"


SPEC_BLOCK = (
    "        # RADIANCE dynamic depth (patch_dynamic_depth.py): choose the draft depth K for this\n"
    "        # batch from num_speculative_tokens_per_batch_size, indexed by the scheduled request\n"
    "        # count. The V2 runner does not forward SchedulerOutput.num_spec_tokens_to_schedule, so\n"
    "        # derive it here from the speculator's own config.\n"
    "        _rd_sched = getattr(self, \"_radiance_dyn_sched\", None)\n"
    "        if _rd_sched is None:\n"
    "            _rd_sched = None\n"
    "            _rd_sc = getattr(self.vllm_config, \"speculative_config\", None)\n"
    "            if _rd_sc is not None and getattr(\n"
    "                _rd_sc, \"num_speculative_tokens_per_batch_size\", None\n"
    "            ):\n"
    "                try:\n"
    "                    from vllm.v1.spec_decode.dynamic.utils import (\n"
    "                        build_dynamic_sd_schedule_lookup as _rd_build,\n"
    "                    )\n"
    "                    _rd_sched = _rd_build(\n"
    "                        _rd_sc.num_speculative_tokens_per_batch_size,\n"
    "                        vllm_max_batch_size=self.max_num_reqs,\n"
    "                        vllm_num_speculative_tokens=self.num_speculative_steps,\n"
    "                    )\n"
    "                except Exception:\n"
    "                    _rd_sched = None\n"
    "            self._radiance_dyn_sched = _rd_sched\n"
    "        _rd_k = self.num_speculative_steps\n"
    "        if _rd_sched is not None and not self.use_fused_multi_step_decode:\n"
    "            _rd_k = max(\n"
    "                1,\n"
    "                min(\n"
    "                    self.num_speculative_steps,\n"
    "                    _rd_sched[num_reqs]\n"
    "                    if num_reqs < len(_rd_sched)\n"
    "                    else self.num_speculative_steps,\n"
    "                ),\n"
    "            )\n"
    "        self._radiance_eff_k = _rd_k\n"
    "        self._radiance_dyn_k = _rd_k\n"
    "        _rd_n = getattr(self, \"_rd_calls\", 0)\n"
    "        if _rd_n < 24:\n"
    "            self._rd_calls = _rd_n + 1\n"
    "            import sys as _rdsys\n"
    "            print(f\"[dyn-depth] propose#{_rd_n} k={_rd_k} num_reqs={num_reqs} \"\n"
    "                  f\"fused={self.use_fused_multi_step_decode} \"\n"
    "                  f\"sched={_rd_sched is not None}\", file=_rdsys.stderr)\n"
)

FORCE_NONFUSED = (
    "        self.use_fused_multi_step_decode = not unsupported_backends\n"
    "        # patch_dynamic_depth: dynamic depth needs the per-step (non-fused) decode loop, because\n"
    "        # the fused graph bakes the full depth into one capture. Force it off when a schedule is set\n"
    "        # so every instance (including whichever serves requests) is depth-parameterisable.\n"
    "        _rd_sc2 = getattr(self.vllm_config, \"speculative_config\", None)\n"
    "        if _rd_sc2 is not None and getattr(\n"
    "            _rd_sc2, \"num_speculative_tokens_per_batch_size\", None\n"
    "        ):\n"
    "            self.use_fused_multi_step_decode = False\n"
)

# runner: replace the plain assignment with the depth-slice (+ optional ngram extend) assembly.
RUNNER_ASSIGN_OLD = (
    "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
)
RUNNER_ASSIGN_NEW = (
    "            # patch_dynamic_depth: keep only K drafts (speculator depth), optionally extend with the\n"
    "            # verbatim n-gram tail, then store the assembled rows and hand them to the handler.\n"
    "            _rd_k2 = getattr(self.speculator, \"_radiance_dyn_k\", 0) or self.num_speculative_steps\n"
    "            _rd_k2 = max(1, min(int(_rd_k2), self.num_speculative_steps))\n"
    "            if _RAD_NGRAM:\n"
    "                draft_tokens = _radiance_ngram_extend(\n"
    "                    self, input_batch, draft_tokens[:, :_rd_k2]\n"
    "                )\n"
    "            else:\n"
    "                draft_tokens = draft_tokens[:, :_rd_k2]\n"
    "            _rd_w = int(draft_tokens.shape[1])\n"
    "            self.req_states.draft_tokens[input_batch.idx_mapping, :_rd_w] = draft_tokens\n"
    "            self._radiance_last_draft = draft_tokens\n"
)

RUNNER_HANDLER_OLD = (
    "            self.draft_tokens_handler.set_draft_tokens(\n"
    "                input_batch,\n"
    "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
    "            )\n"
)
RUNNER_HANDLER_NEW = (
    "            _rd_d = getattr(self, \"_radiance_last_draft\", None)\n"
    "            self.draft_tokens_handler.set_draft_tokens(\n"
    "                input_batch,\n"
    "                _rd_d\n"
    "                if _rd_d is not None\n"
    "                else self.req_states.draft_tokens[input_batch.idx_mapping],\n"
    "            )\n"
)

RUNNER_TAIL = '''

# ---- RADIANCE dynamic draft depth + n-gram tail (patch_dynamic_depth.py) ----------------------
import os as _rad_os
import sys as _rad_sys

_RAD_NGRAM = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM", "0") == "1"
_RAD_NGRAM_STRONG = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_STRONG", "8"))
_RAD_NGRAM_WINDOW_FROM = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_WINDOW_FROM", "32768"))
_RAD_NGRAM_WINDOW = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_WINDOW", "16384"))
# RADIANCE_DRAFT_NGRAM_ADAPT: per-request productivity gate (see ACTIONABLE-AB-PLAN A2/A3).
_RAD_NGRAM_ADAPT = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_ADAPT", "0") == "1"
_RAD_NGRAM_EMA_ALPHA = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_EMA_ALPHA", "0.10"))
_RAD_NGRAM_EMA_MIN = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_EMA_MIN", "0.02"))
_RAD_NGRAM_WARMUP = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_WARMUP", "2"))
_RAD_NGRAM_PROBE = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_PROBE", "32"))
# Variant B: independent n-gram depth M (>K) at low concurrency, decoupled gate. RADIANCE_DRAFT_NGRAM_DEPTH=0
# keeps today's fixed-width behavior; RADIANCE_DRAFT_NGRAM_BS_MAX bounds where M applies.
_RAD_NGRAM_DEPTH = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_DEPTH", "0"))
_RAD_NGRAM_BS_MAX = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_BS_MAX", "2"))
_RAD_NGRAM_MIN = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_MIN", "3"))
_RAD_NGRAM_HIST = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_HIST", "0") == "1"
# C1/C2: match all armed rows in one batched match_gpu launch with a single D2H, instead of one
# launch + one sync per row. Safe at B>=1 now that the _nblk (window size) grid bug is fixed; set
# RADIANCE_DRAFT_NGRAM_BATCH=0 to fall back to the proven per-row path.
_RAD_NGRAM_BATCH = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_BATCH", "1") == "1"


def _radiance_ngram_extend(runner, input_batch, base_tokens):
    """Append verbatim n-gram continuations to the MTP drafts (lossless; the target verifies).

    base_tokens [R, K] are the K MTP drafts for this step. For each row whose best longest-suffix
    match within the request's own context is >= _RAD_NGRAM_STRONG, append the recorded continuation
    after the MTP prefix. Returns a [R, W] int64 GPU tensor (W = max row length, <= K+cap) padded with
    -1, the vLLM invalid-draft placeholder."""
    import numpy as np
    import torch
    import radiance_draft_gpu as gpu

    st = runner.req_states
    idx = input_batch.idx_mapping.to(torch.int64)
    R = int(base_tokens.shape[0])
    K = int(base_tokens.shape[1])
    cap = int(runner.num_speculative_steps)
    dev = base_tokens.device
    if R == 0 or K <= 0:
        return base_tokens
    # RADIANCE_DRAFT_NGRAM_ADAPT: skip the (costly) per-row matcher for rows that have not been
    # productive recently; re-arm on warmup, a periodic probe, or after a hit. When the whole batch is
    # cold, return early with no gather/launches/syncs. Lossless: cold rows keep their MTP draft.
    _rids = list(input_batch.req_ids)
    _nst = getattr(runner, "_rad_ngram_state", None)
    if _nst is None:
        _nst = {}
        runner._rad_ngram_state = _nst
    _run = [True] * R
    if _RAD_NGRAM_ADAPT:
        _keep = set(_rids)
        for _k in [k for k in _nst if k not in _keep]:
            del _nst[_k]
        for _i in range(R):
            _s = _nst.get(_rids[_i])
            if _s is None:
                _s = [0.0, 0, 0, False]  # ema, obs, probe, latch
                _nst[_rids[_i]] = _s
            _arm = (_s[1] < _RAD_NGRAM_WARMUP or _s[3]
                    or _s[0] >= _RAD_NGRAM_EMA_MIN or _s[2] >= _RAD_NGRAM_PROBE)
            _run[_i] = _arm
            if not _arm:
                _s[2] += 1
        if not any(_run):
            return base_tokens
    n_gpu = st.num_computed_tokens.gpu.index_select(0, idx).to(torch.int32)
    n_np = n_gpu.cpu().numpy()
    nmax = int(n_np.max())
    if nmax < 3:
        return base_tokens
    ctx = st.all_token_ids.gpu.index_select(0, idx)
    if nmax > _RAD_NGRAM_WINDOW_FROM:
        base_np = np.maximum(0, n_np - _RAD_NGRAM_WINDOW).astype(np.int32)
    else:
        base_np = np.zeros(R, dtype=np.int32)
    # M1 fix: gpu._nblk expects the window SIZE, not the base. The engine passed base_np here, which
    # inflated the scan grid ~9x (n/W blocks instead of W/512) and drove q far past ML. With the grid
    # correct match_gpu is safe at B>=1, so the arm rows are matched in ONE batched launch with a
    # single D2H (C1/C2), replacing R launches and R per-row syncs. The batch grid uses the max block
    # count across rows; a shorter row's extra blocks are masked (alive = q < n-1) and emit no key.
    _win = int(_RAD_NGRAM_WINDOW) if nmax > _RAD_NGRAM_WINDOW_FROM else 0
    nblks = [gpu._nblk(int(n_np[i]), _win) for i in range(R)]
    pk = np.zeros((R, 2 * cap + gpu._META), dtype=np.int64)
    if _RAD_NGRAM_BATCH:
        _armed = [i for i in range(R) if _run[i]]
        if _armed:
            _bn = len(_armed)
            _nc = 2 * max(nblks[i] for i in _armed)
            bufN = getattr(runner, "_rad_ngram_bufN", None)
            if (
                bufN is None
                or bufN.get("B") != _bn
                or bufN.get("nc") != _nc
                or bufN["pack"].shape[1] != 2 * cap + gpu._META
            ):
                bufN = gpu.make_match_buffers(_bn, cap, _nc, dev)
                bufN["B"] = _bn
                runner._rad_ngram_bufN = bufN
            if _bn == R:
                _ctx_s, _n_s = ctx, n_gpu
            else:
                _ai = torch.tensor(_armed, dtype=torch.int64, device=dev)
                _ctx_s = ctx.index_select(0, _ai)
                _n_s = n_gpu.index_select(0, _ai)
            _base_s = torch.from_numpy(
                np.asarray([base_np[i] for i in _armed], dtype=np.int32)
            ).to(dev)
            _pk_s = gpu.match_gpu(
                _ctx_s, _n_s, cap, 0, _base_s, _nc, False, bufN, gpu._MAXL,
            )
            _pk_np = _pk_s.cpu().numpy()
            for _r, _i in enumerate(_armed):
                pk[_i] = _pk_np[_r]
    else:
        # Fallback: proven per-row B=1 path (RADIANCE_DRAFT_NGRAM_BATCH=0).
        ncmax = 2 * max(nblks)
        buf1 = getattr(runner, "_rad_ngram_bufs1", None)
        if (
            buf1 is None
            or buf1.get("nc") != ncmax
            or buf1["pack"].shape[1] != 2 * cap + gpu._META
        ):
            buf1 = gpu.make_match_buffers(1, cap, ncmax, dev)
            runner._rad_ngram_bufs1 = buf1
        for _i in range(R):
            if not _run[_i]:
                continue
            buf1["base"].copy_(
                torch.from_numpy(np.asarray([base_np[_i]], dtype=np.int32)).to(dev)
            )
            pk_i = gpu.match_gpu(
                ctx[_i : _i + 1], n_gpu[_i : _i + 1], cap, int(n_np[_i]),
                buf1["base"], 2 * nblks[_i], False, buf1, gpu._MAXL,
            )
            pk[_i] = pk_i.cpu().numpy()[0]
    cont1 = pk[:, :cap]
    cont2 = pk[:, cap : 2 * cap]
    meta = pk[:, 2 * cap :]
    clen1, mlen1 = meta[:, 0], meta[:, 1]
    clen2, mlen2 = meta[:, 2], meta[:, 3]
    mtp = base_tokens.cpu().numpy()
    rows = []
    W = K
    ext_rows = ext_toks = 0
    ext_flags = []
    # Variant B: independent n-gram depth M, applied only at low concurrency (R <= BS_MAX).
    _M = _RAD_NGRAM_DEPTH if (_RAD_NGRAM_DEPTH > K and R <= _RAD_NGRAM_BS_MAX) else 0
    _hist = getattr(runner, "_rad_ngram_hist", None)
    if _hist is None:
        _hist = {}
        runner._rad_ngram_hist = _hist
    for i in range(R):
        di = [int(x) for x in mtp[i, : min(K, cap)]]
        use = 0
        _ext_i = False
        _hist[("clen", int(clen1[i]))] = _hist.get(("clen", int(clen1[i])), 0) + 1
        if clen1[i] > 0:
            _hist[("mlen", int(mlen1[i]))] = _hist.get(("mlen", int(mlen1[i])), 0) + 1
        if _RAD_NGRAM_STRONG > 0 and mlen1[i] >= _RAD_NGRAM_STRONG and clen1[i] > 0:
            use = 1
        elif _RAD_NGRAM_STRONG > 0 and mlen2[i] >= _RAD_NGRAM_STRONG and clen2[i] > 0:
            use = 2
        if use:
            c = cont1[i] if use == 1 else cont2[i]
            cl = int(clen1[i]) if use == 1 else int(clen2[i])
            if _M > K:
                # variant B: gate decoupled from width -- fire on a shorter match, emit min(cl, M).
                if cl >= _RAD_NGRAM_MIN:
                    _take = min(cl, _M)
                    di = [int(x) for x in c[:_take]]
                    ext_rows += 1
                    ext_toks += _take
                    _ext_i = True
            elif cl >= K:
                # current: fixed-width tail (== scheduled depth K).
                di = [int(x) for x in c[:K]]
                ext_rows += 1
                ext_toks += K
                _ext_i = True
        if not di and len(mtp[i]):
            di = [int(mtp[i, 0])]
        rows.append(di)
        ext_flags.append(_ext_i)
    W = _M if (_M > K and ext_rows > 0) else K
    cnt = getattr(runner, "_rad_ngram_rows", 0) + R
    runner._rad_ngram_rows = cnt
    if cnt // 500 != (cnt - R) // 500:
        print(f"[ngram] rows={cnt} extended_rows={getattr(runner, '_rad_ngram_ext_rows', 0) + ext_rows} "
              f"appended={getattr(runner, '_rad_ngram_ext_toks', 0) + ext_toks}", file=_rad_sys.stderr)
        if _hist is not None:
            _hs = {f"{k[0]}{k[1]}": v for k, v in sorted(_hist.items(), key=lambda kv: (kv[0][0], kv[0][1]))}
            print(f"[ngram-hist] {_hs}", file=_rad_sys.stderr)
    runner._rad_ngram_ext_rows = getattr(runner, "_rad_ngram_ext_rows", 0) + ext_rows
    runner._rad_ngram_ext_toks = getattr(runner, "_rad_ngram_ext_toks", 0) + ext_toks
    if _RAD_NGRAM_ADAPT:
        for _i in range(R):
            _s = _nst.get(_rids[_i])
            if _s is None:
                continue
            _e = 1.0 if ext_flags[_i] else 0.0
            _s[0] = (1.0 - _RAD_NGRAM_EMA_ALPHA) * _s[0] + _RAD_NGRAM_EMA_ALPHA * _e
            _s[1] += 1
            _s[2] = 0
            _s[3] = bool(ext_flags[_i])
    out = torch.full((R, W), -1, dtype=base_tokens.dtype, device=dev)
    for i, di in enumerate(rows):
        if di:
            out[i, : len(di)] = torch.tensor(di, dtype=base_tokens.dtype, device=dev)
    return out
'''

SPEC_PAIRS = [
    ("force-nonfused",
     "        self.use_fused_multi_step_decode = not unsupported_backends\n",
     FORCE_NONFUSED),
    ("k-block",
     "        num_reqs = input_batch.num_reqs\n"
     "        max_query_len = input_batch.num_scheduled_tokens.max()\n",
     "        num_reqs = input_batch.num_reqs\n"
     + SPEC_BLOCK +
     "        max_query_len = input_batch.num_scheduled_tokens.max()\n"),
    ("loop-nonfused",
     "        slot_mappings_by_layer = None\n"
     "        for step in range(1, self.num_speculative_steps):\n"
     "            # Rebuild every step when positions advance, or just once\n",
     "        slot_mappings_by_layer = None\n"
     "        for step in range(1, getattr(self, \"_radiance_eff_k\", self.num_speculative_steps)):\n"
     "            # Rebuild every step when positions advance, or just once\n"),
    ("loop-fused",
     "        for step in range(1, self.num_speculative_steps):\n"
     "            self.current_draft_step.fill_(step)\n"
     "            self._generate_draft(\n",
     "        for step in range(1, getattr(self, \"_radiance_eff_k\", self.num_speculative_steps)):\n"
     "            self.current_draft_step.fill_(step)\n"
     "            self._generate_draft(\n"),
]

RUNNER_PAIRS = [
    ("assign", RUNNER_ASSIGN_OLD, RUNNER_ASSIGN_NEW),
    ("handler", RUNNER_HANDLER_OLD, RUNNER_HANDLER_NEW),
]


def edit(path, pairs, tail=None):
    src = path.read_text()
    if MARK in src:
        print(f"[dyn-depth] {path.name} already applied")
        return
    for tag, old, new in pairs:
        n = src.count(old)
        if n != 1:
            print(f"[dyn-depth] {path.name}: anchor {tag} matched {n}x, NOT applied", file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    if tail:
        src += tail
    ast.parse(src)
    path.write_text(src)
    print(f"[dyn-depth] applied: {path.name}")


edit(SPEC, SPEC_PAIRS)
edit(RUNNER, RUNNER_PAIRS, tail=RUNNER_TAIL)
