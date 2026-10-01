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
_RAD_NGRAM_EMA_MIN = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_EMA_MIN", "0.10"))
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
# F1/F2 trust gates (cont.79): a long exact suffix match is NOT sufficient to override the MTP draft --
# tool-schema boilerplate repeats produce long matches whose continuation (a different field name)
# the target rejects. AGREE: the tail's first token must equal MTP's drafted token. DET: the top-2
# matches must agree on the first token (two independent occurrences -> a deterministic continuation).
# Both default ON; AGREE=0 DET=0 restores the pre-fix "long match always wins" behaviour.
_RAD_NGRAM_AGREE = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_AGREE", "1") == "1"
_RAD_NGRAM_DET = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_DET", "1") == "1"
# F6 frequency gate (cont.79): when MTP disagrees or the top-2 are ambiguous, allow the override only
# if the continuation is strongly repeated -- at least _MIN_FREQ other occurrences of the same suffix
# share the first token, and the resulting probability exceeds _MIN_PROB (Arctic min_token_prob idea).
_RAD_NGRAM_FREQ = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_FREQ", "1") == "1"
_RAD_NGRAM_MIN_FREQ = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_MIN_FREQ", "1"))
_RAD_NGRAM_MIN_PROB = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_MIN_PROB", "0.5"))
# F7 (cont.85): prefix-preserving EXTENSION (cont.85). Keep the WHOLE MTP draft and APPEND the suffix
# continuation only when it agrees with the FULL MTP prefix -- monotone-safe (can never override/replace
# MTP, so it cannot regress accepted length). RADIANCE_DRAFT_NGRAM_EXT=1 to enable.
_RAD_NGRAM_EXT = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_EXT", "0") == "1"
# Arctic SuffixDecoding backend (cont.82): "triton" = our GPU matcher + F1/F2/F6 gates (default);
# "arctic" = per-request arctic_inference SuffixDecodingCache with a hybrid tau gate -- take the suffix
# draft only when its expected accepted length (score) >= tau, else keep the MTP draft.
_RAD_NGRAM_BACKEND = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_BACKEND", "triton").strip().lower()
_RAD_ARCTIC_TAU = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_TAU", "1.0"))
_RAD_ARCTIC_DEPTH = int(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_ARCTIC_DEPTH", "24"))
_RAD_ARCTIC_FACTOR = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_ARCTIC_FACTOR", "1.0"))
_RAD_ARCTIC_MIN_PROB = float(_rad_os.environ.get("RADIANCE_DRAFT_NGRAM_ARCTIC_MIN_PROB", "0.1"))
# Require the Arctic draft's first token to agree with MTP before overriding (rejects degenerate
# warmup drafts; keeps the hybrid honest). RADIANCE_DRAFT_NGRAM_ARCTIC_AGREE=0 to disable.
_RAD_ARCTIC_AGREE = _rad_os.environ.get("RADIANCE_DRAFT_NGRAM_ARCTIC_AGREE", "1") == "1"


def _radiance_ngram_extend_row(mrow, c, clen, mlen, K, cap):
    """Prefix-preserving extension (pure; unit-tested). Keep the MTP row `mrow` (length K) and, only when
    a long-enough match's continuation `c` agrees with ALL K MTP tokens, append c[K:min(clen,cap)].
    Never overrides MTP -> monotone-safe. Returns the row (length K or more)."""
    di = list(mrow)
    if int(mlen) >= _RAD_NGRAM_STRONG and int(clen) > K and len(di) >= K:
        if all(int(c[j]) == di[j] for j in range(K)):
            for t in range(K, min(int(clen), cap)):
                di.append(int(c[t]))
    return di


def _radiance_ngram_ok(mlen, clen, c0, oth_mlen, oth_clen, oth_c0, mtp0,
                       agree_other=-1, occ_other=0):
    """Trust gate for one candidate (pure; unit-tested in patch_dynamic_depth_policy_test.py).

    True only when the match is long enough, the continuation is non-empty, and (if enabled) the
    continuation's first token agrees with MTP (AGREE) and with the other candidate (DET). When MTP
    disagrees or the top-2 are ambiguous, F6 allows the override only if the continuation is strongly
    repeated (`agree_other`/`occ_other` exclude the reference occurrence). `c0`/`oth_c0` are the first
    continuation tokens (-1 = none); `mtp0` is MTP's drafted token (-1 = unknown); DET only vetoes
    against another STRONG match; STRONG<=0 keeps its kill-switch meaning."""
    if _RAD_NGRAM_STRONG <= 0 or mlen < _RAD_NGRAM_STRONG or clen <= 0:
        return False
    _fast = True
    if _RAD_NGRAM_AGREE and mtp0 >= 0 and c0 != mtp0:
        _fast = False
    if _RAD_NGRAM_DET and oth_mlen >= _RAD_NGRAM_STRONG and oth_clen > 0 and oth_c0 != c0:
        _fast = False
    if _fast:
        return True
    if _RAD_NGRAM_FREQ and agree_other >= _RAD_NGRAM_MIN_FREQ:
        return (agree_other + 1) / (occ_other + 1) >= _RAD_NGRAM_MIN_PROB
    return False


def _radiance_arctic_extend(runner, input_batch, base_tokens):
    """Arctic SuffixDecoding backend (cont.82): per-request SuffixDecodingCache + hybrid tau gate.

    Uses the real frequency-weighted suffix tree (arctic_inference) instead of the Triton matcher. For
    each request we feed prompt+generated tokens, speculate from the last `depth` tokens, and take the
    suffix draft ONLY when its expected accepted length (`score`) >= tau; otherwise the MTP draft is
    kept (hybrid: suffix for repetitive segments, neural for novel). Any error disables arctic and falls
    back to MTP (lossless)."""
    import numpy as np
    import torch
    try:
        from arctic_inference.suffix_decoding import SuffixDecodingCache
    except Exception as _e:
        print(f"[arctic] import failed, falling back to MTP: {_e!r}", file=_rad_sys.stderr)
        return base_tokens
    R = int(base_tokens.shape[0])
    K = int(base_tokens.shape[1])
    cap = int(runner.num_speculative_steps)
    dev = base_tokens.device
    if R == 0 or K <= 0:
        return base_tokens
    ib = input_batch
    idx = getattr(ib, "idx_mapping_np", None)
    if idx is None:
        idx = np.asarray(ib.idx_mapping.cpu()) if hasattr(ib.idx_mapping, "cpu") else np.asarray(ib.idx_mapping)
    rids = list(ib.req_ids)
    ncomp = getattr(ib, "num_computed_tokens_np", None)
    if ncomp is None:
        ncomp = ib.num_computed_tokens_cpu
    plen = getattr(ib, "prefill_len_np", None)
    if plen is None:
        plen = getattr(ib, "num_prompt_tokens", None)
    atoks = runner.req_states.all_token_ids.gpu
    # Vocab bound: warmup/dummy token ids can be out of range; adopting them as a draft makes the
    # verify's embedding index_select fault (HSA on gfx1201). Drop any draft containing an invalid id.
    _vocab = getattr(runner, "vocab_size", None)
    if _vocab is None:
        _mc = getattr(runner, "model_config", None)
        if _mc is not None and hasattr(_mc, "get_vocab_size"):
            _vocab = int(_mc.get_vocab_size())
    if _vocab is None:
        _vc = getattr(runner, "vllm_config", None)
        _mc2 = getattr(_vc, "model_config", None) if _vc is not None else None
        if _mc2 is not None and hasattr(_mc2, "get_vocab_size"):
            _vocab = int(_mc2.get_vocab_size())
    _vocab = int(_vocab) if _vocab else (1 << 30)

    def _cpu(rsi, a, b):
        return np.ascontiguousarray(atoks[rsi, a:b].cpu().numpy(), dtype=np.int32)

    depth = max(2, int(_RAD_ARCTIC_DEPTH))
    caches = getattr(runner, "_rad_arctic_caches", None)
    if caches is None:
        caches = {}
        runner._rad_arctic_caches = caches
    keep = set(rids)
    for _k in [k for k in caches if k not in keep]:
        try:
            caches[_k][0].stop_request(_k)
        except Exception:
            pass
        del caches[_k]
    mtp = base_tokens.cpu().numpy()
    rows = []
    W = K
    taken = toks = 0
    for i in range(R):
        rsi = int(idx[i])
        rid = rids[i]
        n = int(ncomp[i])
        di = [int(x) for x in mtp[i, : min(K, cap)]]
        got = False
        if n >= 3:
            try:
                ent = caches.get(rid)
                if ent is None:
                    c = SuffixDecodingCache(max_tree_depth=depth, max_cached_requests=-1)
                    p0 = int(plen[i]) if plen is not None else n
                    if p0 <= 0:
                        p0 = min(3, n)
                    c.start_request(rid, _cpu(rsi, 0, min(p0, n)))
                    if n > p0:
                        c.add_active_response(rid, _cpu(rsi, p0, n))
                    caches[rid] = [c, n]
                else:
                    c = ent[0]
                    if n > ent[1]:
                        c.add_active_response(rid, _cpu(rsi, ent[1], n))
                        ent[1] = n
                _kt = min(cap, K)
                d = c.speculate(rid, _cpu(rsi, max(0, n - depth), n), max_spec_tokens=_kt,
                                max_spec_factor=float(_RAD_ARCTIC_FACTOR),
                                min_token_prob=float(_RAD_ARCTIC_MIN_PROB))
                tl = [int(x) for x in d.token_ids][:K]
                if tl and float(d.score) >= float(_RAD_ARCTIC_TAU):
                    if (not _RAD_ARCTIC_AGREE) or tl[0] == di[0]:
                        # Merge into the full MTP row so the draft keeps length K with NO -1 padding
                        # (a short arctic-only row leaves -1 which the verify's embedding index_select
                        # loads as a token id -> OOB/HSA on gfx1201). Override slot j where arctic has a
                        # valid, in-vocab token; stop at the first invalid.
                        for _j in range(min(len(tl), len(di))):
                            if not (0 <= tl[_j] < _vocab):
                                break
                            di[_j] = tl[_j]
                        got = True
            except Exception as _e:
                if not getattr(runner, "_rad_arctic_err", False):
                    runner._rad_arctic_err = True
                    print(f"[arctic] disabled after error, falling back to MTP: {_e!r}", file=_rad_sys.stderr)
        if not di and mtp.shape[1]:
            di = [int(mtp[i, 0])]
        rows.append(di)
        if got:
            taken += 1
            toks += len(di)
        if len(di) > W:
            W = len(di)
    out = torch.full((R, W), -1, dtype=base_tokens.dtype, device=dev)
    for i, di in enumerate(rows):
        if di:
            out[i, : len(di)] = torch.tensor(di, dtype=base_tokens.dtype, device=dev)
    cnt = getattr(runner, "_rad_ngram_rows", 0) + R
    runner._rad_ngram_rows = cnt
    if cnt // 500 != (cnt - R) // 500:
        print(f"[arctic] rows={cnt} taken={getattr(runner, '_rad_arctic_rows', 0) + taken} "
              f"toks={getattr(runner, '_rad_arctic_toks', 0) + toks}", file=_rad_sys.stderr)
    runner._rad_arctic_rows = getattr(runner, "_rad_arctic_rows", 0) + taken
    runner._rad_arctic_toks = getattr(runner, "_rad_arctic_toks", 0) + toks
    return out


def _radiance_ngram_extend(runner, input_batch, base_tokens):
    """Append verbatim n-gram continuations to the MTP drafts (lossless; the target verifies).

    base_tokens [R, K] are the K MTP drafts for this step. For each row whose best longest-suffix
    match within the request's own context is >= _RAD_NGRAM_STRONG, append the recorded continuation
    after the MTP prefix. Returns a [R, W] int64 GPU tensor (W = max row length, <= K+cap) padded with
    -1, the vLLM invalid-draft placeholder."""
    import numpy as np
    import torch
    import radiance_draft_gpu as gpu

    if _RAD_NGRAM_BACKEND == "arctic":
        return _radiance_arctic_extend(runner, input_batch, base_tokens)

    # FREQ needs the F6 kernel; if the overlay copy in site-packages predates it, fall back to F1/F2.
    _freq = _RAD_NGRAM_FREQ and hasattr(gpu, "match_count")
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
    occ_np = np.zeros(R, dtype=np.int64)
    agree_np = np.zeros(R, dtype=np.int64)
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
            if _freq:
                bufN["occ"].zero_()
                bufN["agree"].zero_()
                gpu.match_count(_ctx_s, _n_s, bufN["key1"], _base_s,
                                bufN["occ"], bufN["agree"], _nc)
                _fo = bufN["occ"].cpu().numpy()
                _fa = bufN["agree"].cpu().numpy()
                for _r, _i in enumerate(_armed):
                    occ_np[_i] = int(_fo[_r])
                    agree_np[_i] = int(_fa[_r])
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
            if _freq:
                buf1["occ"].zero_()
                buf1["agree"].zero_()
                gpu.match_count(ctx[_i : _i + 1], n_gpu[_i : _i + 1], buf1["key1"], buf1["base"],
                                buf1["occ"], buf1["agree"], 2 * nblks[_i])
                occ_np[_i] = int(buf1["occ"].cpu().numpy()[0])
                agree_np[_i] = int(buf1["agree"].cpu().numpy()[0])
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
    _agree_hist = []
    for i in range(R):
        di = [int(x) for x in mtp[i, : min(K, cap)]]
        use = 0
        _ext_i = False
        _hist[("clen", int(clen1[i]))] = _hist.get(("clen", int(clen1[i])), 0) + 1
        if clen1[i] > 0:
            _hist[("mlen", int(mlen1[i]))] = _hist.get(("mlen", int(mlen1[i])), 0) + 1
        _mtp0 = int(mtp[i, 0]) if mtp.shape[1] else -1
        _c1_0 = int(cont1[i, 0]) if cap > 0 else -1
        _c2_0 = int(cont2[i, 0]) if cap > 0 else -1
        # F1/F2: long match alone is not enough -- require slot-0 agreement with MTP and top-2
        # determinism (see _radiance_ngram_ok).
        if _radiance_ngram_ok(int(mlen1[i]), int(clen1[i]), _c1_0,
                              int(mlen2[i]), int(clen2[i]), _c2_0, _mtp0,
                              int(agree_np[i]), int(occ_np[i])):
            use = 1
        elif _radiance_ngram_ok(int(mlen2[i]), int(clen2[i]), _c2_0,
                                int(mlen1[i]), int(clen1[i]), _c1_0, _mtp0):
            use = 2
        if _RAD_NGRAM_EXT:
            # F7: keep the MTP draft and append the suffix continuation where it agrees with the full
            # MTP prefix (monotone-safe; cannot regress accepted length).
            _pre = len(di)
            di = _radiance_ngram_extend_row(di, cont1[i], int(clen1[i]), int(mlen1[i]), K, cap)
            if len(di) > _pre:
                ext_rows += 1
                ext_toks += (len(di) - _pre)
                _ext_i = True
        elif use:
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
        if len(di) > W:
            W = len(di)
        rows.append(di)
        ext_flags.append(_ext_i)
        # F4 (cont.79): predictive-value feedback = did the n-gram OFFER a slot-0 token agreeing with
        # MTP? Measured pre-gate, so it is not self-confirming. Guard on clen>0 so a row whose matcher
        # did not run (pk zeros -> cont 0) cannot fake agreement when MTP's token id is 0.
        if (int(clen1[i]) > 0 and _c1_0 == _mtp0) or (int(clen2[i]) > 0 and _c2_0 == _mtp0):
            _agree_hist.append(1.0)
        else:
            _agree_hist.append(0.0)
    W = max(W, K)
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
            _e = _agree_hist[_i]
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
