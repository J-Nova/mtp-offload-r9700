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
    # M1 fix: read the pinned CPU sources of truth and stage with a host gather + H2D, instead of a
    # torch gather over the UVA host-mapped `all_token_ids` (an engine-only op never exercised by the
    # standalone repro; a torch gather over the 160k-column UVA host aperture is the prime suspect for
    # the gfx1201 HSA fault). R is 1..8, so the H2D is small.
    _idx_np = idx.cpu().numpy()
    n_np = st.num_computed_tokens_np[_idx_np].astype(np.int32)
    nmax = int(n_np.max())
    if nmax < 3:
        return base_tokens
    _uva = getattr(st.all_token_ids, "_uva_buf", None)
    if _uva is not None:
        ctx = _uva.cpu.index_select(0, idx.cpu()).to(dev)
    else:
        ctx = st.all_token_ids.gpu.index_select(0, idx)
    # M1 fix: never let n exceed the row width (the kernels' suffix/gather loads are unmasked above ML).
    n_np = np.minimum(n_np, int(ctx.shape[1])).astype(np.int32)
    n_gpu = torch.from_numpy(n_np).to(dev)
    if nmax > _RAD_NGRAM_WINDOW_FROM:
        base_np = np.maximum(0, n_np - _RAD_NGRAM_WINDOW).astype(np.int32)
    else:
        base_np = np.zeros(R, dtype=np.int32)
    # patch_dynamic_depth: matcher is run PER ROW with B=1. The B>=2 launch path faults on gfx1201
    # with HSA_STATUS_ERROR_EXCEPTION exactly at the bs=1->bs=2 transition, independent of the draft
    # row width (the fixed-width tail still faulted), so the bug is in match_gpu's B>=2 kernels, not
    # the width. B=1 is proven safe; loop rows and share one row-sized buffer. See WORKLOG cont.28.
    # M1 fix: gpu._nblk expects the window SIZE, not the base. The engine passed base_np here, which
    # inflated the scan grid ~9x (n/W blocks instead of W/512) and drove q far past ML.
    _win = int(_RAD_NGRAM_WINDOW) if nmax > _RAD_NGRAM_WINDOW_FROM else 0
    nblks = [gpu._nblk(int(n_np[i]), _win) for i in range(R)]
    ncmax = 2 * max(nblks)
    buf1 = getattr(runner, "_rad_ngram_bufs1", None)
    if (
        buf1 is None
        or buf1.get("nc") != ncmax
        or buf1["pack"].shape[1] != 2 * cap + gpu._META
    ):
        buf1 = gpu.make_match_buffers(1, cap, ncmax, dev)
        runner._rad_ngram_bufs1 = buf1
    pks = []
    for _i in range(R):
        buf1["base"].copy_(
            torch.from_numpy(np.asarray([base_np[_i]], dtype=np.int32)).to(dev)
        )
        pk_i = gpu.match_gpu(
            ctx[_i : _i + 1], n_gpu[_i : _i + 1], cap, int(n_np[_i]),
            buf1["base"], 2 * nblks[_i], False, buf1, gpu._MAXL,
        )
        pks.append(pk_i.cpu().numpy())
    pk = np.concatenate(pks, axis=0)
    cont1 = pk[:, :cap]
    cont2 = pk[:, cap : 2 * cap]
    meta = pk[:, 2 * cap :]
    clen1, mlen1 = meta[:, 0], meta[:, 1]
    clen2, mlen2 = meta[:, 2], meta[:, 3]
    mtp = base_tokens.cpu().numpy()
    rows = []
    W = K
    ext_rows = ext_toks = 0
    for i in range(R):
        di = [int(x) for x in mtp[i, : min(K, cap)]]
        use = 0
        if _RAD_NGRAM_STRONG > 0 and mlen1[i] >= _RAD_NGRAM_STRONG and clen1[i] > 0:
            use = 1
        elif _RAD_NGRAM_STRONG > 0 and mlen2[i] >= _RAD_NGRAM_STRONG and clen2[i] > 0:
            use = 2
        if use:
            c = cont1[i] if use == 1 else cont2[i]
            cl = int(clen1[i]) if use == 1 else int(clen2[i])
            # patch_dynamic_depth: FIXED-WIDTH tail. The row width must equal the scheduled depth K.
            # Appending the continuation past K (the old `di += c[kk:cl]`, up to num_speculative_steps)
            # corrupts the bs>=2 verify/candidate path -> HSA_STATUS_ERROR_EXCEPTION (WORKLOG cont.28).
            # So only take the n-gram when its recorded continuation can fill the whole depth K, and
            # then replace the K MTP drafts with it. Width stays exactly K.
            if cl >= K:
                di = [int(x) for x in c[:K]]
                ext_rows += 1
                ext_toks += K
        if not di and len(mtp[i]):
            di = [int(mtp[i, 0])]
        rows.append(di)
    W = K
    cnt = getattr(runner, "_rad_ngram_rows", 0) + R
    runner._rad_ngram_rows = cnt
    if cnt // 500 != (cnt - R) // 500:
        print(f"[ngram] rows={cnt} extended_rows={getattr(runner, '_rad_ngram_ext_rows', 0) + ext_rows} "
              f"appended={getattr(runner, '_rad_ngram_ext_toks', 0) + ext_toks}", file=_rad_sys.stderr)
    runner._rad_ngram_ext_rows = getattr(runner, "_rad_ngram_ext_rows", 0) + ext_rows
    runner._rad_ngram_ext_toks = getattr(runner, "_rad_ngram_ext_toks", 0) + ext_toks
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
