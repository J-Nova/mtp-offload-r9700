#!/usr/bin/env python3
"""Local patch (ours, 2026-09-29): fix vLLM 0.29's dynamic-SD branch in the speculator cudagraph manager.

`vllm/v1/worker/gpu/cudagraph_utils.py::CudaGraphManager._init_candidates` expands the candidate
decode-query lengths for native dynamic speculative decoding as

    num_new_sampled_tokens_per_step = self.decode_query_len - self.vllm_config.num_speculative_tokens
    decode_query_lens = {num_spec + num_new_sampled_tokens_per_step for num_spec in dense_schedule[1:]}

That assumes the manager is the MAIN model runner, whose `decode_query_len` is `num_speculative_tokens
+ 1` (so the offset is +1). But `AutoRegressiveSpeculator.init_cudagraph_manager` builds its per-step
DECODE manager with `decode_query_len=1` (draft positions > 0 are one token per request). There the
offset becomes `1 - num_speculative_tokens` (e.g. -7 at SPEC=8), the set becomes {-2,-1,0,1}, and
`round_up(num_tokens, 0)` raises ZeroDivisionError at startup:

    speculator.py:143 init_cudagraph_manager -> cudagraph_utils.py:254 _init_candidates
    -> round_up(num_tokens, decode_query_len) -> ZeroDivisionError

Found 2026-09-29 enabling `num_speculative_tokens_per_batch_size` on the MTP path (V2 runner).

Patch: keep only positive query lengths and fall back to the manager's own `decode_query_len` when the
set is empty. For the main runner the set is unchanged (num_spec+1 >= 1); for the speculator's decode
manager it collapses to {1}, which is exactly the non-dynamic behaviour. Idempotent.

STATUS: NOT applied by the entrypoint. Native dynamic SD was measured to be inert on this stack (the
V2 speculator's draft depth is fixed at num_speculative_tokens regardless of the schedule; see WORKLOG
cont. 23), so the entrypoint plumbing and this invocation were removed. Kept as the reference fix in
case `num_speculative_tokens_per_batch_size` is ever re-armed (any enabling path MUST apply this or the
speculator decode cudagraph manager raises ZeroDivisionError at startup)."""
import sys
import sysconfig
from pathlib import Path

MARK = "patch_dynamic_sd_cudagraph"

F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "v1" / "worker" / "gpu" / "cudagraph_utils.py"
src = F.read_text()
if MARK in src:
    print(f"[dynamic-sd-cg] cudagraph_utils.py already applied")
    sys.exit(0)

OLD = (
    "            decode_query_lens = sorted(\n"
    "                {\n"
    "                    num_spec + num_new_sampled_tokens_per_step\n"
    "                    for num_spec in dense_schedule[1:]\n"
    "                }\n"
    "            )\n"
)
NEW = (
    "            # patch_dynamic_sd_cudagraph.py: the speculator's per-step decode manager is built\n"
    "            # with decode_query_len=1, so the offset above goes negative and the set can contain\n"
    "            # 0 (round_up -> ZeroDivisionError). Keep positive lengths; fall back to the manager's\n"
    "            # own decode_query_len when the set is empty (the speculator decode manager -> {1}).\n"
    "            decode_query_lens = sorted(\n"
    "                _q for _q in {\n"
    "                    num_spec + num_new_sampled_tokens_per_step\n"
    "                    for num_spec in dense_schedule[1:]\n"
    "                } if _q >= 1\n"
    "            ) or [self.decode_query_len]\n"
)
n = src.count(OLD)
if n != 1:
    print(f"[dynamic-sd-cg] anchor found {n} times, NOT applied", file=sys.stderr)
    sys.exit(1)
F.write_text(src.replace(OLD, NEW))
print("[dynamic-sd-cg] applied: positive decode_query_lens (+ decode_query_len fallback) in _init_candidates")
