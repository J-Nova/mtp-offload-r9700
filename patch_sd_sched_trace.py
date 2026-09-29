#!/usr/bin/env python3
"""Diagnostic overlay: prove whether native dynamic SD actually reaches the scheduler.

The A/B for `num_speculative_tokens_per_batch_size` showed ON ~= OFF. That can mean either (a) the
schedule is applied but the speculator's fixed draft depth makes it inert, or (b) the schedule is
never consulted. This prints the scheduler's chosen K once per distinct (batch_size, K) pair so the
log answers it directly.

Edits `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`, right after
`num_spec_tokens_to_schedule` is computed. Idempotent; not a correctness fix (diagnostic only)."""
import sys
import sysconfig
from pathlib import Path

MARK = "patch_sd_sched_trace"

F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
src = F.read_text()
if MARK in src:
    print("[sd-trace] scheduler.py already instrumented")
    sys.exit(0)

OLD = (
    "        num_spec_tokens_to_schedule = self.num_spec_tokens\n"
    "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
    "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
    "                len(num_scheduled_tokens)\n"
    "            ]\n"
)
NEW = (
    "        num_spec_tokens_to_schedule = self.num_spec_tokens\n"
    "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
    "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
    "                len(num_scheduled_tokens)\n"
    "            ]\n"
    "        # patch_sd_sched_trace: log each distinct (bs, chosen K) once\n"
    "        if self.dynamic_sd_lookup is not None:\n"
    "            _k = (len(num_scheduled_tokens), num_spec_tokens_to_schedule)\n"
    "            _seen = getattr(self, '_radiance_sd_seen', None)\n"
    "            if _seen is None:\n"
    "                _seen = self._radiance_sd_seen = set()\n"
    "            if _k not in _seen:\n"
    "                _seen.add(_k)\n"
    "                import sys as _sys\n"
    "                print(f'[sd-trace] bs={_k[0]} nspec_sched={_k[1]} static={self.num_spec_tokens} lookup={self.dynamic_sd_lookup}', file=_sys.stderr)\n"
)
n = src.count(OLD)
if n != 1:
    print(f"[sd-trace] anchor found {n} times, NOT applied", file=sys.stderr)
    sys.exit(1)
F.write_text(src.replace(OLD, NEW))
print("[sd-trace] applied: scheduler num_spec_tokens_to_schedule trace")
