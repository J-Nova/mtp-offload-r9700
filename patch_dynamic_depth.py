#!/usr/bin/env python3
"""RADIANCE dynamic draft depth for the V2 model runner (vLLM 0.29). v2: self-contained.

WHY. On this deployment MTP runs the V2 runner (`vllm/v1/worker/gpu/model_runner.py`) on the
NON-FUSED speculator path (`use_fused_multi_step_decode=False`; probe `fused=False adv=True`), so the
draft loop in `_multi_step_decode` is a plain Python `for step in range(1, num_speculative_steps)`
over per-step FULL-graph replays. vLLM's own dynamic SD (`num_speculative_tokens_per_batch_size`) is
read only by the V1 runner/async scheduler; the V2 runner ignores
`SchedulerOutput.num_spec_tokens_to_schedule`, so depth stays at `num_speculative_steps`.

Measured optimum is concurrency-dependent (bs 1-2 -> k=5, bs 4-8 -> k=4; k<=3 collapses), so a static
depth is always a compromise.

HOW (self-contained; v1's runner->speculator attribute hand-off proved unreliable):
  * Speculator: build the batch->K lookup from its own `vllm_config.speculative_config` and, in
    `propose`, choose K from the scheduled request count (`num_reqs`, the same index the scheduler
    uses); bound `_multi_step_decode` (and the fused loop) to K; expose the chosen K on
    `self._radiance_dyn_k` for the runner.
  * V2 runner: pass only the first K draft columns to `DraftTokensHandler` (`set_draft_tokens`
    sizes on `draft_tokens.shape[1]`, and the scheduler schedules `request.spec_token_ids` as
    returned, so a K-wide list yields K verify rows).
When `use_fused_multi_step_decode` is True the fused graph bakes full depth, so K stays at max (never
a graph mismatch).

Gated: applied when RADIANCE_DYNAMIC_DEPTH=1. Idempotent (marker `patch_dynamic_depth`); every edited
file must `ast.parse` before it is written."""
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
    "                  f\"nrap={getattr(input_batch, 'num_reqs_after_padding', -1)} \"\n"
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
    ("slice-handler",
     "            self.draft_tokens_handler.set_draft_tokens(\n"
     "                input_batch,\n"
     "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
     "            )\n",
     "            # patch_dynamic_depth: hand only the first K draft columns to the handler\n"
     "            self.draft_tokens_handler.set_draft_tokens(\n"
     "                input_batch,\n"
     "                self.req_states.draft_tokens[input_batch.idx_mapping][\n"
     "                    :,\n"
     "                    : max(1, getattr(self.speculator, \"_radiance_dyn_k\", 0)\n"
     "                          or self.num_speculative_steps),\n"
     "                ],\n"
     "            )\n"),
]


def edit(path, pairs):
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
    ast.parse(src)
    path.write_text(src)
    print(f"[dyn-depth] applied: {path.name}")


edit(SPEC, SPEC_PAIRS)
edit(RUNNER, RUNNER_PAIRS)
