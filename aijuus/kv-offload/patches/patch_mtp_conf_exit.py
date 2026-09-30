#!/usr/bin/env python3
"""RADIANCE MTP confidence early-exit for the V2 autoregressive speculator (vLLM 0.29).

Port of the useful half of tcclaviger/vllm:dev (29.06.18) `patches/mtp_confidence_exit`:
per-request early exit of the serial MTP draft loop from the drafter's own top-1 confidence,
plus a length bridge so the target verifies only the drafts each request kept (ragged verify).

Why depth first: our own sweep (WORKLOG cont.26) shows verify width alone is not the lever --
`RADIANCE_DYNAMIC_WIDTH` varies it and the dominant cost is the number of serial MTP forwards
(proposer depth). This patch makes that depth content-adaptive per request, instead of only
batch-size-scheduled (patch_dynamic_depth's static `spec_schedule`), and carries the per-request
length through the existing placeholder-length path so verify rows shrink too.

Mechanism (no new kernel, no rebuild):
  * the per-step top-1 probability of the greedily picked draft is written from the captured
    draft graph into a persistent device buffer (`_radiance_conf`) -- the same in-graph signal
    tcclaviger's `sample_draft_with_confidence` produces;
  * after each replay in `_multi_step_decode` the host multiplies a per-request running product
    and records the position where each row first drops below tau; the loop stops once every row
    has stopped (`draft_lens`);
  * `_radiance_lens` is handed to the runner, which passes it to `DraftTokensHandler`; with
    async scheduling off the handler returns placeholder rows of that length, so the scheduler
    schedules (and the target verifies) each request's kept width -- the existing supported path;
  * `_radiance_dyn_k` is set to max(draft_lens) so patch_dynamic_depth's runner slice is exact.

Inert unless RADIANCE_MTP_CONF_EXIT=1 (env read at vLLM runtime; the patch is idempotent and
always applied). Tau = RADIANCE_MTP_EXIT_TAU or RADIANCE_DRAFT_TAU (default 0.28). Raw top-1
confidence is optimistic relative to true acceptance, so useful tau is usually higher (0.4-0.8).

Lossless: only the number of proposed tokens changes; every kept draft still goes through the
unchanged rejection sampler. A confidence-only exit can only propose fewer tokens.

Every edited file must ast.parse.
"""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_mtp_conf_exit"
SP = Path(sysconfig.get_paths()["purelib"])
SPEC = SP / "vllm/v1/worker/gpu/spec_decode/speculator.py"
AR = SP / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
UTILS = SP / "vllm/v1/worker/gpu/spec_decode/utils.py"
RUNNER = SP / "vllm/v1/worker/gpu/model_runner.py"

# --- 1. speculator.py: capture the top-1 draft probability in-graph -------------------------
GREEDY_OLD = (
    "        logits = self.model.compute_logits(hidden_states)\n"
    "        return logits.argmax(dim=-1)\n"
)
GREEDY_NEW = (
    "        logits = self.model.compute_logits(hidden_states)\n"
    "        # patch_mtp_conf_exit: publish the top-1 probability of the greedily picked draft\n"
    "        # token into the persistent buffer the host early-exit gate reads. Buffer is None\n"
    "        # unless the gate is on and local-argmax reduction is off (it needs full logits).\n"
    "        _rc = getattr(self, \"_radiance_conf\", None)\n"
    "        if _rc is not None:\n"
    "            _rcp = torch.softmax(logits, dim=-1)\n"
    "            _rcv = _rcp.max(dim=-1).values\n"
    "            _rc[: _rcv.shape[0]] = _rcv.to(_rc.dtype)\n"
    "        return logits.argmax(dim=-1)\n"
)

# --- 2. autoregressive/speculator.py: allocate the buffer in __init__ ----------------------
INIT_OLD = "        self.inputs_embeds: torch.Tensor | None = None\n"
INIT_NEW = (
    "        self.inputs_embeds: torch.Tensor | None = None\n"
    "        # patch_mtp_conf_exit: per-request top-1 draft confidence buffer (device) + the\n"
    "        # per-request kept lengths of the last propose. Allocated only when the gate is on.\n"
    "        self._radiance_lens = None\n"
    "        if _RC_ON and not getattr(self, \"use_local_argmax_reduction\", False):\n"
    "            self._radiance_conf = torch.zeros(\n"
    "                self.max_num_reqs, dtype=torch.float32, device=device\n"
    "            )\n"
    "            self._radiance_conf_tau = _RC_TAU\n"
    "        else:\n"
    "            self._radiance_conf = None\n"
)

# propose(): clear the length bridge for this call (also covers the num_speculative_steps==1
# early-return path, which never enters _multi_step_decode).
PROPOSE_OLD = (
    "        self._radiance_eff_k = _rd_k\n"
    "        self._radiance_dyn_k = _rd_k\n"
)
PROPOSE_NEW = (
    "        self._radiance_eff_k = _rd_k\n"
    "        self._radiance_dyn_k = _rd_k\n"
    "        self._radiance_lens = None\n"
)

# --- 3. autoregressive/speculator.py: per-request survival gate around the loop -------------
LOOP_OLD = (
    "        attn_metadata = None\n"
    "        slot_mappings_by_layer = None\n"
    "        for step in range(1, getattr(self, \"_radiance_eff_k\", self.num_speculative_steps)):\n"
)
LOOP_NEW = (
    "        attn_metadata = None\n"
    "        slot_mappings_by_layer = None\n"
    "        # patch_mtp_conf_exit: per-request top-1 confidence product; a row stops at the first\n"
    "        # step its product falls below tau. The loop ends when every row has stopped (or the\n"
    "        # scheduled depth is reached); draft_lens carries each row's kept width.\n"
    "        import numpy as _rc_np\n"
    "        _rc_on = getattr(self, \"_radiance_conf\", None) is not None\n"
    "        _rc_tau = getattr(self, \"_radiance_conf_tau\", 0.28)\n"
    "        _rc_surv = _rc_np.ones(num_reqs, dtype=_rc_np.float32)\n"
    "        _rc_alive = _rc_np.ones(num_reqs, dtype=bool)\n"
    "        _rc_lens = _rc_np.ones(num_reqs, dtype=_rc_np.int64)\n"
    "        _rc_ran = 1\n"
    "        for step in range(1, getattr(self, \"_radiance_eff_k\", self.num_speculative_steps)):\n"
)

STEP_OLD = (
    "            else:\n"
    "                self._generate_draft(\n"
    "                    num_reqs,\n"
    "                    batch_desc.num_tokens,\n"
    "                    attn_metadata,\n"
    "                    slot_mappings_by_layer,\n"
    "                    num_tokens_across_dp=num_tokens_across_dp,\n"
    "                    cudagraph_runtime_mode=batch_desc.cg_mode,\n"
    "                )\n"
    "\n"
    "    def _fused_multi_step_decode(\n"
)
STEP_NEW = (
    "            else:\n"
    "                self._generate_draft(\n"
    "                    num_reqs,\n"
    "                    batch_desc.num_tokens,\n"
    "                    attn_metadata,\n"
    "                    slot_mappings_by_layer,\n"
    "                    num_tokens_across_dp=num_tokens_across_dp,\n"
    "                    cudagraph_runtime_mode=batch_desc.cg_mode,\n"
    "                )\n"
    "            _rc_ran = step + 1\n"
    "            if _rc_on:\n"
    "                _rc_c = self._radiance_conf[:num_reqs].float().cpu().numpy()\n"
    "                _rc_surv = _rc_surv * _rc_c\n"
    "                _rc_new = _rc_alive & (_rc_surv < _rc_tau)\n"
    "                _rc_lens[_rc_new] = step + 1\n"
    "                _rc_alive &= ~_rc_new\n"
    "                _rc_dbg = getattr(self, \"_rc_dbg\", 0)\n"
    "                if _rc_dbg < 30:\n"
    "                    self._rc_dbg = _rc_dbg + 1\n"
    "                    import sys as _rcs\n"
    "                    print(f\"[conf-exit] step={step} n={num_reqs} alive={int(_rc_alive.sum())} \"\n"
    "                          f\"surv_min={float(_rc_surv.min()):.3f} tau={_rc_tau}\", file=_rcs.stderr)\n"
    "                if not _rc_alive.any():\n"
    "                    break\n"
    "        # patch_mtp_conf_exit: rows still alive keep the full run; every row's kept width is\n"
    "        # its stop position or the run length. The runner slices to max(lens) and the handler\n"
    "        # schedules each request's own width.\n"
    "        _rc_lens[_rc_alive] = _rc_ran\n"
    "        self._radiance_lens = _rc_lens\n"
    "        self._radiance_dyn_k = int(_rc_lens.max())\n"
    "        self._radiance_eff_k = int(_rc_lens.max())\n"
    "\n"
    "    def _fused_multi_step_decode(\n"
)

# --- 4. spec_decode/utils.py: length bridge -------------------------------------------------
SET_OLD = (
    "    def set_draft_tokens(\n"
    "        self, input_batch: InputBatch, draft_tokens: torch.Tensor\n"
    "    ) -> None:\n"
    "        self.req_ids = input_batch.req_ids\n"
    "        self.num_draft_tokens = draft_tokens.shape[1]\n"
)
SET_NEW = (
    "    def set_draft_tokens(\n"
    "        self, input_batch: InputBatch, draft_tokens: torch.Tensor,\n"
    "        draft_lens: list[int] | None = None,\n"
    "    ) -> None:\n"
    "        self.req_ids = input_batch.req_ids\n"
    "        self.num_draft_tokens = draft_tokens.shape[1]\n"
    "        # patch_mtp_conf_exit: per-request kept widths (None = uniform width).\n"
    "        self.draft_lens = None if draft_lens is None else [int(_l) for _l in draft_lens]\n"
)

GET_OLD = (
    "    def get_draft_tokens(self) -> DraftTokenIds | None:\n"
    "        if self.draft_tokens_np is not None:\n"
    "            self.copy_event.synchronize()\n"
    "            draft_token_ids = self.draft_tokens_np.tolist()\n"
    "        else:\n"
    "            # This case only happens when async scheduling is disabled.\n"
    "            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]\n"
    "        return DraftTokenIds(self.req_ids, draft_token_ids)\n"
)
GET_NEW = (
    "    def get_draft_tokens(self) -> DraftTokenIds | None:\n"
    "        lens = getattr(self, \"draft_lens\", None)\n"
    "        if self.draft_tokens_np is not None:\n"
    "            self.copy_event.synchronize()\n"
    "            draft_token_ids = self.draft_tokens_np.tolist()\n"
    "            if lens is not None:\n"
    "                draft_token_ids = [row[: _l] for row, _l in zip(draft_token_ids, lens)]\n"
    "        else:\n"
    "            # This case only happens when async scheduling is disabled. The scheduler needs\n"
    "            # only the width; the real drafts stay in the runner. patch_mtp_conf_exit: use the\n"
    "            # per-request width so each request schedules its own kept count (ragged verify).\n"
    "            if lens is None:\n"
    "                draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]\n"
    "            else:\n"
    "                draft_token_ids = [[-1] * _l for _l in lens]\n"
    "        return DraftTokenIds(self.req_ids, draft_token_ids)\n"
)

# --- 5. model_runner.py: pass the lengths through -------------------------------------------
RUN_OLD = (
    "            _rd_d = getattr(self, \"_radiance_last_draft\", None)\n"
    "            self.draft_tokens_handler.set_draft_tokens(\n"
    "                input_batch,\n"
    "                _rd_d\n"
    "                if _rd_d is not None\n"
    "                else self.req_states.draft_tokens[input_batch.idx_mapping],\n"
    "            )\n"
)
RUN_NEW = (
    "            _rd_d = getattr(self, \"_radiance_last_draft\", None)\n"
    "            self.draft_tokens_handler.set_draft_tokens(\n"
    "                input_batch,\n"
    "                _rd_d\n"
    "                if _rd_d is not None\n"
    "                else self.req_states.draft_tokens[input_batch.idx_mapping],\n"
    "                # patch_mtp_conf_exit: per-request kept widths.\n"
    "                draft_lens=getattr(self.speculator, \"_radiance_lens\", None),\n"
    "            )\n"
)

TAIL = '''

# ---- RADIANCE MTP confidence early-exit (patch_mtp_conf_exit) -------------------------------
import os as _rc_os

_RC_ON = _rc_os.environ.get("RADIANCE_MTP_CONF_EXIT", "0") == "1"
_RC_TAU = float(
    _rc_os.environ.get("RADIANCE_MTP_EXIT_TAU")
    or _rc_os.environ.get("RADIANCE_DRAFT_TAU", "0.28")
)
'''


def edit(path, pairs, tail=None):
    src = path.read_text()
    if MARK in src:
        print(f"[conf-exit] {path.name} already applied")
        return
    for tag, old, new in pairs:
        n = src.count(old)
        if n != 1:
            print(f"[conf-exit] {path.name}: anchor {tag} matched {n}x, NOT applied", file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    if tail:
        src += tail
    ast.parse(src)
    path.write_text(src)
    print(f"[conf-exit] applied: {path.name}")


edit(SPEC, [("greedy-conf", GREEDY_OLD, GREEDY_NEW)])
edit(AR, [("init-buffer", INIT_OLD, INIT_NEW),
          ("propose-clear", PROPOSE_OLD, PROPOSE_NEW),
          ("loop-gate", LOOP_OLD, LOOP_NEW),
          ("step-gate", STEP_OLD, STEP_NEW)], tail=TAIL)
edit(UTILS, [("set-lens", SET_OLD, SET_NEW), ("get-lens", GET_OLD, GET_NEW)])
edit(RUNNER, [("pass-lens", RUN_OLD, RUN_NEW)])
