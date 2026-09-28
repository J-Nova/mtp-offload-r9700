#!/usr/bin/env python3
"""Make CPU-tier retention keep a prefix's HEAD, so a returning request always restores the
contiguous part that survives instead of nothing. Two hunks in
distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py.

Ported from zzpanic/qwen3.6-vllm-gfx1201-launchers `kv-cache/patches/patch_swa_align_touch.py`
(same vLLM 0.27.1 / radiance 0.9.3 image), re-anchored to this tree.

WHY PARTIAL REUSE FAILS TODAY (line-level).

  `_lookup` (scheduler.py:631) serves a request from the GPU prefix boundary
  (`num_computed_tokens`) outward: `_maximal_prefix_lookup` (:545) counts consecutive CPU hits
  from `start_chunk_idx = num_computed_tokens // tokens_per_chunk` and STOPS AT THE FIRST MISS, so
  a request is served the longest contiguous run after the boundary. That is already partial --
  a hole truncates the hit, it does not cancel it. The failure is at the HEAD: if the boundary is
  0 (the GPU evicted the prefix) and the CPU has lost chunk 0, the run length is 0 and
  :726-727 `if num_hit_chunks == 0: return 0` cancels the whole request. For full attention this
  is unavoidable -- position p needs the KV of every position < p, so a prefix must be
  contiguous FROM THE START. "Always work with partial reuse" therefore means: never let a
  prefix lose its head; let it lose its TAIL.

  Why the head is what gets lost: `_touch` (:612) refreshes
    * every full-attention chunk (all of them), but
    * for sliding-window groups (Mamba window 1, drafter window 3) only the newest window:
      `offload_keys[num_hit_chunks - window:]`.
  Mamba is a sliding-window group (`get_sliding_window_size_in_chunks` returns 1 for MambaSpec).
  So a conversation's Mamba/drafter snapshots age out from the head while their attention keys
  stay held -- `evict_skew` (the sibling measured 4 of 5 big losses losing every g0 snapshot).
  And because CachePolicy.touch applies its list IN REVERSE (cpu/policies/lru.py: `for key in
  reversed(list(keys))`, and arc.py likewise), touching GROUP BY GROUP makes every g0 key older
  than every g1 key: an M->0 for the request.

HUNK 2 (`RADIANCE_TOUCH_ALL_GROUPS`, `RADIANCE_TOUCH_POSITION_ORDER`) -- on a hit or store, touch
ONE list carrying EVERY group's keys, sorted by chunk end position, head to tail. Reverse
application then leaves the HEAD most recent, so eviction removes whole positions from the TAIL
first and each prefix keeps a contiguous head. A returning request restores that head (partial)
instead of re-prefilling the whole prompt. Same ordering as upstream PR #51787
(`order_request_keys`); that PR deletes `_touch` entirely against a newer tree.

HUNK 1 (`RADIANCE_SWA_STORE_MAMBA_ALIGN`) -- with the Mamba store stride
(`patch_mamba_stride.py`) `_lookup` rounds every hit down to the Mamba grid, but the STORE path
still keeps every sliding-window chunk of each segment (stock computes
`alignment_tokens <= tokens_per_chunk -> None`). The drafter's stored chunks then do not line up
with the chunks a grid-point hit reads, and the sibling measured the drafter lookup cascading
to 0 (reaskbench abbda2). This hunk raises the sliding-window store alignment to the Mamba grid
when it is a whole multiple of the full-attention chunk, and judges reachability on the ABSOLUTE
grid. It is a no-op unless the Mamba grid is coarser than the attention chunk, i.e. unless a
Mamba stride is in force.

GATES. All three default to 0 (off) = today's behaviour, so the patch is inert until enabled:
  RADIANCE_TOUCH_ALL_GROUPS=1 RADIANCE_TOUCH_POSITION_ORDER=1 RADIANCE_SWA_STORE_MAMBA_ALIGN=1
Enable touch-all+order first (it is the partial-reuse fix); enable swa-align together with a
Mamba stride. Idempotent; run once pre-serve.
"""
import os
import sys
import sysconfig
from pathlib import Path

# _patchlib.py lives at the repo root (four levels up from patches/); insert it
# explicitly since sys.path[0] is aijuus/ when run as `python3 aijuus/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from _patchlib import apply  # noqa: E402

SP = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"]))
SCHED = (
    SP
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "offloading"
    / "scheduler.py"
)

print("[radiance] offload swa-align / touch-order")

# --- module flag + absolute-grid reachability helper (before SchedulerOffloadConfig) ---------
apply(
    SCHED,
    anchor="\nclass SchedulerOffloadConfig(NamedTuple):\n",
    new=(
        "\n"
        "import os as _radiance_os\n"
        "\n"
        "# radiance swa-align: set by SchedulerOffloadConfig.from_spec when the sliding-window\n"
        "# store alignment was raised to the Mamba grid. The store path then judges reachability\n"
        "# on the ABSOLUTE grid: a tail-relative (partial-segment) rule stores every chunk as it\n"
        "# briefly becomes the newest one during chunked prefill.\n"
        "_RADIANCE_SWA_GRID_ACTIVE = False\n"
        "\n"
        "\n"
        "def _radiance_store_reachable(abs_idx, storable, align, window, is_eagle):\n"
        "    if not _RADIANCE_SWA_GRID_ACTIVE or align is None or window is None:\n"
        "        return is_store_reachable_swa_chunk(abs_idx, storable, align, window, is_eagle)\n"
        "    # A hit lands on a grid point g (a multiple of `align` chunks). _lookup scans for\n"
        "    # `window` consecutive chunks ending at g; an eagle group queries ONE chunk past g and\n"
        "    # pops it, so it needs chunks g-window .. g. Stock is_store_reachable_swa_chunk keeps\n"
        "    # the trailing window+1 chunks BEFORE g instead, which misses chunk g.\n"
        "    pos = abs_idx % align\n"
        "    if pos >= align - window:\n"
        "        return True\n"
        "    return bool(is_eagle) and pos == 0\n"
        "\n"
        "\n"
        "class SchedulerOffloadConfig(NamedTuple):\n"
    ),
    sentinel="_RADIANCE_SWA_GRID_ACTIVE = False",
    label="1 scheduler: RADIANCE_SWA_GRID_ACTIVE + _radiance_store_reachable",
)

# --- HUNK 1: raise the sliding-window store alignment to the Mamba grid ----------------------
apply(
    SCHED,
    anchor=(
        "        alignment_tokens: int | None = None\n"
        "        if len(full_attn_tokens_per_chunk) == 1:\n"
        "            alignment_tokens = full_attn_tokens_per_chunk.pop()\n"
    ),
    new=(
        "        alignment_tokens: int | None = None\n"
        "        if len(full_attn_tokens_per_chunk) == 1:\n"
        "            alignment_tokens = full_attn_tokens_per_chunk.pop()\n"
        "        # radiance swa-align: hits are ALSO rounded down to the Mamba alignment (x the\n"
        "        # Mamba store stride), so a sliding-window chunk earlier in that coarser segment\n"
        "        # can never serve a hit either. Kill switch: RADIANCE_SWA_STORE_MAMBA_ALIGN=0.\n"
        "        if (\n"
        '            _radiance_os.environ.get("RADIANCE_SWA_STORE_MAMBA_ALIGN", "0") == "1"\n'
        "            and alignment_tokens is not None\n"
        "        ):\n"
        "            _rad_mamba_align = resolve_mamba_align_size(spec, kv_cache_config)\n"
        "            if (\n"
        "                _rad_mamba_align is not None\n"
        "                and _rad_mamba_align > alignment_tokens\n"
        "                and _rad_mamba_align % alignment_tokens == 0\n"
        "            ):\n"
        "                logger.info(\n"
        '                    "[radiance] swa-align: sliding-window store alignment %d -> %d "\n'
        '                    "tokens (hits land on the Mamba grid)",\n'
        "                    alignment_tokens,\n"
        "                    _rad_mamba_align,\n"
        "                )\n"
        "                alignment_tokens = _rad_mamba_align\n"
        "                global _RADIANCE_SWA_GRID_ACTIVE\n"
        "                _RADIANCE_SWA_GRID_ACTIVE = True\n"
    ),
    sentinel="radiance swa-align: hits are ALSO",
    label="2 scheduler: sliding-window store alignment = Mamba grid",
)

apply(
    SCHED,
    anchor=(
        "                    if not is_store_reachable_swa_chunk(\n"
        "                        abs_chunk_idx,\n"
        "                        num_chunks,\n"
        "                        group_config.alignment_chunk_count,\n"
    ),
    new=(
        "                    # radiance swa-align: absolute grid, lookup-consistent (see\n"
        "                    # _radiance_store_reachable); stock rule when the grid is not active\n"
        "                    if not _radiance_store_reachable(\n"
        "                        abs_chunk_idx,\n"
        "                        num_chunks,\n"
        "                        group_config.alignment_chunk_count,\n"
    ),
    sentinel="radiance swa-align: absolute grid, lookup-consistent",
    label="3 scheduler: absolute-grid, lookup-consistent reachability",
)

# --- HUNK 2: touch-all + position order (the head/partial-retention fix) ---------------------
apply(
    SCHED,
    anchor=(
        "    def _touch(self, req_status: RequestOffloadState):\n"
        "        for group_config, group_state in zip(\n"
        "            self.config.kv_group_configs, req_status.group_states\n"
        "        ):\n"
        "            if group_config.sliding_window_size_in_chunks is None:\n"
    ),
    new=(
        "    def _touch(self, req_status: RequestOffloadState):\n"
        "        # radiance touch-all: refresh every group's chunks like attention, so a prefix's\n"
        "        # Mamba/drafter snapshots are not evicted while its attention survives. Without\n"
        "        # this, the head ages out and a returning request restores nothing (all-or-nothing\n"
        "        # at _lookup's num_hit_chunks == 0). Kill switch: RADIANCE_TOUCH_ALL_GROUPS=0.\n"
        '        _rad_touch_all = _radiance_os.environ.get("RADIANCE_TOUCH_ALL_GROUPS", "0") == "1"\n'
        "        if _rad_touch_all and (\n"
        '            _radiance_os.environ.get("RADIANCE_TOUCH_POSITION_ORDER", "0") == "1"\n'
        "        ):\n"
        "            # radiance touch-order: ONE touch per request, every group's keys interleaved\n"
        "            # by chunk end position, head to tail. Policies apply a touch list in reverse\n"
        "            # (lru.py/arc.py), so the head ends up most recent and eviction takes whole\n"
        "            # positions from the tail -- a prefix keeps a contiguous head to serve\n"
        "            # PARTIALLY instead of losing the head and restoring nothing. Same ordering\n"
        "            # as upstream PR #51787 order_request_keys.\n"
        "            # Kill switch: RADIANCE_TOUCH_POSITION_ORDER=0.\n"
        "            _rad_keyed = []\n"
        "            for group_config, group_state in zip(\n"
        "                self.config.kv_group_configs, req_status.group_states\n"
        "            ):\n"
        "                _rad_tpc = group_config.tokens_per_chunk\n"
        "                _rad_gidx = group_config.group_idx\n"
        "                for _rad_i, _rad_key in enumerate(group_state.offload_keys):\n"
        "                    _rad_keyed.append(((_rad_i + 1) * _rad_tpc, _rad_gidx, _rad_key))\n"
        "            _rad_keyed.sort(key=lambda t: (t[0], t[1]))\n"
        "            self.manager.touch([t[2] for t in _rad_keyed], req_status.req_context)\n"
        "            return\n"
        "        for group_config, group_state in zip(\n"
        "            self.config.kv_group_configs, req_status.group_states\n"
        "        ):\n"
        "            if group_config.sliding_window_size_in_chunks is None or _rad_touch_all:\n"
    ),
    sentinel="radiance touch-all:",
    label="4 scheduler: touch-all + position order",
)

print(
    "[radiance] applied -- touch-all="
    f"{os.environ.get('RADIANCE_TOUCH_ALL_GROUPS', '0')} "
    f"order={os.environ.get('RADIANCE_TOUCH_POSITION_ORDER', '0')} "
    f"swa-align={os.environ.get('RADIANCE_SWA_STORE_MAMBA_ALIGN', '0')}"
)
