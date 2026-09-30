#!/bin/bash
# apply-kv-patches.sh - Apply all KV offload patches
# This script centralizes the patch invocation list to avoid drift between
# compose entrypoint and serve-mxfp4.sh.
#
# Best-effort by design: the container entrypoint sources this under `set -e`,
# so a single hunk whose anchor upstream reworded must not take the whole boot
# down. Each patch that cannot apply is reported by name and skipped; the only
# hard requirement is that this script itself exits 0. Functional patches
# (mixed-hit, eagle-fallback, eagle-groups, head-cap, mamba/swa) are named in
# the log when they miss so the miss is visible rather than silent.

PATCH_DIR="${PATCH_DIR:-aijuus/kv-offload/patches}"

run_patch() {
  local patch="$1" what="$2"
  if ! python3 "${PATCH_DIR}/${patch}"; then
    echo "[radiance] WARNING: ${what} did NOT apply (anchor drift?) -- continuing without it" >&2
  fi
}

# Phase 1 (FATAL-intent): mixed-hit safety + instrumentation. Must run before other offload patches.
if [ "${RADIANCE_OFFLOAD_MIXED_HIT:-1}" != 0 ]; then run_patch patch_offload_mixed_hit.py "mixed-hit safety"; fi
if [ "${RADIANCE_OFFLOAD_INSTRUMENTATION:-1}" != 0 ]; then run_patch patch_offload_instrumentation.py "offload instrumentation"; fi

# Existing offload patches
if [ "${RADIANCE_OFFLOAD_DEBUG:-0}" = 1 ]; then run_patch patch_offload_debug.py "offload debug (diagnostic)"; fi   # diagnostic only
if [ "${RADIANCE_OFFLOAD_EAGLE_FIX:-1}" != 0 ]; then run_patch patch_offload_eagle_fallback.py "eagle fallback (hybrid Mamba/SWA correctness)"; fi
if [ "${RADIANCE_OFFLOAD_EAGLE_GROUPS:-1}" != 0 ]; then run_patch patch_offload_eagle_groups.py "eagle group annotation"; fi
run_patch patch_mamba_stride.py "mamba store cadence"
run_patch patch_offload_swa_touch.py "swa-align/touch-order"
run_patch patch_offload_fs_tier.py "fs tier fan-out (do NOT enable the fs disk tier if this missed)"
run_patch patch_offload_head_cap.py "head cap (CPU-only retention; RAM tier keeps only the tail if this missed)"
if [ "${RADIANCE_OFFLOAD_TRACE:-0}" = 1 ]; then run_patch patch_offload_trace.py "offload trace (deep diagnostic)"; fi

# Phase 2 (behavioral): reconcile re-ask + prompt last-block alignment.
if [ "${RADIANCE_RECONCILE_REASK:-1}" != 0 ]; then run_patch patch_reconcile_reask.py "reconcile re-ask"; fi
if [ "${RADIANCE_ALIGN_PROMPT_LAST_BLOCK:-1}" != 0 ]; then run_patch patch_sched_align_last_block.py "prompt last-block alignment"; fi

# Phase 5 (metrics): lookup counters, debug instrument, tier report, promotion wallclock, miss deferral.
if [ "${RADIANCE_OFFLOAD_LOOKUP_METRICS:-1}" = 1 ]; then run_patch patch_offload_lookup_metrics.py "lookup metrics"; fi
if [ "${RADIANCE_OFFLOAD_DEBUG_INSTRUMENT:-1}" = 1 ]; then run_patch patch_offload_debug_instrument.py "debug instrument"; fi
if [ "${RADIANCE_OFFLOAD_TIER_REPORT:-1}" = 1 ]; then run_patch patch_offload_tier_report.py "tier report"; fi
if [ "${RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK:-1}" = 1 ]; then run_patch patch_offload_promotion_wallclock.py "promotion wallclock"; fi
if [ "${RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS:-1}" = 1 ]; then run_patch patch_offload_miss_deferral_metrics.py "miss deferral metrics"; fi

# E2 lazy-commit: take the store D2H off the step path (inert unless RADIANCE_OFFLOAD_LAZY_COMMIT=1).
run_patch patch_offload_lazy_commit.py "offload lazy commit (E2)"
# E1 suffix-only invalidation plumbing (inert unless RADIANCE_OFFLOAD_SUFFIX_INV=1).
run_patch patch_offload_suffix_inv.py "offload suffix invalidation (E1)"
# E1 Stage 2: include eagle/MTP groups (inert unless RADIANCE_OFFLOAD_EAGLE_INCLUDE=1).
run_patch patch_offload_eagle_include.py "offload eagle include (E1 Stage 2)"

exit 0
