#!/bin/bash
# apply-kv-patches.sh - Apply all KV offload patches
# This script centralizes the patch invocation list to avoid drift between
# compose entrypoint and serve-mxfp4.sh.

PATCH_DIR="${PATCH_DIR:-aijuus/kv-offload/patches}"

# Phase 1 (FATAL): mixed-hit safety + instrumentation. Must run before other offload patches.
if [ "${RADIANCE_OFFLOAD_MIXED_HIT:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_offload_mixed_hit.py"; fi
if [ "${RADIANCE_OFFLOAD_INSTRUMENTATION:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_offload_instrumentation.py"; fi

# Existing offload patches
if [ "${RADIANCE_OFFLOAD_DEBUG:-0}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_debug.py"; fi   # diagnostic only
if [ "${RADIANCE_OFFLOAD_EAGLE_FIX:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_offload_eagle_fallback.py"; fi   # hybrid (Mamba/SWA) correctness
if [ "${RADIANCE_OFFLOAD_EAGLE_GROUPS:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_offload_eagle_groups.py"; fi   # per-group restore (~2x faster)
python3 "${PATCH_DIR}/patch_mamba_stride.py" \
  || echo "[radiance] WARNING: mamba store-cadence patch did NOT apply -- every chunk stores all six Mamba groups"
python3 "${PATCH_DIR}/patch_offload_swa_touch.py" \
  || echo "[radiance] WARNING: swa-align/touch-order patch did NOT apply -- heads still age out"
python3 "${PATCH_DIR}/patch_offload_fs_tier.py" \
  || echo "[radiance] WARNING: fs-tier patch did NOT apply -- do NOT enable the fs disk tier"
python3 "${PATCH_DIR}/patch_offload_head_cap.py"   # head-only CPU retention; inert unless max_offload_tokens/RADIANCE_OFFLOAD_MAX_TOKENS is set
if [ "${RADIANCE_OFFLOAD_TRACE:-0}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_trace.py"; fi   # deep diagnostic

# Phase 2 (behavioral): reconcile re-ask + prompt last-block alignment.
if [ "${RADIANCE_RECONCILE_REASK:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_reconcile_reask.py"; fi
if [ "${RADIANCE_ALIGN_PROMPT_LAST_BLOCK:-1}" != 0 ]; then python3 "${PATCH_DIR}/patch_sched_align_last_block.py"; fi

# Phase 5 (metrics): lookup counters, debug instrument, tier report, promotion wallclock, miss deferral.
if [ "${RADIANCE_OFFLOAD_LOOKUP_METRICS:-1}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_lookup_metrics.py"; fi
if [ "${RADIANCE_OFFLOAD_DEBUG_INSTRUMENT:-1}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_debug_instrument.py"; fi
if [ "${RADIANCE_OFFLOAD_TIER_REPORT:-1}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_tier_report.py"; fi
if [ "${RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK:-1}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_promotion_wallclock.py"; fi
if [ "${RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS:-1}" = 1 ]; then python3 "${PATCH_DIR}/patch_offload_miss_deferral_metrics.py"; fi
