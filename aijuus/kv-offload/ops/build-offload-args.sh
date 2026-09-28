#!/bin/bash
# Shared KV offload configuration logic.
# Source this script from both serve-mxfp4.sh and the compose entrypoint.
# Requires: KV_OFFLOAD_GIB, KV_OFFLOAD_DISK_DIR (optional), SERVE_RANK (optional for ENG_ID)
# Sets: OPOLICY, OFF_READ, OFF_WRITE, OFF_ARG, ENG_ID (if SERVE_RANK set)

# Offload tuning knobs, resolved once. Used by both the transfer-config JSON
# and the shape-label metric the Grafana "Offload tuning A/B" row reads, so a
# policy/thread/async change shows up as a label flip on the next redeploy.
OPOLICY=${KV_OFFLOAD_EVICTION_POLICY:-lru}
OFF_READ=${KV_OFFLOAD_READ_THREADS:-32}
OFF_WRITE=${KV_OFFLOAD_WRITE_THREADS:-16}

# Resolve disk tier path from host dir if provided
if [ -z "${KV_OFFLOAD_DISK_DIR:-}" ] && [ -n "${KV_OFFLOAD_DISK_HOST_DIR:-}" ]; then
  KV_OFFLOAD_DISK_DIR=/kvcache
fi

# Engine ID for shared /dev/shm region naming (only in HA mode)
if [ -n "${SERVE_RANK:-}" ]; then
  ENG_ID="radrank$SERVE_RANK"
fi

# Stale RAM tier cleanup: remove a leaked /dev/shm/vllm_offload_*.mmap.
# In HA mode (SERVE_RANK set) it removes ONLY our own radrank$SERVE_RANK region and
# never the peer's live one; without a rank it sweeps all files not held by a live
# process (fuser/lsof). Removing our own stale region is what prevents the next start
# from dying in SharedOffloadRegion (FileExistsError -> _wait_for_file_size timeout).
if [ -n "${KV_OFFLOAD_GIB:-}" ] && [ "$KV_OFFLOAD_GIB" != 0 ]; then
  bash aijuus/kv-offload/ops/clean-stale-ram-tier.sh "${SERVE_RANK:+radrank$SERVE_RANK}"
fi

# KV offload (env-gated, default off). --kv-offloading-size sets cpu_bytes_to_use AND
# selects OffloadingConnector; the JSON only adds extra_config.
# Head cap (patch_offload_head_cap.py) is what distinguishes the two modes:
#   Mode A (CPU only, no KV_OFFLOAD_DISK_DIR): max_offload_tokens="auto" -- fit the CPU
#     tier exactly, so over-subscription restores a contiguous partial prefix, never nothing.
#   Mode B (CPU + shared fs, KV_OFFLOAD_DISK_DIR set): NO cap -- the disk holds the
#     overflow, and a per-request cap would defeat it. Reaper REQUIRED.
# KV_OFFLOAD_HEAD_CAP overrides the topology default: "auto" | <N> | "off".
# No spaces in the JSON, so word-splitting the unquoted OFF_ARG is safe.
OFF_ARG=""
if [ -n "${KV_OFFLOAD_GIB:-}" ] && [ "$KV_OFFLOAD_GIB" != 0 ]; then
  HC=${KV_OFFLOAD_HEAD_CAP:-}
  if [ -z "$HC" ]; then
    if [ -n "${KV_OFFLOAD_DISK_DIR:-}" ]; then HC=off; else HC=auto; fi
  fi
  EC="\"offload_prompt_only\":true"
  if [ "$HC" != off ] && [ "$HC" != 0 ]; then
    EC="\"max_offload_tokens\":\"$HC\",$EC"
  fi
  if [ -n "${KV_OFFLOAD_DISK_DIR:-}" ]; then
    EC="$EC,\"spec_name\":\"TieringOffloadingSpec\",\"eviction_policy\":\"$OPOLICY\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_OFFLOAD_DISK_DIR\",\"n_read_threads\":$OFF_READ,\"n_write_threads\":$OFF_WRITE}]"
    export RADIANCE_LOOKUP_INVALIDATE=1 RADIANCE_FS_FAILED_LOAD_FORGET=1
    echo "[radiance] fs KV tier ON: root_dir=$KV_OFFLOAD_DISK_DIR (host-side reaper REQUIRED)"
  fi
  OFF_ARG="--kv-offloading-size $KV_OFFLOAD_GIB --kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$EC}}"
  echo "[radiance] KV offload ON: cpu tier=${KV_OFFLOAD_GIB}GiB head_cap=$HC policy=$OPOLICY"
else
  echo "[radiance] KV offload OFF (plain serving)"
fi
