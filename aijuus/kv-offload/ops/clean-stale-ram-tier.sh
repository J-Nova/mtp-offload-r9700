#!/bin/bash
# clean-stale-ram-tier.sh - Remove stale /dev/shm/vllm_offload_*.mmap files
# Usage: clean-stale-ram-tier.sh [ENG_ID]
# If ENG_ID is provided (HA mode), skip files matching any known instance pattern (radrank0*, radrank1*)
# to avoid deleting the peer instance's live KV offload regions.

ENG_ID="${1:-}"

shopt -s nullglob
for _f in /dev/shm/vllm_offload_*.mmap; do
  [ -e "${_f}" ] || continue

  # In HA mode, skip files matching any known instance pattern (both instances)
  if [ -n "${ENG_ID}" ]; then
    case "${_f}" in
      /dev/shm/vllm_offload_radrank[01]*.mmap) continue ;;
    esac
  fi

  # Skip files held by a live process (fuser/lsof check)
  if command -v fuser >/dev/null 2>&1 && fuser -s "${_f}" 2>/dev/null; then
    continue
  fi
  if command -v lsof >/dev/null 2>&1 && lsof "${_f}" >/dev/null 2>&1; then
    continue
  fi

  echo "[kv-cache] removing stale RAM tier $(basename "${_f}") ($(( $(stat -c %s "${_f}") >> 20 )) MiB)"
  rm -f "${_f}"
done
shopt -u nullglob
