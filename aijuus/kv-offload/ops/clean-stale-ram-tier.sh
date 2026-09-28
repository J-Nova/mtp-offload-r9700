#!/bin/bash
# clean-stale-ram-tier.sh - Remove a stale /dev/shm/vllm_offload_*.mmap region.
# Usage: clean-stale-ram-tier.sh [ENG_ID]
#
# vLLM unlinks its CPU offload region only in cleanup(), so a crashed/killed
# engine leaks /dev/shm/vllm_offload_<engine_id>.mmap. A stale file breaks the
# next start: SharedOffloadRegion opens with O_CREAT|O_EXCL -> FileExistsError,
# then falls back to _wait_for_file_size() on the old (wrong-size) file and
# times out ("Timed out waiting for mmap file to reach N bytes").
#
# ENG_ID is the engine_id pinned per instance (radrank0 / radrank1). Because
# ipc: host shares /dev/shm between HA instances, we must only ever remove OUR
# OWN region and never the peer's live one:
#   - with ENG_ID (HA): remove exactly /dev/shm/vllm_offload_<ENG_ID>.mmap.
#   - without ENG_ID (single instance): sweep every region not held by a live
#     process (fuser/lsof).

ENG_ID="${1:-}"

remove_stale() {
  local f="$1"
  [ -e "$f" ] || return 0
  echo "[kv-cache] removing stale RAM tier $(basename "$f") ($(( $(stat -c %s "$f") >> 20 )) MiB)"
  rm -f "$f"
}

if [ -n "${ENG_ID}" ]; then
  # HA mode: only our own named region. The engine is not up yet, so any file
  # bearing our name is stale by definition. Unlink is safe even if a dying
  # process still holds an fd -- the name is freed and the new engine (which
  # uses O_CREAT|O_EXCL) can recreate it. The peer's region is never touched.
  remove_stale "/dev/shm/vllm_offload_${ENG_ID}.mmap"
  exit 0
fi

# Single-instance mode: sweep every region that no live process holds.
shopt -s nullglob
for _f in /dev/shm/vllm_offload_*.mmap; do
  [ -e "${_f}" ] || continue

  if command -v fuser >/dev/null 2>&1 && fuser -s "${_f}" 2>/dev/null; then
    continue
  fi
  if command -v lsof >/dev/null 2>&1 && lsof "${_f}" >/dev/null 2>&1; then
    continue
  fi

  remove_stale "${_f}"
done
shopt -u nullglob
