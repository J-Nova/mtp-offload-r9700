#!/usr/bin/env bash
# Create a FIXED-SIZE, dedicated filesystem for the vLLM fs KV secondary tier.
#
# WHY. The fs tier has no size parameter (see patch_offload_fs_tier.py): it writes
# into `root_dir` and never refuses. `ops/kvcache-reap.sh KVCACHE_MAX_GIB=N` bounds
# the KV directory on a shared disk, but that is a soft target enforced every few
# minutes -- between reaps the tier can still fill whatever is free. A dedicated
# volume of exactly N GiB is the HARD bound: the tier cannot exceed it, and the
# space is RESERVED so nothing else on the host can take it. Pair the two: the
# volume is the hard ceiling, the reaper keeps usage comfortably below it so stores
# never fail.
#
# This makes a loopback ext4 image. `fallocate` reserves the blocks up front so the
# host filesystem cannot hand them to anything else; if `fallocate` is unavailable
# the image is sparse (a soft reservation) and the file's size still caps the tier.
#
# Idempotent: an existing image is kept; an already-mounted target is left alone.
# Requires root. Set DRY_RUN=1 to print the commands instead of running them.
#
#   sudo KVCACHE_GIB=200 ./ops/make-kvcache-volume.sh
#   # then, in both vllm services: KV_OFFLOAD_DISK_DIR=/kvcache/blocks
#   # and on the host:            KVCACHE_ROOT=/kvcache/blocks KVCACHE_MAX_GIB=190 \
#   #                             ./ops/kvcache-reap.sh   (systemd timer, see .service/.timer)
set -euo pipefail

GIB=${KVCACHE_GIB:-200}
IMG=${KVCACHE_IMG:-/var/lib/kvcache.img}
MOUNT=${KVCACHE_MOUNT:-/kvcache}
FSTAB=${KVCACHE_FSTAB:-1}   # 1 = add a persistent /etc/fstab entry
DRY=${DRY_RUN:-}

run() {
  if [ -n "$DRY" ]; then echo "+ $*"; else "$@"; fi
}

[ -n "$DRY" ] || [ "$(id -u)" = 0 ] || { echo "make-kvcache-volume: must run as root (or DRY_RUN=1 to preview)" >&2; exit 1; }

if mountpoint -q "$MOUNT"; then
  echo "make-kvcache-volume: $MOUNT is already a mountpoint; nothing to do"
  exit 0
fi

if [ ! -f "$IMG" ]; then
  echo "make-kvcache-volume: creating ${GIB} GiB image at $IMG"
  if command -v fallocate >/dev/null 2>&1; then
    # fallocate reserves the blocks now, so nothing else on the host can take them.
    run fallocate -l "${GIB}G" "$IMG"
  else
    echo "make-kvcache-volume: fallocate not found; using a sparse image (soft reservation)" >&2
    run truncate -s "${GIB}G" "$IMG"
  fi
  run mkfs.ext4 -q -m 0 -F "$IMG"
else
  echo "make-kvcache-volume: reusing existing image $IMG ($(du -h "$IMG" | cut -f1) on disk)"
fi

run mkdir -p "$MOUNT"
run mount -o loop,noatime,nodiratime "$IMG" "$MOUNT"

if [ "$FSTAB" = 1 ] && ! grep -qs "[[:space:]]$MOUNT[[:space:]]" /etc/fstab; then
  line="$IMG $MOUNT ext4 loop,noatime,nodiratime 0 0"
  echo "make-kvcache-volume: adding fstab entry: $line"
  if [ -n "$DRY" ]; then echo "+ append to /etc/fstab: $line"; else
    printf '%s\n' "$line" >> /etc/fstab
  fi
fi

run mkdir -p "$MOUNT/blocks"
echo "make-kvcache-volume: done -- $MOUNT (${GIB} GiB)."
echo "  vllm services : KV_OFFLOAD_DISK_DIR=$MOUNT/blocks"
echo "  host reaper   : KVCACHE_ROOT=$MOUNT/blocks KVCACHE_MAX_GIB=$(( GIB - 10 )) ./ops/kvcache-reap.sh"
