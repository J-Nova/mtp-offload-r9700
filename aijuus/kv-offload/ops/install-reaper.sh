#!/usr/bin/env bash
# Install the fs-tier reaper as a systemd timer, and write its environment file.
#
# The reaper is the fs tier's eviction policy (the tier never deletes). The paths must
# line up with what the deployment mounts: the compose mounts the host directory
# ${KV_OFFLOAD_DISK_HOST_DIR:-/var/lib/radiance-kvcache} into both instances at /kvcache,
# and the containers write under /kvcache/blocks -- so the host-side KV root is
# <host dir>/blocks. Change KV_OFFLOAD_DISK_HOST_DIR and the reaper now, together.
#
#   sudo KVCACHE_ROOT=/var/lib/radiance-kvcache/blocks KVCACHE_MAX_GIB=190 ./ops/install-reaper.sh
#   # on a dedicated volume instead:
#   sudo KVCACHE_ROOT=/kvcache-mount/blocks KVCACHE_MAX_GIB=190 ./ops/install-reaper.sh
#   DRY_RUN=1 ./ops/install-reaper.sh        # print what it would do
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Canonical cap lives in the repo .env (shared with compose + the exporter); an explicit
# env var still wins. This keeps one number from drifting between the reaper and compose.
# Repo root .env is three levels up from aijuus/kv-offload/ops (the pre-restructure
# kv-cache/ops copy had it one level up); prefer whichever exists.
ENV_FILE="$HERE/../../../.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HERE/../.env"
env_get() { { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | tail -1; } || true; }
ROOT=${KVCACHE_ROOT:-}
if [ -z "$ROOT" ]; then
  _hostdir=$(env_get KV_OFFLOAD_DISK_HOST_DIR)
  ROOT=${_hostdir:-/var/lib/radiance-kvcache}/blocks
fi
MAX_GIB=${KVCACHE_MAX_GIB:-$(env_get KVCACHE_MAX_GIB)}
MAX_GIB=${MAX_GIB:-0}
MIN_AGE_MIN=${KVCACHE_MIN_AGE_MIN:-90}
TARGET_PCT=${KVCACHE_TARGET_PCT:-65}
DRY=${DRY_RUN:-}

run() { if [ -n "$DRY" ]; then echo "+ $*"; else "$@"; fi; }

[ -n "$DRY" ] || [ "$(id -u)" = 0 ] || { echo "install-reaper: must run as root (or DRY_RUN=1)" >&2; exit 1; }

run install -m 0755 "$HERE/kvcache-reap.sh" /usr/local/bin/kvcache-reap.sh
run install -m 0644 "$HERE/kvcache-reap.service" /etc/systemd/system/kvcache-reap.service
run install -m 0644 "$HERE/kvcache-reap.timer" /etc/systemd/system/kvcache-reap.timer

conf="# Managed by ops/install-reaper.sh -- environment for kvcache-reap.service
KVCACHE_ROOT=$ROOT
KVCACHE_MIN_AGE_MIN=$MIN_AGE_MIN
KVCACHE_TARGET_PCT=$TARGET_PCT
KVCACHE_MAX_GIB=$MAX_GIB"
if [ -n "$DRY" ]; then
  echo "+ write /etc/default/kvcache-reap:"; printf '%s\n' "$conf" | sed 's/^/    /'
else
  install -d -m 0755 /etc/default
  printf '%s\n' "$conf" > /etc/default/kvcache-reap
fi

run systemctl daemon-reload
run systemctl enable --now kvcache-reap.timer
echo "install-reaper: installed. Check: systemctl list-timers kvcache-reap ; journalctl -u kvcache-reap"
if [ "$MAX_GIB" = 0 ]; then
  echo "install-reaper: note KVCACHE_MAX_GIB=0 -> capacity is bounded by the % rule only." >&2
  echo "install-reaper: set KVCACHE_MAX_GIB to the tier's byte budget (e.g. volume GiB - 10)." >&2
fi
