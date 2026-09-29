#!/usr/bin/env bash
# One-shot setup for the KV-offload disk tier (Mode B):  "the whole thing".
#
# What it does, in order:
#   1. prepares the shared disk-tier backing store on the host
#        - plain directory (soft-capped by the reaper), or
#        - a fixed-size, RESERVED ext4 volume with --volume-gib (hard cap)
#   2. installs + starts the fs-tier reaper (systemd timer) with a GiB cap
#      (the tier has no capacity/eviction of its own -- the reaper IS the policy)
#   3. prints the Coolify/environment variables that switch the deployment on
#      (and writes them to --env-file if given)
#   4. validates: timer active, one reaper run, paths, and the env to apply
#
# Reuses ops/make-kvcache-volume.sh and ops/install-reaper.sh; this is just the
# single front door. Idempotent: re-running reconfigures, never duplicates.
#
#   sudo ./ops/setup-kv-offload.sh                       # 12 GiB RAM/inst, 100 GiB disk cap
#   sudo ./ops/setup-kv-offload.sh --ram-gib 12 --disk-gib 200
#   sudo ./ops/setup-kv-offload.sh --volume-gib 500      # reserve a 500 GiB volume instead
#   sudo ./ops/setup-kv-offload.sh --uninstall
#   ./ops/setup-kv-offload.sh --dry-run                  # print actions, change nothing
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RAM_GIB=12
DISK_GIB=100
DISK_DIR=/var/lib/radiance-kvcache
VOLUME_GIB=0
MIN_AGE=90
ENV_FILE=""
ACTION=setup
DRY=0

usage() {
  cat <<'EOF'
Setup the KV-offload disk tier (Mode B) for this host in one step.

Usage: setup-kv-offload.sh [options]

Options:
  --ram-gib N       CPU tier per instance, GiB (KV_OFFLOAD_GIB)        [12]
  --disk-gib N      cap on the KV directory, GiB (KVCACHE_MAX_GIB)     [100]
  --disk-dir PATH   host directory backing the disk tier               [/var/lib/radiance-kvcache]
  --volume-gib N    reserve a fixed N-GiB ext4 volume at --disk-dir instead of a plain dir
  --min-age MIN     reaper minimum-age floor, minutes                  [90]
  --env-file PATH   write the deployment env vars to PATH (creates/updates keys)
  --uninstall       stop+remove the reaper timer/units/env (leaves KV data)
  --dry-run         print actions; change nothing
  -h, --help        this help

After it runs, set the printed variables in Coolify on BOTH vllm services and redeploy.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ram-gib)     RAM_GIB="${2:?}"; shift 2 ;;
    --disk-gib)    DISK_GIB="${2:?}"; shift 2 ;;
    --disk-dir)    DISK_DIR="${2:?}"; shift 2 ;;
    --volume-gib)  VOLUME_GIB="${2:?}"; shift 2 ;;
    --min-age)     MIN_AGE="${2:?}"; shift 2 ;;
    --env-file)    ENV_FILE="${2:?}"; shift 2 ;;
    --uninstall)   ACTION=uninstall; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "setup-kv-offload: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }

if [ "$DRY" = 1 ]; then export DRY_RUN=1; fi

if [ "$DRY" != 1 ] && [ "$(id -u)" != 0 ]; then
  echo "setup-kv-offload: must run as root (try: sudo $0 ...)  [--dry-run works unprivileged]" >&2
  exit 1
fi

# ---------------------------------------------------------------- uninstall
if [ "$ACTION" = uninstall ]; then
  echo "setup-kv-offload: uninstalling the reaper (KV data under $DISK_DIR is left in place)"
  run systemctl disable --now kvcache-reap.timer 2>/dev/null || true
  run rm -f /etc/systemd/system/kvcache-reap.service /etc/systemd/system/kvcache-reap.timer
  run rm -f /usr/local/bin/kvcache-reap.sh /etc/default/kvcache-reap
  run systemctl daemon-reload 2>/dev/null || true
  echo "setup-kv-offload: reaper removed. Remember to unset KV_OFFLOAD_GIB / KV_OFFLOAD_DISK_HOST_DIR."
  exit 0
fi

echo "setup-kv-offload: Mode B -- RAM ${RAM_GIB} GiB/instance, disk cap ${DISK_GIB} GiB, dir ${DISK_DIR}"
[ "$DRY" = 1 ] && echo "setup-kv-offload: DRY RUN -- nothing will be changed"

# ---------------------------------------------------------------- 1. backing store
if [ "$VOLUME_GIB" -gt 0 ]; then
  echo "setup-kv-offload: step 1/4 -- reserved ${VOLUME_GIB} GiB volume at ${DISK_DIR}"
  KVCACHE_GIB="$VOLUME_GIB" KVCACHE_MOUNT="$DISK_DIR" "$HERE/make-kvcache-volume.sh"
else
  echo "setup-kv-offload: step 1/4 -- plain directory (soft-capped by the reaper)"
  run mkdir -p "$DISK_DIR/blocks"
fi

if [ "$DISK_DIR" = /var/lib/radiance-kvcache ]; then
  HOST_DIR_DEFAULT=1
else
  HOST_DIR_DEFAULT=0
fi

# ---------------------------------------------------------------- 2. reaper
echo "setup-kv-offload: step 2/4 -- installing the fs-tier reaper (the eviction policy)"
KVCACHE_ROOT="$DISK_DIR/blocks" KVCACHE_MAX_GIB="$DISK_GIB" \
  KVCACHE_MIN_AGE_MIN="$MIN_AGE" "$HERE/install-reaper.sh"

# ---------------------------------------------------------------- 3. env vars
ENV_BLOCK="KV_OFFLOAD_GIB=$RAM_GIB
KV_OFFLOAD_DISK_HOST_DIR=$DISK_DIR"
if [ "$HOST_DIR_DEFAULT" = 0 ]; then
  # Non-default host dir: also point the in-container path explicitly (it defaults to
  # /kvcache/blocks, which matches the compose bind of KV_OFFLOAD_DISK_HOST_DIR).
  ENV_BLOCK="$ENV_BLOCK
KV_OFFLOAD_DISK_DIR=/kvcache/blocks"
fi

echo "setup-kv-offload: step 3/4 -- deployment variables"
if [ -n "$ENV_FILE" ]; then
  if [ "$DRY" = 1 ]; then
    echo "+ write $ENV_FILE:"; printf '%s\n' "$ENV_BLOCK" | sed 's/^/    /'
  else
    touch "$ENV_FILE"
    while IFS='=' read -r k v; do
      if grep -q "^${k}=" "$ENV_FILE"; then
        sed -i "s|^${k}=.*|${k}=${v}|" "$ENV_FILE"
      else
        printf '%s=%s\n' "$k" "$v" >> "$ENV_FILE"
      fi
    done <<< "$ENV_BLOCK"
    echo "setup-kv-offload: wrote $ENV_FILE"
  fi
fi
printf '%s\n' "$ENV_BLOCK" | sed 's/^/    /'

# ---------------------------------------------------------------- 4. validate
echo "setup-kv-offload: step 4/4 -- validation"
if [ "$DRY" = 1 ]; then
  echo "+ systemctl start kvcache-reap.service ; systemctl is-active kvcache-reap.timer"
else
  systemctl start kvcache-reap.service || true
  sleep 1
  echo "    timer: $(systemctl is-active kvcache-reap.timer)  ($(systemctl is-enabled kvcache-reap.timer 2>/dev/null || echo not-found))"
  echo "    reaper root: $DISK_DIR/blocks"
  echo "    reaper cap : ${DISK_GIB} GiB"
  journalctl -u kvcache-reap --no-pager -n 2 2>/dev/null | sed 's/^/    /' || true
fi

cat <<EOF

setup-kv-offload: done.

Next (both vllm services in Coolify), then redeploy:
    KV_OFFLOAD_GIB=$RAM_GIB
    KV_OFFLOAD_DISK_HOST_DIR=$DISK_DIR
$( [ "$HOST_DIR_DEFAULT" = 0 ] && echo "    KV_OFFLOAD_DISK_DIR=/kvcache/blocks" || true )

Clients: send a stable 'x-session-id' header so a conversation sticks to one instance.
Verify after deploy:
    docker logs <vllm> | grep -E "fs tier applied|fs KV tier ON|draft attention groups"
    journalctl -u kvcache-reap -f
Roll back: unset those vars, or run: sudo $0 --uninstall
EOF
