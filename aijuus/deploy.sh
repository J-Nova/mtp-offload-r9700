#!/usr/bin/env bash
# aijuus/deploy.sh -- one correct path from overlay -> image -> Coolify.
#
# WHY THIS EXISTS
#   Nothing applies the overlay (aijuus/patches/*.patch) for you:
#     * build.sh bakes whatever the working tree contains -- if the overlay is
#       not applied, it silently bakes the pristine upstream files;
#     * Coolify never rebuilds -- it pulls the prebuilt image and only runs the
#       runtime aijuus/kv-offload/patches/patch_*.py scripts from the /patches mount.
#   This wrapper closes that gap and makes the "when do I apply / rebuild?"
#   decision explicit.
#
# USAGE
#   aijuus/deploy.sh check                  # report state (default; read-only)
#   aijuus/deploy.sh apply                  # apply the overlay to the working tree
#   aijuus/deploy.sh build [build.sh args]  # apply overlay, then ./build.sh ...
#   aijuus/deploy.sh full  [build.sh args]  # apply overlay, then ./build.sh --push ...
#
# OPTIONS
#   --no-apply   do not touch the overlay (you already applied/reverted it)
#   -h|--help
#
# EXAMPLES
#   aijuus/deploy.sh full --registry=registry.example.com/aijuus
#   aijuus/deploy.sh build --jobs=8
#
# REMEMBER
#   * Coolify only deploys -- it never builds. Rebuild ONLY for Dockerfile/base or
#     baked-only files (see the rebuild rule in aijuus/README.md); runtime changes
#     just need a redeploy/restart.
#   * revert.sh before `git fetch && rebase`; apply again afterwards.
#   * While deployed, keep the overlay APPLIED: the vLLM entrypoint re-copies
#     radiance_preamble.py / radiance_drafthead.py from the live tree at every
#     boot, so reverting silently undoes those two patches on next restart.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PATCHES_DIR="$ROOT/aijuus/patches"
COMPOSE="$ROOT/aijuus/coolify-compose-2gpu.yml"

CMD="check"
NO_APPLY=0
BUILD_ARGS=()

for a in "$@"; do
  case "$a" in
    check|apply|build|full) CMD="$a" ;;
    --no-apply) NO_APPLY=1 ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) BUILD_ARGS+=("$a") ;;
  esac
done

shopt -s nullglob
patches=("$PATCHES_DIR"/*.patch)
TOTAL=${#patches[@]}

applied_count() {
  local applied=0 p
  for p in "${patches[@]}"; do
    git apply --reverse --check "$p" >/dev/null 2>&1 && applied=$((applied + 1))
  done
  echo "$applied"
}

overlay_fully_applied() {
  [ "$TOTAL" -eq 0 ] || [ "$(applied_count)" -eq "$TOTAL" ]
}

require_overlay() {
  if overlay_fully_applied; then
    echo "overlay: $(applied_count)/$TOTAL patches already applied"
    return 0
  fi
  echo "overlay: $(applied_count)/$TOTAL applied -> running aijuus/apply.sh"
  bash aijuus/apply.sh
  if ! overlay_fully_applied; then
    echo "ERROR: overlay still not fully applied after apply.sh -- resolve conflicts before building" >&2
    exit 1
  fi
}

compose_image_default() {
  local v
  v=$(grep -m1 -oE '\$\{IMAGE:-[^}]+\}' "$COMPOSE" | sed 's/^\${IMAGE:-//; s/}$//')
  [ -n "$v" ] || v=$(grep -m1 -E '^[[:space:]]*image:' "$COMPOSE" | sed -E 's/^[[:space:]]*image:[[:space:]]*//')
  printf '%s' "$v"
}

running_image() {
  command -v docker >/dev/null 2>&1 || return 1
  local id
  id=$(docker ps -q --filter label=com.docker.compose.service=vllm-0 | head -1)
  [ -n "$id" ] || return 1
  docker inspect "$id" --format '{{.Config.Image}}'
}

runtime_patch_state() {
  command -v docker >/dev/null 2>&1 || return 1
  local id
  id=$(docker ps -q --filter label=com.docker.compose.service=vllm-0 | head -1)
  [ -n "$id" ] || return 1
  if docker exec "$id" sh -c 'grep -q RADIANCE_OFFLOAD_EAGLE_GROUPS /opt/vllm/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py' 2>/dev/null; then
    echo "applied in running vllm-0"
  else
    echo "NOT applied in running vllm-0"
  fi
}

rebuild_guidance() {
  cat <<'EOF'
rebuild needed?:   NO for runtime-deliverable changes (aijuus/kv-offload/patches/patch_*.py, compose/env,
                   registry/router/controller, and 062/063 via apply + restart)
                   -> just redeploy/restart the containers; the entrypoint applies them.
                   YES only for build-time files: Dockerfile/base (overlay 010/011, or
                   build.sh --base-only/--full) and baked-only files (060 radiance_draft.py,
                   061 radiance_draft_gpu.py, 070/071 paroquant). Then: build -> push ->
                   point the vLLM image / an IMAGE env at the tag -> redeploy. Coolify never builds.
EOF
}

case "$CMD" in
  apply)
    bash aijuus/apply.sh
    ;;

  build|full)
    echo "NOTE: rebuilding bakes the working tree. Runtime-only changes do NOT need this --"
    echo "      redeploy/restart instead. Build only for Dockerfile/base or baked-only files."
    echo
    if [ "$NO_APPLY" = 1 ]; then
      echo "(--no-apply: leaving the overlay as-is)"
    else
      require_overlay
    fi
    if [ "$CMD" = full ]; then
      ./build.sh --push "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
    else
      ./build.sh "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
    fi
    echo
    echo "== next =="
    echo "  Point Coolify / the vLLM services' image at the tag build.sh printed"
    echo "  (ggz14/vllm-radiance-mxfp4:\$VERSION-\$SHA, or the pushed <registry>/... form),"
    echo "  then redeploy. Runtime-only changes need no rebuild: just restart vllm-0/vllm-1."
    ;;

  check)
    applied=$(applied_count)
    echo "branch:            $(git rev-parse --abbrev-ref HEAD)"
    echo "overlay:           $applied/$TOTAL patches applied"
    if overlay_fully_applied; then
      echo "                   OK -- build.sh would bake your overlay"
    else
      echo "                   WARNING -- build.sh would bake the UNPATCHED upstream files;"
      echo "                   run: aijuus/deploy.sh apply"
    fi
    echo "compose image:     $(compose_image_default)  (point this / an IMAGE env at the built tag to deploy a build)"
    rebuild_guidance
    if img=$(running_image); then
      echo "running vllm-0:    $img"
      if overlay_fully_applied; then
        :
      else
        echo "                   NOTE: overlay is NOT applied, but the entrypoint re-copies"
        echo "                   radiance_preamble.py / radiance_drafthead.py from this tree"
        echo "                   each boot -- reverting undoes those two on next restart."
      fi
    else
      echo "running vllm-0:    (no container / docker unavailable)"
    fi
    if rps=$(runtime_patch_state); then
      echo "runtime patches:   $rps"
    fi
    ;;
esac
