#!/usr/bin/env bash
# Apply the aijuus overlay onto the pristine upstream tree.
#
# The personal files under aijuus/ are additions (never conflict with upstream).
# The changes to upstream-owned files live as patches in aijuus/patches/ and are
# applied here, on demand, to the repo working tree right before a build or run.
# Nothing here is ever committed, so `git fetch && git rebase origin/main` stays
# conflict-free as long as the overlay is reverted first.
#
# Usage:  aijuus/apply.sh      # apply all patches
#         aijuus/revert.sh     # undo them
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$ROOT/aijuus/patches"
cd "$ROOT"

shopt -s nullglob
applied=0 skipped=0
for p in "$PATCHES"/*.patch; do
  name="$(basename "$p")"
  if git apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "skip   $name (already applied)"
    skipped=$((skipped + 1))
  elif git apply --check "$p" >/dev/null 2>&1; then
    git apply "$p"
    echo "apply  $name"
    applied=$((applied + 1))
  else
    echo "FAIL   $name does not apply cleanly (upstream changed under it?)" >&2
    exit 1
  fi
done
echo "aijuus overlay: $applied applied, $skipped skipped."
