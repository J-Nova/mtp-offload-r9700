#!/usr/bin/env bash
# Revert the aijuus overlay, returning upstream-owned files to pristine state.
# Run this before `git fetch && git rebase origin/main` so the working tree is clean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$ROOT/aijuus/patches"
cd "$ROOT"

shopt -s nullglob
reverted=0
for p in $(ls -r "$PATCHES"/*.patch); do
  name="$(basename "$p")"
  if git apply --reverse --check "$p" >/dev/null 2>&1; then
    git apply --reverse "$p"
    echo "revert $name"
    reverted=$((reverted + 1))
  else
    echo "skip   $name (not applied)"
  fi
done
echo "aijuus overlay: $reverted reverted."
