#!/usr/bin/env bash
# Gather everything needed to reconcile the aijuus overlay with incoming upstream.
#
# Read-only against your branch: fetches origin, computes the overlap between
# aijuus/patches/*.patch and incoming upstream files, and dry-runs every patch
# against the NEW upstream in a throwaway worktree. Writes a report under
# aijuus/reports/<timestamp>/ and prints its path.
#
# It never rebases, applies, or commits anything in your working tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BR="$(git rev-parse --abbrev-ref HEAD)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/aijuus/reports/$STAMP"
mkdir -p "$OUT"

git fetch origin --quiet

{
  echo "branch:        $BR ($(git rev-parse --short HEAD))"
  echo "origin/main:   $(git rev-parse --short origin/main)"
  echo "behind:        $(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
  echo "ahead:         $(git rev-list --count origin/main..HEAD 2>/dev/null || echo '?')"
  echo "overlay state: $(if [ -n "$(git status --porcelain --untracked-files=no -- ':!aijuus')" ]; then echo APPLIED-OR-DIRTY; else echo pristine; fi)"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt" >&2

git status --porcelain > "$OUT/workspace-status.txt" || true
git log --oneline HEAD..origin/main > "$OUT/incoming-commits.txt"
git log --name-only --format='' HEAD..origin/main | awk 'NF' | sort -u > "$OUT/incoming-files.txt"

grep -h '^+++ b/' aijuus/patches/*.patch | sed 's#^+++ b/##' | sort -u > "$OUT/patch-targets.txt"
comm -12 "$OUT/patch-targets.txt" "$OUT/incoming-files.txt" > "$OUT/overlap-files.txt"

if [ ! -s "$OUT/overlap-files.txt" ]; then
  echo "no patch target collides with incoming upstream." > "$OUT/patch-status.txt"
  git apply --check aijuus/patches/*.patch 2>/dev/null || true
  for p in aijuus/patches/*.patch; do
    if git apply --reverse --check "$p" >/dev/null 2>&1; then s="already-applied"
    elif git apply --check "$p" >/dev/null 2>&1; then s="applies-clean"
    else s="NEEDS-ATTENTION"; fi
    printf '%-40s %s\n' "$(basename "$p")" "$s" >> "$OUT/patch-status.txt"
  done
  echo "$OUT"
  exit 0
fi

# --- capture each colliding file's upstream diff and my patch(es) -----------
mkdir -p "$OUT/overlap"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  safe="$(printf '%s' "$f" | tr '/' '_')"
  git log -p HEAD..origin/main -- "$f" > "$OUT/overlap/${safe}.upstream.diff" 2>/dev/null || true
  for p in aijuus/patches/*.patch; do
    if grep -q "^+++ b/$f\$" "$p"; then
      cp "$p" "$OUT/overlap/${safe}.$(basename "$p")"
    fi
  done
done < "$OUT/overlap-files.txt"

# --- dry-run every patch against the NEW upstream in a scratch worktree ----
SCRATCH="/tmp/aijuus-check-$$-$(date +%s)"
git worktree add --detach --quiet "$SCRATCH" origin/main
: > "$OUT/patch-status.txt"
: > "$OUT/scratch-conflicts.txt"
for p in "$ROOT"/aijuus/patches/*.patch; do
  name="$(basename "$p")"
  git -C "$SCRATCH" reset --hard --quiet origin/main
  git -C "$SCRATCH" clean -fdq
  if git -C "$SCRATCH" apply --3way "$p" >/dev/null 2>&1; then
    if grep -rl '^<<<<<<< ' "$SCRATCH" 2>/dev/null | grep -qv '/.git/'; then
      printf '%-40s CONFLICT (3-way left markers)\n' "$name" >> "$OUT/patch-status.txt"
      grep -rl '^<<<<<<< ' "$SCRATCH" 2>/dev/null | grep -v '/.git/' \
        | sed "s#^$SCRATCH/##" | sed "s#^#$name #" >> "$OUT/scratch-conflicts.txt"
    else
      printf '%-40s applies-clean\n' "$name" >> "$OUT/patch-status.txt"
    fi
  else
    printf '%-40s FAILED (does not apply)\n' "$name" >> "$OUT/patch-status.txt"
    echo "$name (no 3-way resolution)" >> "$OUT/scratch-conflicts.txt"
  fi
done
git worktree remove --force "$SCRATCH" >/dev/null 2>&1 || true
rm -rf "$SCRATCH"

echo "$OUT"
