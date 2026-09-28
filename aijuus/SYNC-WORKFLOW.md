# aijuus upstream-sync workflow

Playbook for reconciling the `aijuus` overlay with incoming upstream commits.
Invoke by telling the agent any of: `sync aijuus`, `aijuus sync`, or
"pull upstream and reconcile my patches".

## Hard rules

- Do NOT rebase, apply, regenerate, drop, or commit anything until the user
  explicitly approves after the report. The workflow MUST end by asking.
- Never use `git stash` (shared across worktrees).
- Keep `main` and `backup/*` untouched.

## Steps

1. Confirm you are on branch `aijuus` (`git rev-parse --abbrev-ref HEAD`).
   If not, stop and say so.

2. If the overlay is currently applied (`git status --porcelain --untracked-files=no`
   `-- ':!aijuus'` shows modified upstream-owned files), run `bash aijuus/revert.sh` first.
   State that you did.

3. Run `bash aijuus/upstream-check.sh`. It fetches origin, computes the overlap
   between `aijuus/patches/*.patch` and incoming upstream, and dry-runs every
   patch against the new upstream in a throwaway worktree. It prints the report
   directory `aijuus/reports/<timestamp>/` (gitignored).

4. Read the report: `summary.txt`, `incoming-commits.txt`, `overlap-files.txt`,
   `patch-status.txt`, `scratch-conflicts.txt`, and the per-file material under
   `overlap/` (upstream diff + my patch).

5. For every overlapping file, compare my patch against the upstream diff and
   classify:
   - UPSTREAM SUPERSEDES — upstream now does what my patch did → drop mine.
   - TAKE UPSTREAM — upstream's version is better → drop mine.
   - KEEP MINE — upstream changed unrelated lines, patch still applies → keep.
   - MERGE — both changed the same region → propose the exact merged result.
   Say which is better and why (correctness, completeness, perf, maintainability).
   Flag duplicated work (e.g. my `aijuus/kv-offload/patches/patch_offload_*` vs upstream
   `kv-cache/patch_*`) and anything removable from either side.

6. Report concisely:
   - incoming commits and the files they touch
   - table: file | patch status | mine vs upstream | recommendation | risk
   - net effect on the overlay (removed / kept / merged)

7. ASK the user (mandatory, `question` tool) how to proceed:
   - rebase onto origin/main and keep all patches
   - rebase and drop specific patches
   - rebase and merge specific files
   - review one file in detail first
   - do nothing / abort

8. Only after the user answers, perform exactly the chosen actions, then re-run
   `bash aijuus/upstream-check.sh` and confirm `overlap-files.txt` and
   `scratch-conflicts.txt` are empty. Do not commit unless the user asks.

## Notes

- This playbook is the source of truth. If Kilo's markdown frontmatter parser is
  working again, the same content can be dropped into `.kilo/command/aijuus-sync.md`
  to expose it as a `/aijuus-sync` slash command.
