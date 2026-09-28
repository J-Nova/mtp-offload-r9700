# Project instructions

## aijuus overlay

This repo keeps all personal work in `aijuus/`. Upstream-owned files are never committed
in modified form; those edits live as patches in `aijuus/patches/` and are applied with
`aijuus/apply.sh` (undone with `aijuus/revert.sh`).

- `aijuus/` additions are always present and never conflict with upstream.
- `aijuus/patches/*.patch` are replayed onto the working tree only for build/run.
- Revert the overlay before touching upstream history so pulls stay conflict-free.

## Upstream-sync workflow

When the user asks to sync with upstream, pull upstream and reconcile patches, or says
`sync aijuus` / `aijuus sync` / `/aijuus-sync`:

1. Read `aijuus/SYNC-WORKFLOW.md` and follow it exactly.
2. The workflow MUST end by asking the user what to do before making any change.
