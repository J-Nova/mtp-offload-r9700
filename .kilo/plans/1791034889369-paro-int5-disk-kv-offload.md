# Plan: KV-cache disk offloading for the int5 paro deployment

## Goal
Enable KV-cache **disk** offloading for `paro5-27B-int5` (the int5 / ParoQuant model, `topology=tp2`), using a **tiny CPU staging tier + the shared fs disk tier** (Mode B). The disk tier does the real work; the small CPU tier only stages `disk → GPU` promotions.

Chosen shape (user decision): *tiny CPU staging + disk* — NOT pure-disk (which would need a connector patch), NOT the full 16 GiB DP-model tier.

## Why this is a one-line change (context / findings)
- The whole offload stack is gated on `KV_OFFLOAD_GIB != 0`. The `--kv-offloading-size` flag both sizes the CPU (RAM) tier **and** selects the `OffloadingConnector`. The fs disk tier is a *secondary* tier added only when `KV_OFFLOAD_DISK_DIR` is set, and it promotes **into** the CPU staging tier (`disk → CPU → GPU`). So the disk tier is only reachable with a non-zero CPU tier.
- `paro5-27B-int5` currently sets `KV_OFFLOAD_GIB: "0"` → offload fully off (this is the only thing standing between paro and disk offload).
- Everything else is already in place and shared with the DP models:
  - `KV_OFFLOAD_DISK_DIR` defaults to `/kvcache/blocks` in the compose (`coolify-compose-2gpu.yml:199`); the paro registry entry does **not** override it, so it is inherited.
  - Backing store `/var/lib/radiance-kvcache/blocks` exists (60 GiB in use), reaper installed (`kvcache-reap.service/.timer`, cap `KVCACHE_MAX_GIB=60`, `MIN_AGE=90`).
  - `PYTHONHASHSEED=0` (compose:111) → content-deterministic block hashes, so the shared disk tier works and even allows cross-model prefix reuse.
  - The correctness patches (mixed-hit, eagle-groups, reconcile-reask, align-last-block, fs fan-out/invalidate/forget) are all applied via `apply-kv-patches.sh` and are env-gated **on** in the compose for every model, including paro.
- Data flow after the change: `KV_OFFLOAD_GIB=2` → `OVMODE=on` (entrypoint:148) → `OFF_ARG` is built with `--kv-offloading-size 2` + the fs secondary tier (entrypoint:435-450). Head cap defaults to `off` in Mode B (disk holds the overflow). The 2 GiB CPU tier is a small fast cache + staging buffer; the disk holds the bulk.

## Change
1. **`aijuus/model-registry.json`** — `paro5-27B-int5.server_env`:
   - `"KV_OFFLOAD_GIB": "0"` → `"KV_OFFLOAD_GIB": "2"`  (recommended default; tunable 2–8 GiB)
   - Update the trailing `note` to record that disk offload is now on (tiny 2 GiB CPU staging + shared fs tier) and why (disk reuse without a large pinned CPU tier).
   - No `KV_OFFLOAD_DISK_DIR` needed in the entry — the compose default `/kvcache/blocks` is inherited.
2. **(Optional, cosmetic) `aijuus/kv-offload/ops/entrypoint.sh:410`** — the comment "Switching to a no-offload model (tp2, e.g. paro5-27B-int5, KV_OFFLOAD_GIB=0)" is now stale (paro has offload). Generalize it to "switching to a model with a different rank set / no offload". The cross-rank sweep logic (413-417) stays as-is and is still required.

No compose change, no reaper change, no patch change.

## Sizing rationale (2 GiB default)
- One paro chunk = 8192 tokens; a CPU block ≈ 2× GPU bytes/token (~76.8 KB/token) → one chunk ≈ ~0.6 GiB. 2 GiB stages a couple of chunks and doubles as a small hot-prefix cache.
- The disk tier holds the full (long) prefixes; the CPU tier size only bounds how much of a long prefix is restored from disk *per window*. 2 GiB is fine for the disk tier's real value — shared/short prefixes (system prompts, common conversation heads) — and keeps pinned RAM minimal.
- Tune up (4–8 GiB) only if the workload is dominated by long shared prefixes (>~28k tokens) and restores feel slow; A/B in the Grafana "Offload tuning A/B" row.

## Risks (validate before trusting)
- **Offload + TP=2 is a new combination.** Offload was validated on the TP=1 DP models. The connector's CPU tier is keyed by `engine_id` (`radrank0`); a TP=2 engine has 2 TP ranks sharing that id, so how the single CPU region handles the two sharded KV views is unverified. This is the main risk.
- **Offload patches on the int5 path.** The correctness patches are validated for the MXFP4 DP models; paro uses `RADIANCE_KV_GROUP_OPT=1` and a different KV-group structure. The patches are model-agnostic (they act on KV blocks, not weights) so they should apply, but this is unverified.
- **Small staging tier** limits per-window disk-restore length (see sizing). Acceptable for the intended use.

## Validation
1. **Boot logs** (after `POST /load {"model":"paro5-27B-int5"}` / redeploy):
   - `docker logs <vllm> | grep -E "KV offload ON|fs KV tier ON|fs tier applied"` → expect `cpu tier=2GiB` and the fs tier line with `root_dir=/kvcache/blocks`.
   - Confirm **no** `FATAL: async scheduling + KV offload` (default `VLLM_ASYNC_SCHEDULING=0`, so the gate is not hit).
   - Confirm the stale-RAM-tier sweep ran and no leaked `/dev/shm/vllm_offload_*.mmap` remains.
2. **Correctness gate (required before trusting):** with paro loaded, run `aijuus/kv-offload/ops/turnbench.py` (multi-turn cached-vs-cold, token+logprob bit-identical) and/or `equivbench.py` (KV-hit vs full recompute). This is what de-risks the offload+TP=2 and int5-path risks above.
3. **Disk tier health:** `radiance_kvcache_bytes` / `radiance_kvcache_blocks` (Grafana / kvcache-exporter) grow as prefixes are stored; reaper keeps it under 60 GiB; `journalctl -u kvcache-reap` clean.
4. **Shape label:** `radiance_serve_info{...offload_mode="on"}` flips for the paro instance (node-exporter textfile).

## Rollback
Set `KV_OFFLOAD_GIB` back to `"0"` in the registry and redeploy. The shared disk blocks remain (harmless, reaper-managed). No other state to undo.

## Out of scope
- Pure-disk (zero CPU RAM) — would require patching the in-image `OffloadingConnector`; rejected in favor of the tiny-staging shape.
- Resizing the disk backing store / reaper cap (already 60 GiB, shared).
- Per-model `KV_OFFLOAD_DISK_DIR` override (compose default suffices).
