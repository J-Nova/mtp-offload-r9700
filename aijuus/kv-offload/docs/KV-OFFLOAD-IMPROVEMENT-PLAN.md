# KV Offload Improvement Plan (Detailed)

Based on comprehensive file-by-file, function-by-function analysis of upstream PR #52 (zzpanic, commit a5036dc) vs our aijuus implementation.

See `KV-OFFLOAD-COMPREHENSIVE-ANALYSIS.md` for the full file-by-file breakdown.

## Current State

### What we have (aijuus)
- **Interface**: `KV_OFFLOAD_GIB`, `KV_OFFLOAD_DISK_DIR`/`KV_OFFLOAD_DISK_HOST_DIR`, `KV_OFFLOAD_EVICTION_POLICY`, `KV_OFFLOAD_READ_THREADS` (32), `KV_OFFLOAD_WRITE_THREADS` (16), `KV_OFFLOAD_HEAD_CAP`
- **Patches** (7):
  - `patch_offload_debug.py` — basic logging
  - `patch_offload_eagle_fallback.py` — hybrid (Mamba/SWA) offload correctness (unique to us)
  - `patch_offload_eagle_groups.py` — EAGLE group annotation (port of upstream)
  - `patch_offload_fs_tier.py` — fs tier with fanout, lookup invalidate, failed-load forget (port of upstream)
  - `patch_offload_head_cap.py` — head-only CPU retention (unique to us)
  - `patch_offload_swa_touch.py` — SWA align/touch (port of upstream)
  - `patch_offload_trace.py` — deep diagnostic (unique to us)
- **Ops**: `kvcache-reap.sh` (more sophisticated than upstream: age + byte cap + capacity + emergency stages, hard-cap enforcement with exit 2, MIN_AGE_MIN=90), `install-reaper.sh`, systemd service/timer
- **Unique features**: Head cap (Mode A auto-fit CPU / Mode B no cap for disk), configurable threads/policy, `offload_prompt_only: true`, eagle fallback, trace

### What upstream has that we're missing
- **Patches** (8 missing):
  - `patch_offload_mixed_hit.py` — **FATAL**: mixed-hit safety
  - `patch_offload_instrumentation.py` — **FATAL**: store/lookup Prometheus metrics
  - `patch_mamba_stride.py` — Mamba store stride (we have env var, need patch)
  - `patch_reconcile_reask.py` — hybrid reconcile optimization
  - `patch_sched_align_last_block.py` — prompt last block alignment
  - `patch_offload_lookup_metrics.py` — 13 lookup counters
  - `patch_offload_debug_instrument.py` — bounded JSON-line event sink
  - `patch_offload_tier_report.py` — tier reporting
  - `patch_offload_promotion_wallclock.py` — promotion refusal/wallclock
  - `patch_offload_miss_deferral_metrics.py` — post-deferral outcomes
- **Tools**: `kvwatch.py`, `turnbench.py`, `tierbench.py`, `equivbench.py`, `_kvinstr.py`
- **Features**: Stale RAM tier cleanup on startup

## Detailed Improvement Plan

### Phase 1: Critical Correctness Patches (FATAL)

These are required for bit-identical serving and must be integrated first.

#### 1. `patch_offload_mixed_hit.py`

**Purpose**: Makes the native `OffloadingConnector` safe when the KV group prefix hit lags the request's `num_computed_tokens`. Without this patch, a request can be served from a prefix that the GPU already evicted, causing bit-identical output to break.

**Functions to port**:
- `patch_mixed_hit()` in `scheduler.py`:
  - Patches `OffloadingConnectorScheduler._lookup()` to handle the case where `num_hit_chunks * tokens_per_chunk < num_computed_tokens`
  - When the CPU tier has a longer prefix than what the GPU reports, the CPU tier is used instead
  - Prevents the "mixed hit" scenario where part of the prefix comes from GPU and part from CPU, which can cause token mismatches

- `_mixed_hit_safe()` in `scheduler.py`:
  - Helper that determines whether a mixed hit is safe
  - Checks if the CPU tier has a contiguous prefix that extends beyond the GPU's reported boundary
  - Returns the CPU tier's hit length instead of the GPU's when safe

**Integration**:
- Gate: `RADIANCE_OFFLOAD_MIXED_HIT` (default 1 when offload is on)
- Must be applied before any other offload patches
- Validate with `turnbench.py` after integration

#### 2. `patch_offload_instrumentation.py`

**Purpose**: Makes the store path observable by widening histograms and adding evictable/free metrics. Required for the mixed-hit patch to report correctly.

**Functions to port**:
- `patch_instrumentation()` in `cpu/manager.py`:
  - Adds Prometheus metrics for the store path:
    - `vllm:kv_offload_store_bytes_total` — Total bytes stored
    - `vllm:kv_offload_store_chunks_total` — Total chunks stored
    - `vllm:kv_offload_evictable_bytes` — Currently evictable bytes
    - `vllm:kv_offload_free_bytes` — Currently free bytes
    - `vllm:kv_offload_store_latency_seconds` — Store latency histogram

- `patch_lookup_instrumentation()` in `cpu/manager.py`:
  - Adds Prometheus metrics for the lookup path:
    - `vllm:kv_offload_lookup_hits_total` — Total lookup hits
    - `vllm:kv_offload_lookup_misses_total` — Total lookup misses
    - `vllm:kv_offload_lookup_pending_total` — Total pending lookups
    - `vllm:kv_offload_lookup_latency_seconds` — Lookup latency histogram

**Integration**:
- Gate: `RADIANCE_OFFLOAD_INSTRUMENTATION` (default 1 when offload is on)
- Required for mixed-hit reporting
- Validate with `turnbench.py` after integration

#### 3. `patch_mamba_stride.py`

**Purpose**: Stores the Mamba/GDN groups every Nth chunk instead of every chunk. This is the density lever for the CPU tier.

**Functions to port**:
- `patch_mamba_stride()` in `scheduler.py`:
  - Patches the store path to only store Mamba/GDN groups every Nth chunk
  - A Mamba group holds ONE recurrent state, not a per-token history
  - The load path fetches exactly one Mamba chunk per request (the last one)
  - The store path writes a fresh snapshot of all six Mamba/GDN groups at every chunk boundary, which is a 4.17x amplification
  - The patch keeps every Nth Mamba snapshot (N = `RADIANCE_MAMBA_STORE_STRIDE`)

- `resolve_mamba_align_size()` in `scheduler.py`:
  - Returns the Mamba alignment size, which is N * tokens_per_chunk
  - The hit window must land on the same grid as the store grid
  - This rounding creates the dead zone: a prefix shorter than N * tokens_per_chunk gets zero external hit

**Integration**:
- Gate: `RADIANCE_MAMBA_STORE_STRIDE` (default 4 when offload is on)
- We already have the env var but need the actual patch
- Validate with `turnbench.py` after integration

### Phase 2: Behavioral Correctness Patches

These improve correctness for hybrid (Mamba/SWA) models.

#### 4. `patch_reconcile_reask.py`

**Purpose**: Stops hybrid Mamba+attention requests from recomputing prefix offload tier holds.

**Functions to port**:
- `patch_reconcile_reask()` in `scheduler.py`:
  - Patches the reconcile path to avoid recomputing prefix offload tier holds for hybrid Mamba+attention requests
  - Hybrid Mamba+attention requests have a different reconcile path than pure attention requests
  - Prevents unnecessary recomputation and improves performance

**Integration**:
- Gate: `RADIANCE_RECONCILE_REASK` (default 1 when offload is on)
- Validate with `turnbench.py` after integration

#### 5. `patch_sched_align_last_block.py`

**Purpose**: Stops prompt's final prefill chunk at last full block boundary.

**Functions to port**:
- `patch_sched_align_last_block()` in `scheduler.py`:
  - Patches the scheduler to align the prompt's final prefill chunk at the last full block boundary
  - The prompt's final prefill chunk may not align with the block boundary
  - Improves the efficiency of the offload path

**Integration**:
- Gate: `RADIANCE_ALIGN_PROMPT_LAST_BLOCK` (default 1 when offload is on)
- Validate with `turnbench.py` after integration

### Phase 3: Operational Safety

#### 6. Stale RAM tier cleanup

**Purpose**: Cleans up `/dev/shm/vllm_offload_*.mmap` left by stopped containers on startup.

**Implementation**:
- Add to `serve-mxfp4.patch` in the preflight section
- Uses `fuser`/`lsof` to check if files are held by live processes before removing
- Code pattern (from upstream):
  ```bash
  shopt -s nullglob
  for _f in /dev/shm/vllm_offload_*.mmap; do
    if command -v fuser >/dev/null 2>&1; then fuser -s "$_f" 2>/dev/null && continue
    elif command -v lsof >/dev/null 2>&1; then lsof -t -- "$_f" >/dev/null 2>&1 && continue
    else continue; fi
    echo "[kv-cache] removing stale RAM tier $(basename "$_f") ($(( $(stat -c %s "$_f") >> 20 )) MiB)" >&2
    [ -n "${DRY_RUN:-}" ] || rm -f -- "$_f"
  done
  shopt -u nullglob
  ```

**Integration**:
- Add to the preflight section of `serve-mxfp4.patch`
- Only runs when `KV_OFFLOAD_GIB` is set
- Prevents stale RAM tier files from consuming memory and preventing new boots

**HA correction (2026-09-28)**: the fuser/lsof sweep above cannot see the peer from
inside the container (separate PID namespace, shared `ipc: host` /dev/shm), so HA mode
passes a per-instance `ENG_ID` (`radrank0`/`radrank1`). The first implementation skipped
*every* `radrank[01]*.mmap` when an `ENG_ID` was given, which also skipped the instance's
OWN leaked region -- so the next start still died in `SharedOffloadRegion`
(`FileExistsError` -> `_wait_for_file_size` timeout). Correct behavior: with `ENG_ID`,
remove exactly `/dev/shm/vllm_offload_<ENG_ID>.mmap` and never the peer's; without it,
fall back to the fuser/lsof sweep. See `ops/clean-stale-ram-tier.sh`.

#### 7. PYTHONHASHSEED=0 for disk mode

**Purpose**: Block filenames are content hashes, so consistent hashing is required.

**Current state**: We already have `-e PYTHONHASHSEED=0` in our patch. Verify it's always set when disk tier is on.

**Integration**:
- Ensure `PYTHONHASHSEED=0` is always passed when `KV_OFFLOAD_DISK_DIR` is set
- No changes needed if already present

### Phase 4: Monitoring & Verification Tools

#### 8. `kvwatch.py`

**Purpose**: Live KV-cache summary for watch. Reads stock vLLM metrics from `/metrics` endpoint.

**Functions to port**:
- `main()`: Main entry point, reads metrics from `/metrics` endpoint, displays live hit rates
- `fetch_metrics()`: Makes HTTP GET request to `/metrics`, parses Prometheus metrics format
- `display_metrics()`: Formats metrics for display (hit rates, bytes loaded/stored, deferred lookups)

**What it shows**:
- GPU and offload-tier hit rates (lifetime and since last refresh)
- Bytes the tier loaded and stored
- Deferred lookups
- Transfer buffer utilization

**Integration**:
- Port from `kv-cache/kvwatch.py` to `aijuus/kv-offload/ops/kvwatch.py`
- Usage: `watch -n 5 python3 aijuus/kv-offload/ops/kvwatch.py`
- Reads from `http://127.0.0.1:$PORT/metrics` (`KVWATCH_METRICS` overrides)

#### 9. `turnbench.py` + `tierbench.py` + `equivbench.py`

**Purpose**: Bit-identical correctness gate and tier attribution tools.

**`turnbench.py`**:
- `main()`: Runs a multi-turn workload (3 sessions x 7 turns), compares cached vs cold
- `run_turn()`: Makes HTTP POST request to vLLM API
- `compare_results()`: Compares token-for-token and logprob-for-logprob
- `report()`: Reports bit-identical correctness

**`tierbench.py`**:
- `main()`: Runs a deterministic workload, attributes KV hits to specific tiers
- `run_workload()`: Makes HTTP POST requests to vLLM API
- `attribute_tiers()`: Parses response to determine which tier served the KV
- `report()`: Reports tier attribution

**`equivbench.py`**:
- `main()`: Runs workload with KV cache hits and with full recompute, compares results
- `run_with_cache()`: Makes HTTP POST requests with prefix caching enabled
- `run_without_cache()`: Makes HTTP POST requests with prefix caching disabled
- `compare_results()`: Compares token-for-token and logprob-for-logprob
- `report()`: Reports equivalence

**Integration**:
- Port from `kv-cache/` to `aijuus/kv-offload/ops/`
- `turnbench.py` is the validation gate before deploying offload changes
- `tierbench.py` and `equivbench.py` are optional but useful for deeper validation

#### 10. `_kvinstr.py`

**Purpose**: Bounded JSON-line event sink.

**Functions to port**:
- `KVInstrSink` class:
  - Uses a bounded queue to store events
  - Events are written to a JSON-line file
  - The queue is bounded to prevent memory exhaustion
  - Events include timestamp, request ID, group index, block hash, and event type

- `write_event()`: Adds event to bounded queue, drops oldest if full, writes to JSON-line file
- `flush()`: Writes all pending events to JSON-line file, clears queue

**Integration**:
- Port from `kv-cache/_kvinstr.py` to `aijuus/kv-offload/tools/_kvinstr.py`
- Used by `patch_offload_debug_instrument.py` (Phase 5)

### Phase 5: Metrics & Instrumentation (Optional)

These provide deeper visibility but are not required for correctness.

#### 11. `patch_offload_lookup_metrics.py`

**Purpose**: Makes the lookup path observable by adding 13 new counters.

**Functions to port**:
- `patch_lookup_metrics()` in `scheduler.py`:
  - Adds 13 Prometheus counters for the lookup path:
    - `vllm:kv_offload_lookup_total` — Total lookups
    - `vllm:kv_offload_lookup_hit_total` — Total lookup hits
    - `vllm:kv_offload_lookup_miss_total` — Total lookup misses
    - `vllm:kv_offload_lookup_deferred_total` — Total deferred lookups
    - `vllm:kv_offload_lookup_promoted_total` — Total promoted lookups
    - `vllm:kv_offload_lookup_evicted_total` — Total evicted lookups
    - `vllm:kv_offload_lookup_stale_total` — Total stale lookups
    - `vllm:kv_offload_lookup_failed_total` — Total failed lookups
    - `vllm:kv_offload_lookup_timeout_total` — Total timeout lookups
    - `vllm:kv_offload_lookup_retry_total` — Total retry lookups
    - `vllm:kv_offload_lookup_cancelled_total` — Total cancelled lookups
    - `vllm:kv_offload_lookup_pending_total` — Total pending lookups
    - `vllm:kv_offload_lookup_completed_total` — Total completed lookups

**Integration**:
- Gate: `RADIANCE_OFFLOAD_LOOKUP_METRICS` (default 0 when offload is on)
- Optional, non-fatal

#### 12. `patch_offload_debug_instrument.py`

**Purpose**: Records lookup/store events to a bounded JSON-line sink.

**Functions to port**:
- `patch_debug_instrument()` in `scheduler.py`:
  - Adds debug instrumentation to the lookup/store path
  - Records lookup/store events to a bounded JSON-line sink (via `_kvinstr.py`)
  - Events include timestamp, request ID, group index, block hash, and event type

**Integration**:
- Gate: `RADIANCE_OFFLOAD_DEBUG_INSTRUMENT` (default 0 when offload is on)
- Requires `_kvinstr.py` (Phase 4)
- Optional, non-fatal

#### 13. `patch_offload_tier_report.py`

**Purpose**: Adds tier reporting instrumentation.

**Functions to port**:
- `patch_tier_report()` in `tiering/manager.py`:
  - Reports tier usage, hit rates, and latency
  - The report is written to a JSON file
  - Includes CPU and fs tier metrics

**Integration**:
- Gate: `RADIANCE_OFFLOAD_TIER_REPORT` (default 0 when offload is on)
- Optional, non-fatal

#### 14. `patch_offload_promotion_wallclock.py`

**Purpose**: Attributes promotion refusals and wallclock reanchoring.

**Functions to port**:
- `patch_promotion_wallclock()` in `tiering/manager.py`:
  - Attributes promotion refusals to specific causes
  - Reanchors the wallclock to prevent drift
  - Includes promotion latency and refusal reasons

**Integration**:
- Gate: `RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK` (default 0 when offload is on)
- Optional, non-fatal

#### 15. `patch_offload_miss_deferral_metrics.py`

**Purpose**: Instruments post-deferral outcomes.

**Functions to port**:
- `patch_miss_deferral_metrics()` in `scheduler.py`:
  - Instruments the outcomes of deferred lookups
  - Tracks whether deferred lookups eventually hit or miss
  - Includes deferral latency and outcome distribution

**Integration**:
- Gate: `RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS` (default 0 when offload is on)
- Optional, non-fatal

### Phase 6: Integration

#### 16. Update `serve-mxfp4.patch`

**Changes**:
- Add the new patch invocations to the container prelude:
  ```bash
  # Phase 1 (FATAL)
  if [ "${RADIANCE_OFFLOAD_MIXED_HIT:-1}" != 0 ]; then python3 aijuus/kv-offload/patches/patch_offload_mixed_hit.py; fi
  if [ "${RADIANCE_OFFLOAD_INSTRUMENTATION:-1}" != 0 ]; then python3 aijuus/kv-offload/patches/patch_offload_instrumentation.py; fi
  if [ "${RADIANCE_MAMBA_STORE_STRIDE:-4}" != 1 ]; then python3 aijuus/kv-offload/patches/patch_mamba_stride.py; fi
  
  # Phase 2 (behavioral)
  if [ "${RADIANCE_RECONCILE_REASK:-1}" != 0 ]; then python3 aijuus/kv-offload/patches/patch_reconcile_reask.py; fi
  if [ "${RADIANCE_ALIGN_PROMPT_LAST_BLOCK:-1}" != 0 ]; then python3 aijuus/kv-offload/patches/patch_sched_align_last_block.py; fi
  
  # Phase 5 (optional metrics)
  if [ "${RADIANCE_OFFLOAD_LOOKUP_METRICS:-0}" = 1 ]; then python3 aijuus/kv-offload/patches/patch_offload_lookup_metrics.py; fi
  if [ "${RADIANCE_OFFLOAD_DEBUG_INSTRUMENT:-0}" = 1 ]; then python3 aijuus/kv-offload/patches/patch_offload_debug_instrument.py; fi
  if [ "${RADIANCE_OFFLOAD_TIER_REPORT:-0}" = 1 ]; then python3 aijuus/kv-offload/patches/patch_offload_tier_report.py; fi
  if [ "${RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK:-0}" = 1 ]; then python3 aijuus/kv-offload/patches/patch_offload_promotion_wallclock.py; fi
  if [ "${RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS:-0}" = 1 ]; then python3 aijuus/kv-offload/patches/patch_offload_miss_deferral_metrics.py; fi
  ```
- Add new env vars to the `-e` list:
  ```bash
  ${RADIANCE_OFFLOAD_MIXED_HIT:+-e RADIANCE_OFFLOAD_MIXED_HIT="$RADIANCE_OFFLOAD_MIXED_HIT"} \
  ${RADIANCE_OFFLOAD_INSTRUMENTATION:+-e RADIANCE_OFFLOAD_INSTRUMENTATION="$RADIANCE_OFFLOAD_INSTRUMENTATION"} \
  ${RADIANCE_RECONCILE_REASK:+-e RADIANCE_RECONCILE_REASK="$RADIANCE_RECONCILE_REASK"} \
  ${RADIANCE_ALIGN_PROMPT_LAST_BLOCK:+-e RADIANCE_ALIGN_PROMPT_LAST_BLOCK="$RADIANCE_ALIGN_PROMPT_LAST_BLOCK"} \
  ${RADIANCE_OFFLOAD_LOOKUP_METRICS:+-e RADIANCE_OFFLOAD_LOOKUP_METRICS="$RADIANCE_OFFLOAD_LOOKUP_METRICS"} \
  ${RADIANCE_OFFLOAD_DEBUG_INSTRUMENT:+-e RADIANCE_OFFLOAD_DEBUG_INSTRUMENT="$RADIANCE_OFFLOAD_DEBUG_INSTRUMENT"} \
  ${RADIANCE_OFFLOAD_TIER_REPORT:+-e RADIANCE_OFFLOAD_TIER_REPORT="$RADIANCE_OFFLOAD_TIER_REPORT"} \
  ${RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK:+-e RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK="$RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK"} \
  ${RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS:+-e RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS="$RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS"} \
  ```
- Add stale RAM cleanup to preflight (see Phase 3)
- Keep our head cap, configurable threads, and eviction policy features

#### 17. Update `coolify-compose-2gpu.yml`

**Changes**:
- Add new env vars for the new patches:
  ```yaml
  RADIANCE_OFFLOAD_MIXED_HIT: "1"
  RADIANCE_OFFLOAD_INSTRUMENTATION: "1"
  RADIANCE_MAMBA_STORE_STRIDE: "4"
  RADIANCE_RECONCILE_REASK: "1"
  RADIANCE_ALIGN_PROMPT_LAST_BLOCK: "1"
  ```
- Keep our head cap and thread settings:
  ```yaml
  KV_OFFLOAD_HEAD_CAP: "auto"
  KV_OFFLOAD_READ_THREADS: "32"
  KV_OFFLOAD_WRITE_THREADS: "16"
  KV_OFFLOAD_EVICTION_POLICY: "lru"
  ```

#### 18. Update documentation

**Changes**:
- Document new env vars and their defaults in README.md
- Add `kvwatch.py` usage to ops documentation
- Add `turnbench.py` as the correctness validation gate
- Update `MXFP4-NOTES.md` with the new offload capabilities

## What We Keep (Our Advantages)

- **Head cap** (`KV_OFFLOAD_HEAD_CAP`): Mode A auto-fit CPU / Mode B no cap for disk — upstream doesn't have this
- **Configurable threads**: `KV_OFFLOAD_READ_THREADS` (32) / `KV_OFFLOAD_WRITE_THREADS` (16) vs upstream's hardcoded 8/4
- **Configurable eviction policy**: `KV_OFFLOAD_EVICTION_POLICY` (default lru)
- **More sophisticated reaper**: Byte cap stage (A2), hard-cap enforcement with exit 2, more conservative MIN_AGE_MIN (90 vs 15)
- **Eagle fallback patch**: `patch_offload_eagle_fallback.py` for hybrid correctness
- **Trace patch**: `patch_offload_trace.py` for deep diagnostics

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Mixed-hit patch | Low (upstream validated) | Run turnbench.py before deploy |
| Instrumentation patch | Low (upstream validated) | Run turnbench.py before deploy |
| Mamba stride patch | Low (we already have env var) | Run turnbench.py before deploy |
| Reconcile reask patch | Low (upstream validated) | Run turnbench.py before deploy |
| Align last block patch | Low (upstream validated) | Run turnbench.py before deploy |
| Stale RAM cleanup | Medium (could delete active buffers) | fuser/lsof check before delete |
| Metrics patches | Low (observability only) | Non-fatal, degrade gracefully |

## Execution Order

1. Phase 1 (FATAL patches) → validate with turnbench.py
2. Phase 2 (behavioral patches) → validate with turnbench.py
3. Phase 3 (operational safety) → test stale cleanup in isolation
4. Phase 4 (monitoring tools) → deploy kvwatch.py, turnbench.py
5. Phase 5 (metrics) → optional, add as needed
6. Phase 6 (integration) → update compose and docs

## Expected Outcome

After implementing all phases, our KV offload implementation will:
- Have bit-identical serving (via mixed-hit + instrumentation patches)
- Have better performance for hybrid models (via reconcile reask + align last block)
- Have operational safety (stale RAM cleanup)
- Have live monitoring (kvwatch.py)
- Have correctness validation (turnbench.py)
- Have deeper visibility (optional metrics patches)
- Retain all our unique advantages (head cap, configurable threads/policy, sophisticated reaper)
