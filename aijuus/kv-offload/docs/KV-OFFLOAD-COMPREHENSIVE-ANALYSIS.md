# Comprehensive KV Offload Analysis: Upstream (a5036dc) vs aijuus

This document provides a detailed file-by-file, function-by-function comparison of the upstream KV cache offload implementation (commit a5036dc) and the aijuus implementation, identifying specific code/logic to port.

## Table of Contents

1. [patch_offload_mixed_hit.py](#1-patch_offload_mixed_hitpy)
2. [patch_offload_instrumentation.py](#2-patch_offload_instrumentationpy)
3. [patch_eagle_groups.py](#3-patch_eagle_groupspy)
4. [patch_mamba_stride.py](#4-patch_mamba_stridepy)
5. [patch_reconcile_reask.py](#5-patch_reconcile_reaskpy)
6. [patch_swa_align_touch.py](#6-patch_swa_align_touchpy)
7. [patch_sched_align_last_block.py](#7-patch_sched_align_last_blockpy)
8. [patch_offload_lookup_metrics.py](#8-patch_offload_lookup_metricspy)
9. [patch_offload_debug_instrument.py](#9-patch_offload_debug_instrumentpy)
10. [patch_offload_fs_fanout.py](#10-patch_offload_fs_fanoutpy)
11. [patch_offload_tier_report.py](#11-patch_offload_tier_reportpy)
12. [patch_offload_promotion_wallclock.py](#12-patch_offload_promotion_wallclockpy)
13. [patch_lookup_invalidate.py](#13-patch_lookup_invalidatepy)
14. [patch_fs_failed_load.py](#14-patch_fs_failed_loadpy)
15. [patch_offload_miss_deferral_metrics.py](#15-patch_offload_miss_deferral_metricspy)
16. [kvwatch.py](#16-kvwatchpy)
17. [_kvinstr.py](#17-_kvinstrpy)
18. [kvcache-reap.sh](#18-kvcache-reapsh)
19. [turnbench.py](#19-turnbenchpy)
20. [tierbench.py](#20-tierbenchpy)
21. [equivbench.py](#21-equivbenchpy)
22. [serve-mxfp4.sh](#22-serve-mxfp4sh)

---

## 1. patch_offload_mixed_hit.py

### Purpose/Overview
Makes the native `OffloadingConnector` safe when the KV group prefix hit lags the request's `num_computed_tokens`. Without this patch, a request can be served from a prefix that the GPU already evicted, causing bit-identical output to break.

**UPSTREAM STATUS: FATAL** — Required for correct bit-identical serving.

### Function-by-Function Breakdown

#### `patch_mixed_hit()`
- **Location**: `scheduler.py`
- **What it does**: Patches `OffloadingConnectorScheduler._lookup()` to handle the case where the KV group's prefix hit (`num_hit_chunks`) is less than what the request's `num_computed_tokens` suggests.
- **Key logic**:
  - When `num_hit_chunks * tokens_per_chunk < num_computed_tokens`, the connector would normally serve from the GPU boundary, but the GPU may have already evicted that prefix.
  - The patch ensures that if the CPU tier has a longer prefix than what the GPU reports, the CPU tier is used instead.
  - This prevents the "mixed hit" scenario where part of the prefix comes from GPU and part from CPU, which can cause token mismatches.

#### `_mixed_hit_safe()`
- **Location**: `scheduler.py`
- **What it does**: Helper function that determines whether a mixed hit is safe.
- **Key logic**:
  - Checks if the CPU tier has a contiguous prefix that extends beyond the GPU's reported boundary.
  - If so, it returns the CPU tier's hit length instead of the GPU's.
  - This ensures that the served prefix is always contiguous and bit-identical.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch. Our `patch_offload_eagle_fallback.py` and `patch_offload_eagle_groups.py` handle the EAGLE group mislabeling issue, but they do not address the mixed-hit safety issue.

**Gap**: Our implementation is vulnerable to the mixed-hit scenario where the GPU evicts a prefix that the CPU tier still has. This can cause bit-identical output to break.

### Specific Improvements to Port

1. **Port `patch_mixed_hit()`**: Add the mixed-hit safety logic to our `scheduler.py` patch.
2. **Gate**: `RADIANCE_OFFLOAD_MIXED_HIT` (default 1 when offload is on).
3. **Integration**: This patch must be applied before any other offload patches to ensure bit-identical serving.

### Integration Notes
- This patch is FATAL for bit-identical serving.
- Must be integrated before deploying offload changes.
- Validate with `turnbench.py` after integration.

---

## 2. patch_offload_instrumentation.py

### Purpose/Overview
Makes the store path observable by widening histograms and adding evictable/free metrics. This is required for the mixed-hit patch to report correctly.

**UPSTREAM STATUS: FATAL** — Mixed-hit reports through this patch.

### Function-by-Function Breakdown

#### `patch_instrumentation()`
- **Location**: `cpu/manager.py`
- **What it does**: Adds Prometheus metrics for the store path.
- **Key metrics added**:
  - `vllm:kv_offload_store_bytes_total` — Total bytes stored
  - `vllm:kv_offload_store_chunks_total` — Total chunks stored
  - `vllm:kv_offload_evictable_bytes` — Currently evictable bytes
  - `vllm:kv_offload_free_bytes` — Currently free bytes
  - `vllm:kv_offload_store_latency_seconds` — Store latency histogram

#### `patch_lookup_instrumentation()`
- **Location**: `cpu/manager.py`
- **What it does**: Adds Prometheus metrics for the lookup path.
- **Key metrics added**:
  - `vllm:kv_offload_lookup_hits_total` — Total lookup hits
  - `vllm:kv_offload_lookup_misses_total` — Total lookup misses
  - `vllm:kv_offload_lookup_pending_total` — Total pending lookups
  - `vllm:kv_offload_lookup_latency_seconds` — Lookup latency histogram

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_debug.py` which provides basic logging, but we do NOT have the Prometheus instrumentation that upstream provides.

**Gap**: Our implementation lacks the detailed metrics that upstream provides for monitoring the offload path.

### Specific Improvements to Port

1. **Port `patch_instrumentation()`**: Add the Prometheus metrics to our `cpu/manager.py` patch.
2. **Port `patch_lookup_instrumentation()`**: Add the lookup metrics to our `cpu/manager.py` patch.
3. **Gate**: `RADIANCE_OFFLOAD_INSTRUMENTATION` (default 1 when offload is on).
4. **Integration**: This patch is required for the mixed-hit patch to report correctly.

### Integration Notes
- This patch is FATAL for mixed-hit reporting.
- Must be integrated before deploying offload changes.
- Validate with `turnbench.py` after integration.

---

## 3. patch_eagle_groups.py

### Purpose/Overview
Annotates the EAGLE/MTP draft KV group positionally for hybrid (Mamba + attention) models. This is the correct fix for the EAGLE group mislabeling issue.

### Function-by-Function Breakdown

#### `_annotate_eagle_groups_deepseek_v4()`
- **Location**: `kv_cache_utils.py`
- **What it does**: Annotates the EAGLE/MTP draft KV group.
- **Key logic**:
  - The draft model's attention layer is registered last, so flag whichever group holds the last layer.
  - This is a fact about how vLLM registers a draft model, not about DeepSeek.
  - The patch drops the `model_version == "deepseek_v4"` early return and calls the annotator on the general hybrid page-size path too.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_eagle_groups.py` which is a port of this upstream patch. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gate: `RADIANCE_OFFLOAD_EAGLE_GROUPS` (default 1 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 4. patch_mamba_stride.py

### Purpose/Overview
Stores the Mamba/GDN groups every Nth chunk instead of every chunk. This is the density lever for the CPU tier.

### Function-by-Function Breakdown

#### `patch_mamba_stride()`
- **Location**: `scheduler.py`
- **What it does**: Patches the store path to only store Mamba/GDN groups every Nth chunk.
- **Key logic**:
  - A Mamba group holds ONE recurrent state, not a per-token history.
  - The load path fetches exactly one Mamba chunk per request (the last one).
  - The store path writes a fresh snapshot of all six Mamba/GDN groups at every chunk boundary, which is a 4.17x amplification.
  - The patch keeps every Nth Mamba snapshot (N = RADIANCE_MAMBA_STORE_STRIDE).

#### `resolve_mamba_align_size()`
- **Location**: `scheduler.py`
- **What it does**: Returns the Mamba alignment size, which is N * tokens_per_chunk.
- **Key logic**:
  - The hit window must land on the same grid as the store grid.
  - This rounding creates the dead zone: a prefix shorter than N * tokens_per_chunk gets zero external hit.

### Comparison to Our Implementation

**Our implementation**: We have `patch_mamba_stride.py` which is a port of this upstream patch. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gate: `RADIANCE_MAMBA_STORE_STRIDE` (default 1 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 5. patch_reconcile_reask.py

### Purpose/Overview
Stops hybrid Mamba+attention requests from recomputing prefix offload tier holds.

### Function-by-Function Breakdown

#### `patch_reconcile_reask()`
- **Location**: `scheduler.py`
- **What it does**: Patches the reconcile path to avoid recomputing prefix offload tier holds for hybrid Mamba+attention requests.
- **Key logic**:
  - Hybrid Mamba+attention requests have a different reconcile path than pure attention requests.
  - The patch ensures that the prefix offload tier holds are not recomputed for hybrid requests.
  - This prevents unnecessary recomputation and improves performance.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch.

**Gap**: Our implementation is missing the reconcile reask optimization for hybrid Mamba+attention requests.

### Specific Improvements to Port

1. **Port `patch_reconcile_reask()`**: Add the reconcile reask logic to our `scheduler.py` patch.
2. **Gate**: `RADIANCE_RECONCILE_REASK` (default 1 when offload is on).
3. **Integration**: This patch improves performance for hybrid Mamba+attention requests.

### Integration Notes
- This patch improves performance for hybrid Mamba+attention requests.
- Validate with `turnbench.py` after integration.

---

## 6. patch_swa_align_touch.py

### Purpose/Overview
Stores SWA chunks only where hit can land, aligns touch behavior. This is the partial-reuse fix.

### Function-by-Function Breakdown

#### `patch_swa_align()`
- **Location**: `scheduler.py`
- **What it does**: Patches the store path to align SWA chunks with the Mamba grid.
- **Key logic**:
  - The drafter's stored chunks must line up with the chunks a grid-point hit reads.
  - The patch raises the sliding-window store alignment to the Mamba grid when it is a whole multiple of the full-attention chunk.
  - It judges reachability on the ABSOLUTE grid.

#### `patch_touch()`
- **Location**: `scheduler.py`
- **What it does**: Patches the touch path to touch all groups in position order.
- **Key logic**:
  - A prefix's Mamba/drafter snapshots age out from the head while their attention keys stay held.
  - The patch touches ONE list carrying EVERY group's keys, sorted by chunk end position, head to tail.
  - Reverse application then leaves the HEAD most recent, so eviction removes whole positions from the TAIL first.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_swa_touch.py` which is a port of this upstream patch. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gates: `RADIANCE_TOUCH_ALL_GROUPS`, `RADIANCE_TOUCH_POSITION_ORDER`, `RADIANCE_SWA_STORE_MAMBA_ALIGN` (all default 0 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 7. patch_sched_align_last_block.py

### Purpose/Overview
Stops prompt's final prefill chunk at last full block boundary.

### Function-by-Function Breakdown

#### `patch_sched_align_last_block()`
- **Location**: `scheduler.py`
- **What it does**: Patches the scheduler to align the prompt's final prefill chunk at the last full block boundary.
- **Key logic**:
  - The prompt's final prefill chunk may not align with the block boundary.
  - The patch ensures that the final prefill chunk is aligned with the last full block boundary.
  - This improves the efficiency of the offload path.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch.

**Gap**: Our implementation is missing the scheduler block alignment optimization.

### Specific Improvements to Port

1. **Port `patch_sched_align_last_block()`**: Add the scheduler block alignment logic to our `scheduler.py` patch.
2. **Gate**: `RADIANCE_ALIGN_PROMPT_LAST_BLOCK` (default 1 when offload is on).
3. **Integration**: This patch improves the efficiency of the offload path.

### Integration Notes
- This patch improves the efficiency of the offload path.
- Validate with `turnbench.py` after integration.

---

## 8. patch_offload_lookup_metrics.py

### Purpose/Overview
Makes the lookup path observable by adding 13 new counters.

### Function-by-Function Breakdown

#### `patch_lookup_metrics()`
- **Location**: `scheduler.py`
- **What it does**: Adds Prometheus metrics for the lookup path.
- **Key metrics added**:
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

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_debug.py` which provides basic logging, but we do NOT have the detailed lookup metrics that upstream provides.

**Gap**: Our implementation lacks the detailed lookup metrics that upstream provides.

### Specific Improvements to Port

1. **Port `patch_lookup_metrics()`**: Add the lookup metrics to our `scheduler.py` patch.
2. **Gate**: `RADIANCE_OFFLOAD_LOOKUP_METRICS` (default 0 when offload is on).
3. **Integration**: This patch provides deeper visibility into the lookup path.

### Integration Notes
- This patch provides deeper visibility into the lookup path.
- Optional, not required for correctness.
- Validate with `turnbench.py` after integration.

---

## 9. patch_offload_debug_instrument.py

### Purpose/Overview
Records lookup/store events to a bounded JSON-line sink.

### Function-by-Function Breakdown

#### `patch_debug_instrument()`
- **Location**: `scheduler.py`
- **What it does**: Adds debug instrumentation to the lookup/store path.
- **Key logic**:
  - Records lookup/store events to a bounded JSON-line sink.
  - The sink is bounded to prevent memory exhaustion.
  - Events include timestamp, request ID, group index, block hash, and event type.

#### `_kvinstr.py`
- **Location**: `_kvinstr.py`
- **What it does**: Implements the bounded JSON-line sink.
- **Key logic**:
  - Uses a bounded queue to store events.
  - Events are written to a JSON-line file.
  - The queue is bounded to prevent memory exhaustion.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_debug.py` and `patch_offload_trace.py` which provide basic logging, but we do NOT have the bounded JSON-line sink that upstream provides.

**Gap**: Our implementation lacks the bounded JSON-line sink that upstream provides.

### Specific Improvements to Port

1. **Port `patch_debug_instrument()`**: Add the debug instrumentation to our `scheduler.py` patch.
2. **Port `_kvinstr.py`**: Add the bounded JSON-line sink to our implementation.
3. **Gate**: `RADIANCE_OFFLOAD_DEBUG_INSTRUMENT` (default 0 when offload is on).
4. **Integration**: This patch provides deeper visibility into the lookup/store path.

### Integration Notes
- This patch provides deeper visibility into the lookup/store path.
- Optional, not required for correctness.
- Validate with `turnbench.py` after integration.

---

## 10. patch_offload_fs_fanout.py

### Purpose/Overview
Fans fs-tier jobs across the thread pool. This is a port of vLLM PR #49225.

### Function-by-Function Breakdown

#### `patch_fs_fanout()`
- **Location**: `fs/manager.py`
- **What it does**: Patches the fs tier to fan jobs across the thread pool.
- **Key logic**:
  - The fs tier uses a thread pool for I/O operations.
  - The patch fans jobs across the thread pool to improve I/O throughput.
  - The fanout degree is determined by the byte budget and the number of threads.

#### `_radiance_fanout_degree()`
- **Location**: `fs/manager.py`
- **What it does**: Calculates the fanout degree for a job.
- **Key logic**:
  - The fanout degree is determined by the byte budget and the number of threads.
  - The budget is divided by the in-flight job count.
  - Extra batches buy queue entries and wake-ups without moving more bytes.

#### `_radiance_batches()`
- **Location**: `fs/manager.py`
- **What it does**: Splits a job into batches.
- **Key logic**:
  - The job is split into batches, with the largest remainder first.
  - Each batch is a partial job that can be processed by a thread in the pool.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_fs_tier.py` which includes the fs fanout logic. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gates: `RADIANCE_FS_FANOUT_TARGET_MB`, `RADIANCE_FS_FANOUT_MAX` (default 32/0 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 11. patch_offload_tier_report.py

### Purpose/Overview
Adds tier reporting instrumentation.

### Function-by-Function Breakdown

#### `patch_tier_report()`
- **Location**: `tiering/manager.py`
- **What it does**: Adds tier reporting instrumentation.
- **Key logic**:
  - Reports tier usage, hit rates, and latency.
  - The report is written to a JSON file.
  - The report includes CPU and fs tier metrics.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch.

**Gap**: Our implementation lacks the tier reporting instrumentation.

### Specific Improvements to Port

1. **Port `patch_tier_report()`**: Add the tier reporting instrumentation to our `tiering/manager.py` patch.
2. **Gate**: `RADIANCE_OFFLOAD_TIER_REPORT` (default 0 when offload is on).
3. **Integration**: This patch provides deeper visibility into the tier usage.

### Integration Notes
- This patch provides deeper visibility into the tier usage.
- Optional, not required for correctness.
- Validate with `turnbench.py` after integration.

---

## 12. patch_offload_promotion_wallclock.py

### Purpose/Overview
Attributes promotion refusals and wallclock reanchoring.

### Function-by-Function Breakdown

#### `patch_promotion_wallclock()`
- **Location**: `tiering/manager.py`
- **What it does**: Adds promotion refusal instrumentation and wallclock reanchoring.
- **Key logic**:
  - Attributes promotion refusals to specific causes.
  - Reanchors the wallclock to prevent drift.
  - The instrumentation includes promotion latency and refusal reasons.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch.

**Gap**: Our implementation lacks the promotion refusal instrumentation and wallclock reanchoring.

### Specific Improvements to Port

1. **Port `patch_promotion_wallclock()`**: Add the promotion refusal instrumentation and wallclock reanchoring to our `tiering/manager.py` patch.
2. **Gate**: `RADIANCE_OFFLOAD_PROMOTION_WALLCLOCK` (default 0 when offload is on).
3. **Integration**: This patch provides deeper visibility into the promotion path.

### Integration Notes
- This patch provides deeper visibility into the promotion path.
- Optional, not required for correctness.
- Validate with `turnbench.py` after integration.

---

## 13. patch_lookup_invalidate.py

### Purpose/Overview
Invalidates the fs tier async-lookup cache when a store lands.

### Function-by-Function Breakdown

#### `patch_lookup_invalidate()`
- **Location**: `async_lookup.py`
- **What it does**: Adds `invalidate()` and `forget()` methods to the async lookup manager.
- **Key logic**:
  - `invalidate()` drops the cached `absent` verdict when a store lands.
  - `forget()` drops the cached verdict for failed load keys.
  - This prevents stale `absent` verdicts from causing unnecessary recomputation.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_fs_tier.py` which includes the lookup invalidation logic. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gates: `RADIANCE_LOOKUP_INVALIDATE`, `RADIANCE_FS_FAILED_LOAD_FORGET` (default 1 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 14. patch_fs_failed_load.py

### Purpose/Overview
Forgets the fs tier cached lookup verdict for failed loads.

### Function-by-Function Breakdown

#### `patch_fs_failed_load()`
- **Location**: `fs/manager.py`
- **What it does**: Patches the fs tier to forget the cached lookup verdict for failed loads.
- **Key logic**:
  - A failed fs read fails the fs->CPU promotion.
  - The patch forgets the cached lookup verdict for the failed keys.
  - This prevents the request from re-promoting a missing file forever.

### Comparison to Our Implementation

**Our implementation**: We have `patch_offload_fs_tier.py` which includes the failed load forget logic. Our implementation is essentially identical to upstream.

**Gap**: None. Our implementation is a direct port of the upstream patch.

### Specific Improvements to Port

None. Our implementation is already a port of this upstream patch.

### Integration Notes
- Our implementation is already integrated.
- Gate: `RADIANCE_FS_FAILED_LOAD_FORGET` (default 1 when offload is on).
- Validate with `turnbench.py` after any changes.

---

## 15. patch_offload_miss_deferral_metrics.py

### Purpose/Overview
Instruments post-deferral outcomes.

### Function-by-Function Breakdown

#### `patch_miss_deferral_metrics()`
- **Location**: `scheduler.py`
- **What it does**: Adds metrics for post-deferral outcomes.
- **Key logic**:
  - Instruments the outcomes of deferred lookups.
  - Tracks whether deferred lookups eventually hit or miss.
  - The metrics include deferral latency and outcome distribution.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this patch.

**Gap**: Our implementation lacks the post-deferral outcome metrics.

### Specific Improvements to Port

1. **Port `patch_miss_deferral_metrics()`**: Add the post-deferral outcome metrics to our `scheduler.py` patch.
2. **Gate**: `RADIANCE_OFFLOAD_MISS_DEFERRAL_METRICS` (default 0 when offload is on).
3. **Integration**: This patch provides deeper visibility into the deferral path.

### Integration Notes
- This patch provides deeper visibility into the deferral path.
- Optional, not required for correctness.
- Validate with `turnbench.py` after integration.

---

## 16. kvwatch.py

### Purpose/Overview
Live KV-cache summary for watch. Reads stock vLLM metrics from `/metrics` endpoint.

### Function-by-Function Breakdown

#### `main()`
- **Location**: `kvwatch.py`
- **What it does**: Main entry point for the kvwatch tool.
- **Key logic**:
  - Reads stock vLLM metrics from the `/metrics` endpoint.
  - Displays live hit rates (GPU + offload tier), bytes loaded/stored, deferred lookups.
  - Updates every second.

#### `fetch_metrics()`
- **Location**: `kvwatch.py`
- **What it does**: Fetches metrics from the `/metrics` endpoint.
- **Key logic**:
  - Makes an HTTP GET request to the `/metrics` endpoint.
  - Parses the Prometheus metrics format.
  - Returns the parsed metrics.

#### `display_metrics()`
- **Location**: `kvwatch.py`
- **What it does**: Displays the metrics in a human-readable format.
- **Key logic**:
  - Formats the metrics for display.
  - Shows hit rates, bytes loaded/stored, deferred lookups.
  - Updates the display every second.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this tool.

**Gap**: Our implementation lacks the live KV-cache monitoring tool.

### Specific Improvements to Port

1. **Port `kvwatch.py`**: Add the live KV-cache monitoring tool to our `aijuus/kv-offload/ops/` directory.
2. **Integration**: This tool provides live monitoring of the offload path.

### Integration Notes
- This tool provides live monitoring of the offload path.
- Optional, not required for correctness.
- Deploy alongside the offload patches.

---

## 17. _kvinstr.py

### Purpose/Overview
Bounded JSON-line event sink.

### Function-by-Function Breakdown

#### `KVInstrSink`
- **Location**: `_kvinstr.py`
- **What it does**: Implements the bounded JSON-line event sink.
- **Key logic**:
  - Uses a bounded queue to store events.
  - Events are written to a JSON-line file.
  - The queue is bounded to prevent memory exhaustion.
  - Events include timestamp, request ID, group index, block hash, and event type.

#### `write_event()`
- **Location**: `_kvinstr.py`
- **What it does**: Writes an event to the sink.
- **Key logic**:
  - Adds the event to the bounded queue.
  - If the queue is full, the oldest event is dropped.
  - The event is written to the JSON-line file.

#### `flush()`
- **Location**: `_kvinstr.py`
- **What it does**: Flushes the sink to disk.
- **Key logic**:
  - Writes all pending events to the JSON-line file.
  - Clears the queue.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this tool.

**Gap**: Our implementation lacks the bounded JSON-line event sink.

### Specific Improvements to Port

1. **Port `_kvinstr.py`**: Add the bounded JSON-line event sink to our `aijuus/` directory.
2. **Integration**: This tool is used by `patch_offload_debug_instrument.py`.

### Integration Notes
- This tool is used by `patch_offload_debug_instrument.py`.
- Optional, not required for correctness.
- Deploy alongside the debug instrumentation patch.

---

## 18. kvcache-reap.sh

### Purpose/Overview
Eviction policy for the fs secondary tier. Deletes oldest blocks first.

### Function-by-Function Breakdown

#### `main()`
- **Location**: `kvcache-reap.sh`
- **What it does**: Main entry point for the reaper.
- **Key logic**:
  - Deletes oldest blocks first.
  - Uses mtime as insertion time = FIFO.
  - Respects MIN_AGE_MIN floor.
  - Has multiple stages: age, byte cap, capacity, emergency.

#### `pct()`
- **Location**: `kvcache-reap.sh`
- **What it does**: Gets the filesystem usage percentage.
- **Key logic**:
  - Uses `df --output=pcent` to get the usage percentage.
  - Returns the percentage as an integer.

#### `dir_bytes()`
- **Location**: `kvcache-reap.sh`
- **What it does**: Gets the bytes used by KV blocks.
- **Key logic**:
  - Uses `find` to list all `.bin` files.
  - Sums the file sizes.
  - Returns the total bytes.

#### `rm_one()`
- **Location**: `kvcache-reap.sh`
- **What it does**: Removes a single file.
- **Key logic**:
  - Removes the file if not in dry-run mode.
  - Increments the removed counter.

#### Stage A: age
- **Location**: `kvcache-reap.sh`
- **What it does**: Deletes blocks older than MAX_AGE_HOURS.
- **Key logic**:
  - Uses `find` to list files older than MAX_AGE_HOURS.
  - Deletes them oldest-first.
  - Respects MAX_DELETE cap.

#### Stage A2: hard byte cap
- **Location**: `kvcache-reap.sh`
- **What it does**: Deletes blocks to stay under the byte cap.
- **Key logic**:
  - Uses `find` to list files oldest-first.
  - Deletes them until the byte cap is met.
  - May cross the MIN_AGE_MIN floor when the cap is exceeded.

#### Stage B: capacity
- **Location**: `kvcache-reap.sh`
- **What it does**: Deletes blocks to stay under the filesystem usage target.
- **Key logic**:
  - Uses `find` to list files older than MIN_AGE_MIN.
  - Deletes them oldest-first.
  - Never crosses the MIN_AGE_MIN floor.

#### Stage C: emergency
- **Location**: `kvcache-reap.sh`
- **What it does**: Deletes blocks when the filesystem is nearly full.
- **Key logic**:
  - Uses `find` to list files older than EMERGENCY_MIN_AGE_MIN.
  - Deletes them oldest-first.
  - Crosses the MIN_AGE_MIN floor when necessary.

### Comparison to Our Implementation

**Our implementation**: We have `kvcache-reap.sh` which is more sophisticated than upstream. Our implementation has:
- Age + byte cap + capacity + emergency stages
- Hard-cap enforcement with exit 2
- More conservative MIN_AGE_MIN (90 vs 15)

**Gap**: Our implementation is more sophisticated than upstream. We do not need to port anything from upstream.

### Specific Improvements to Port

None. Our implementation is already more sophisticated than upstream.

### Integration Notes
- Our implementation is already integrated.
- Validate with `turnbench.py` after any changes.

---

## 19. turnbench.py

### Purpose/Overview
Multi-turn correctness gate, three-agent workload measurement. Compares cached vs cold token-for-token and logprob-for-logprob.

### Function-by-Function Breakdown

#### `main()`
- **Location**: `turnbench.py`
- **What it does**: Main entry point for the turnbench tool.
- **Key logic**:
  - Runs a multi-turn workload.
  - Compares cached vs cold token-for-token and logprob-for-logprob.
  - Reports bit-identical correctness.

#### `run_turn()`
- **Location**: `turnbench.py`
- **What it does**: Runs a single turn.
- **Key logic**:
  - Makes an HTTP POST request to the vLLM API.
  - Returns the response.

#### `compare_results()`
- **Location**: `turnbench.py`
- **What it does**: Compares cached vs cold results.
- **Key logic**:
  - Compares token-for-token and logprob-for-logprob.
  - Reports any mismatches.

#### `report()`
- **Location**: `turnbench.py`
- **What it does**: Reports the results.
- **Key logic**:
  - Prints the results in a human-readable format.
  - Reports bit-identical correctness.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this tool.

**Gap**: Our implementation lacks the multi-turn correctness gate.

### Specific Improvements to Port

1. **Port `turnbench.py`**: Add the multi-turn correctness gate to our `aijuus/kv-offload/ops/` directory.
2. **Integration**: This tool is the validation gate before deploying offload changes.

### Integration Notes
- This tool is the validation gate before deploying offload changes.
- Required for correctness validation.
- Deploy alongside the offload patches.

---

## 20. tierbench.py

### Purpose/Overview
Deterministic KV tier attribution bench.

### Function-by-Function Breakdown

#### `main()`
- **Location**: `tierbench.py`
- **What it does**: Main entry point for the tierbench tool.
- **Key logic**:
  - Runs a deterministic workload.
  - Attributes KV hits to specific tiers.
  - Reports tier attribution.

#### `run_workload()`
- **Location**: `tierbench.py`
- **What it does**: Runs the deterministic workload.
- **Key logic**:
  - Makes HTTP POST requests to the vLLM API.
  - Returns the responses.

#### `attribute_tiers()`
- **Location**: `tierbench.py`
- **What it does**: Attributes KV hits to specific tiers.
- **Key logic**:
  - Parses the response to determine which tier served the KV.
  - Reports the tier attribution.

#### `report()`
- **Location**: `tierbench.py`
- **What it does**: Reports the results.
- **Key logic**:
  - Prints the results in a human-readable format.
  - Reports tier attribution.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this tool.

**Gap**: Our implementation lacks the deterministic KV tier attribution bench.

### Specific Improvements to Port

1. **Port `tierbench.py`**: Add the deterministic KV tier attribution bench to our `aijuus/kv-offload/ops/` directory.
2. **Integration**: This tool provides deeper visibility into the tier usage.

### Integration Notes
- This tool provides deeper visibility into the tier usage.
- Optional, not required for correctness.
- Deploy alongside the offload patches.

---

## 21. equivbench.py

### Purpose/Overview
KV cache hit vs full recompute equivalence check.

### Function-by-Function Breakdown

#### `main()`
- **Location**: `equivbench.py`
- **What it does**: Main entry point for the equivbench tool.
- **Key logic**:
  - Runs a workload with KV cache hits.
  - Runs the same workload with full recompute.
  - Compares the results.

#### `run_with_cache()`
- **Location**: `equivbench.py`
- **What it does**: Runs the workload with KV cache hits.
- **Key logic**:
  - Makes HTTP POST requests to the vLLM API.
  - Returns the responses.

#### `run_without_cache()`
- **Location**: `equivbench.py`
- **What it does**: Runs the workload with full recompute.
- **Key logic**:
  - Makes HTTP POST requests to the vLLM API with prefix caching disabled.
  - Returns the responses.

#### `compare_results()`
- **Location**: `equivbench.py`
- **What it does**: Compares the results.
- **Key logic**:
  - Compares token-for-token and logprob-for-logprob.
  - Reports any mismatches.

#### `report()`
- **Location**: `equivbench.py`
- **What it does**: Reports the results.
- **Key logic**:
  - Prints the results in a human-readable format.
  - Reports equivalence.

### Comparison to Our Implementation

**Our implementation**: We do NOT have this tool.

**Gap**: Our implementation lacks the KV cache hit vs full recompute equivalence check.

### Specific Improvements to Port

1. **Port `equivbench.py`**: Add the KV cache hit vs full recompute equivalence check to our `aijuus/kv-offload/ops/` directory.
2. **Integration**: This tool provides deeper validation of the offload path.

### Integration Notes
- This tool provides deeper validation of the offload path.
- Optional, not required for correctness.
- Deploy alongside the offload patches.

---

## 22. serve-mxfp4.sh

### Purpose/Overview
KVCACHE block changes in the serve script.

### Function-by-Function Breakdown

#### KV offload section
- **Location**: `serve-mxfp4.sh`
- **What it does**: Configures the KV offload.
- **Key logic**:
  - Sets up the OffloadingConnector with CPU/RAM primary tier.
  - Optionally adds the fs secondary tier.
  - Configures head cap, eviction policy, and thread counts.
  - Sets PYTHONHASHSEED=0 for disk mode.

#### Stale RAM tier cleanup
- **Location**: `serve-mxfp4.sh`
- **What it does**: Cleans up stale RAM tier files on startup.
- **Key logic**:
  - Uses `fuser`/`lsof` to check if files are held by live processes before removing.
  - Removes `/dev/shm/vllm_offload_*.mmap` files that are not held by live processes.

### Comparison to Our Implementation

**Our implementation**: We have `serve-mxfp4.sh` which includes the KV offload section. Our implementation is more sophisticated than upstream with:
- Head cap (Mode A auto-fit CPU / Mode B no cap for disk)
- Configurable threads/policy
- `offload_prompt_only: true`
- Eagle fallback

**Gap**: Our implementation is missing the stale RAM tier cleanup that upstream has.

### Specific Improvements to Port

1. **Port stale RAM tier cleanup**: Add the stale RAM tier cleanup to our `serve-mxfp4.sh` in the preflight section.
2. **Integration**: This cleanup prevents stale RAM tier files from consuming memory.

### Integration Notes
- This cleanup prevents stale RAM tier files from consuming memory.
- Optional, not required for correctness.
- Deploy alongside the offload patches.

---

## Summary of Gaps and Port Recommendations

### Critical (FATAL) Patches to Port
1. `patch_offload_mixed_hit.py` — Required for bit-identical serving
2. `patch_offload_instrumentation.py` — Required for mixed-hit reporting

### Behavioral Correctness Patches to Port
3. `patch_reconcile_reask.py` — Improves performance for hybrid Mamba+attention requests
4. `patch_sched_align_last_block.py` — Improves efficiency of the offload path

### Operational Safety to Port
5. Stale RAM tier cleanup — Prevents stale RAM tier files from consuming memory

### Monitoring & Verification Tools to Port
6. `kvwatch.py` — Live KV-cache monitoring
7. `turnbench.py` — Multi-turn correctness gate
8. `tierbench.py` — Deterministic KV tier attribution bench
9. `equivbench.py` — KV cache hit vs full recompute equivalence check
10. `_kvinstr.py` — Bounded JSON-line event sink

### Metrics & Instrumentation Patches to Port (Optional)
11. `patch_offload_lookup_metrics.py` — Lookup metrics
12. `patch_offload_debug_instrument.py` — Debug instrumentation
13. `patch_offload_tier_report.py` — Tier reporting
14. `patch_offload_promotion_wallclock.py` — Promotion/wallclock metrics
15. `patch_offload_miss_deferral_metrics.py` — Miss deferral metrics

### Our Advantages (Keep)
- Head cap (`KV_OFFLOAD_HEAD_CAP`): Mode A auto-fit CPU / Mode B no cap for disk
- Configurable threads: `KV_OFFLOAD_READ_THREADS` (32) / `KV_OFFLOAD_WRITE_THREADS` (16)
- Configurable eviction policy: `KV_OFFLOAD_EVICTION_POLICY` (default lru)
- More sophisticated reaper: Byte cap stage (A2), hard-cap enforcement with exit 2, more conservative MIN_AGE_MIN (90 vs 15)
- Eagle fallback patch: `patch_offload_eagle_fallback.py` for hybrid correctness
