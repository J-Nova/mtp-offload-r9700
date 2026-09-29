# tcclaviger/vllm MTP Optimization Implementation Plan

## Executive Summary

Based on the research findings from tcclaviger's vLLM Docker image (versions 29.05.2 and 29.05.12) and blog posts, this plan outlines all beneficial optimizations to implement into our Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp setup.

**Key findings:**
- We're ~2× better per GPU than tcclaviger (79.9 t/s vs 40.8 t/s per GPU)
- tcclaviger scales better with concurrency (643.7 t/s @ 16 vs our 313.9 t/s @ 8)
- **MAJOR CHANGE in 29.05.12**: `DAVETHA_DRAFTER_QUANT` (int4 draft lm_head) was **REMOVED** on 2026-09-27
- **New approach in 29.05.12**: Reduced vocab draft head — cuts only observed/needed vocab rows from checkpoint's bf16 lm_head at load time using `draft_keep_file` (JSON of observed vocab IDs)
- **W4 draft head integration only in `qwen4_exp/amd/mtp.py`** — NOT in `qwen3_5_mtp.py` (our model uses Qwen3_5MTP/Qwen3NextMTP)
- **Developer comment**: ".12 onwards has a version of deadcodes int2 verifier/predictor adopted modifed and optimized then uses some of davethas pixie dust to reuse decisions that were validated. Sometimes is just much easier to reuse existing kernels in fp8hip."
- We already have: dynamic MTP (confidence + n-gram), R4D attention, AR quantization, confidence-gated early exit
- Our r4d.so already contains `r4d_gemm_w4a16_nt_m64` C entry point (confirmed via `nm -D`)
- Our current draft head: 2-bit quantized with Triton kernel (`radiance_drafthead.py`); tcclaviger: 4-bit with HIP kernel

## Current State

### Our Setup
- **Model**: Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp
- **MTP**: SPEC=8
- **TP**: 1 (per GPU)
- **Architecture**: 2 GPUs data-parallel via model-router LB
- **Image**: `stilldeadcode/vllm-radiance:0.9.3`
- **vLLM**: 0.27.1
- **torch**: 2.11.0+rocm7.14
- **Python**: 3.12
- **Base OS**: Ubuntu 24.04
- **Key settings**: MAX_NUM_SEQS=8, MAX_NUM_BATCHED_TOKENS=4096, KV_OFFLOAD_GIB=12, KV_OFFLOAD_DISK_DIR=/kvcache/blocks

### Our Benchmark Baselines
| Benchmark | Combined Decode | Update p99 | TTFT p50 | @ 8 concurrent | Prefill |
|-----------|----------------|------------|----------|----------------|---------|
| mtp8 (32k, single GPU) | 79.9 t/s | 62.1 ms | 123 ms | - | - |
| ocp-mtp8 (32k, 2 GPU) | 79.9 t/s | 62.1 ms | 123 ms | 163.1 t/s | 2,375 t/s @ 32k |
| mtp8-205k (205k, 2 GPU) | 80.4 t/s | 62.4 ms | 88 ms | 313.9 t/s | 2,368 t/s @ 64k |
| tc-dflash24 (ThinkingCap, 65k) | 115.5 t/s | 47.6 ms | 66 ms | 131.4 t/s | - |

### tcclaviger Benchmark (Qwen3.8-Flash-Next-MXFP4-TP4-MTP4)
- Combined decode: 163.3 t/s
- ITL 1% low: 136.6 t/s
- TTFT p50: 58 ms
- Throughput @ 16 concurrent: 643.7 t/s

## What We Already Have (No Action Needed)

| Feature | Our Implementation | tcclaviger Equivalent |
|---------|-------------------|----------------------|
| Dynamic MTP | `RADIANCE_DYNAMIC_DRAFT` (confidence + n-gram + batch-size schedule) | `num_speculative_tokens_per_batch_size` (batch-size only) |
| Confidence early exit | `RADIANCE_DRAFT_TAU=0.35` | `draft_confidence_threshold` |
| Batch-size schedule | `RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4` | Same concept |
| R4D attention | `RADIANCE_USE_R4D=1`, `R4D_ATTN_FP8=3` | `R4D_ATTN` (default OFF) |
| AR quantization | `RADIANCE_USE_R4D_AR_QUANT=1` | `clav_ar_ext` |
| GDN HIP kernel | `radiance_gdn.py` | `gdn_hip` |
| Preshuffle | `RADIANCE_PRESHUFFLE=1` | `clav_reshape_cache` |
| Fuse RMS quant | `RADIANCE_FUSE_RMS_QUANT=1` | `clav_silu_quant` |
| r4d.so W4A16 GEMM | `/opt/vllm/lib/python3.12/site-packages/r4d.so` has `r4d_gemm_w4a16_nt_m64` | Same library |

## Implementation Plan

### Phase 0: Decision Point — Old vs New Approach

**Priority**: CRITICAL (blocks all other work)
**Status**: DECIDED — Option B (reduced vocab approach) selected

**Context:**
- tcclaviger removed `DAVETHA_DRAFTER_QUANT` in 29.05.12 (2026-09-27)
- New approach: reduced vocab draft head using `draft_keep_file`
- We have patches 090/091 ready for the old DAVETHA_DRAFTER_QUANT approach (kept as fallback in Plan A)
- Decision made: proceed with Option B (reduced vocab approach)

**Option A: Old DAVETHA_DRAFTER_QUANT approach (patches 090/091)**
- **STATUS: FALLBACK PLAN** — documented in `TCCLA-VLLM-MTP-PLAN-A-FALLBACK.md`
- Only use if Option B fails
- int4 group-128 quantization of full vocab lm_head
- Online quantization at warmup (DAVETHA_DRAFTER_QUANT=1)
- Measured speedup: 3.85x on draft lm_head (R9700 gfx1201)
- Production result: step 41.7 -> 38.9 ms, acceptance -3..-9% relative, net +4..+8% tok/s
- Quality: 9.99% relative Frobenius error at g128 with clip search
- Pros: Patches already created and verified, simpler to implement
- Cons: Removed by tcclaviger, may not be optimal

**Option B: New reduced vocab approach (29.05.12)**
- **STATUS: ACTIVE** — implementation in progress
- Cuts only observed/needed vocab rows from checkpoint's bf16 lm_head at load time
- Uses `draft_keep_file` (JSON of observed vocab IDs)
- `CLAV_DRAFT_HEAD` (default "reduced"), `CLAV_DRAFT_HEAD_REPLICATE` (default "0")
- `get_top_tokens()` avoids full-vocab all-gather (only (value, id) pairs)
- Auto layout selection (replicated vs split)
- Pros: tcclaviger's current recommended approach, avoids full-vocab all-gather
- Cons: Requires generating `draft_keep_file`, need to port W4 integration from `qwen4_exp/amd/mtp.py` to our `Qwen3_5MTP`/`Qwen3NextMTP` classes, more complex

**Code Analysis Complete:**
- `qwen3_5_mtp.py` (329 lines) and `qwen3_next_mtp.py` (249 lines) in 29.05.12 have NO W4 draft head integration
- W4 draft head integration only in `qwen4_exp/amd/mtp.py` (970 lines)
- `MTPSpeculator` (59 lines) has `share_mtp_topk_indices` for QSA index sharing — only applicable to QSA models (Qwen4Exp), not our model
- `AutoRegressiveSpeculator` (1232 lines) has fork-local `patches/mtp_confidence_exit` (confidence-gated early exit) — we already have this via `RADIANCE_DRAFT_TAU`

**Implementation Steps for Option B:**
1. ~~Generate `draft_keep_file` by observing token usage during benchmark run~~ — IN PROGRESS (patches 092/093 created for token collection)
2. Port W4 draft head integration from `qwen4_exp/amd/mtp.py` to `Qwen3_5MTP`/`Qwen3NextMTP`
3. ~~Create new patches (092, 093) for the reduced vocab approach~~ — DONE (092: token collector hook, 093: Dockerfile COPY)
4. ~~Revert patches 090/091 (old approach)~~ — NOT NEEDED (kept as fallback in Plan A)
5. Build Docker image, deploy, and benchmark

### Phase 0.5: vLLM 0.29 correctness gates (from the R9700 branch)

**Priority**: HIGH — do BEFORE any RADIANCE_* A/B
**Status**: PENDING
**Source**: `aijuus/refs/r9700-tp1/REVIEW-LOG.md` (mtstanfield/vllm-mxfp4@r9700-tp1)

Two 0.29 findings that make our own measurements untrustworthy until fixed. Neither changes
behaviour on its own; both are integrations, not A/Bs:

1. **V2 runner / inert V1 hooks.** vLLM 0.29 runs the V2 model runner; fork-local hooks that
   target the V1 runner path are inert (review §191). Audit our runtime hooks
   (`radiance_kernels.install_all` and the KV-offload/eagle hooks) to confirm they fire on V2.
2. **Second compile cache ignores the environment.** `torch_compile_cache/<hash>/rank_R_D/{backbone,eagle_head}`
   is keyed on [env, config, traced code, compiler] and loaded by piece index, so an
   env-selected graph variant can load another variant's pieces and die in inductor
   (`copy_misaligned_inputs`). Port `aijuus/refs/r9700-tp1/patch_aot_envkey.py` to append the
   sorted RADIANCE_* env to the piecewise hash factors. Our deployment flips many RADIANCE_*
   toggles, so this is required before trusting any of them.

### Phase 1.0: Draft-head surface port (R9700 branch, lossless)

**Priority**: HIGH — cheapest gain, feeds and de-risks Phase 1.1
**Status**: PARTIAL — `RADIANCE_DRAFT_VOCAB` + `RADIANCE_DRAFT_EXACTSET` implemented in
`radiance_drafthead.py` and GPU-validated (outside-`-inf`; argmax matches the full head wherever
the winner is kept; EXACTSET leaves exactly RERANK eligible/row); `RADIANCE_DRAFT_FUSED` still
pending. Wired into the MTP registry entry (`RADIANCE_DRAFT_VOCAB=/patches/aijuus/draft_keep/keep-union.txt`,
`RADIANCE_DRAFT_EXACTSET=1`). Keep file built by `aijuus/draft_keep/build_vocab.py` = union of the
branch seed and our collector output (caveat (a)).
**Source**: `aijuus/refs/r9700-tp1/REVIEW-LOG.md` §2 + `aijuus/draft_keep/qwen38-draft-vocab-49152.txt`

Ports onto OUR existing `radiance_drafthead.py` (the int2 head we already run). Its hooks —
`_head_matrix`, `_head_is_empty`, `_apply_head_int2`, `_rerank_exact`, `_quantize_head_now`,
`RADIANCE_FAST_DRAFT`, `logits_processor` — match the branch's, so the patches are portable:

1. `RADIANCE_DRAFT_VOCAB=<file>` — score only the kept rows: `index_select` them into a
   sub-head, run the existing int2 coarse pass + exact rerank on the sub-matrix, fill the rest
   of the row `-inf`. Drafter-side only; the target verifies with its own head, so output
   cannot change (only acceptance/speed). Measured: greedy +6%, sampled +5%, byte-identical
   text (SPEC 4). Reuse the branch's 49,152-id list to seed the keep set.
2. `RADIANCE_DRAFT_EXACTSET=1` — only the exactly-reranked candidates are eligible for sampled
   drafts; lossless (+0.6% overall). Rejected there: `RADIANCE_DRAFT_TOPKP=1` (costs time,
   drops accepted tokens).
3. Fused draft head `RADIANCE_DRAFT_FUSED=1` — masks padding rows, sums x groups from the tile
   it already loads, rerank writes bf16 logits straight into the `-inf` row; 15 → 6 launches,
   byte-identical drafts/logits.

**Caveats (must hold):**
- (a) **The 49,152-id list is the branch's workload, not ours.** Use it to seed, but UNION it
  with our own collector output (`aijuus/collect_tokens.py` → `merge.py`) for the blend model —
  do not ship their list unmodified as our `draft_keep_file`.
- (b) **Env-gated off by default** (`RADIANCE_DRAFT_VOCAB` unset; `EXACTSET`/`FUSED` default 0)
  so it cannot perturb the current production path until we enable it deliberately.

**Output**: a validated keep list (the Phase 1.1 `draft_keep_file` input) plus a measured,
low-risk gain that does not depend on the W4 port.

### Phase 1.1: Reduced Vocab Draft Head (Option B) — consumes the Phase 1.0 keep list

**Priority**: CRITICAL
**Expected Impact**: +4..+8% tok/s net (similar to old approach, but more efficient)
**Status**: IN PROGRESS

**Details:**
- Reduced vocab draft head cuts only observed/needed vocab rows from checkpoint's bf16 lm_head at load time
- Uses `draft_keep_file` — JSON file containing observed vocab IDs
- `CLAV_DRAFT_HEAD` env var (default "reduced")
- `CLAV_DRAFT_HEAD_REPLICATE` env var (default "0") — controls replicated vs split layout
- `get_top_tokens()` avoids full-vocab all-gather (only (value, id) pairs)
- Auto layout selection based on device count
- W4 draft head integration is in `qwen4_exp/amd/mtp.py` — need to port to our `Qwen3_5MTP`/`Qwen3NextMTP` classes
- tcclaviger's `draft_keep_file` (`/app/tools/draft_vocab/keep_observed_qfn.json`): 52,380 tokens out of 124,160 (57.8% reduction) — but this is for Qwen4Exp (max ID 248,076), not our model

**Implementation Steps:**
1. **Generate `draft_keep_file`**: IN PROGRESS
   - Created `aijuus/collect_tokens.py` — token ID collector module
   - Token collector is wired at RUNTIME (no baked patch): `radiance_kernels._install_token_collector()` wraps the MTP head's `compute_logits` (default path) and `get_top_tokens`; the entrypoint overlays `aijuus/collect_tokens.py`. Patches 092/093 were removed 2026-09-28 (092 replaced by `092-radiance-kernels.patch`).
   - Next: Build image with patches, run benchmark with `RADIANCE_COLLECT_TOKENS=1`, collect token IDs
2. **Port W4 integration**: PENDING
   - Extract `qwen4_exp/amd/mtp.py` from tcclaviger 29.05.12 image (already extracted to `/tmp/kilo/qwen4_exp_mtp_29.05.12.py`)
   - Identify W4 draft head integration code (`_cut_reduced_head`, `_install_reduced_w4_head`, `get_top_tokens`, `compute_logits`)
   - Port to our `Qwen3_5MTP`/`Qwen3NextMTP` classes
   - Ensure compatibility with our model architecture
3. **Create patches**: PARTIALLY DONE
   - Patch 092: Token collector hook (DONE)
   - Patch 093: Dockerfile COPY for token collector (DONE)
   - Next: Patches for W4 draft head integration and `draft_keep_file` loading
4. **Build and test**: PENDING
   - Build Docker image with new patches
   - Deploy to test environment
   - Benchmark with and without reduced vocab
   - Compare acceptance rate and throughput

**Risk**: Medium — requires porting code from Qwen4Exp to Qwen3_5MTP, may need adjustments for our model architecture.

### Phase 2: High-Impact Configuration Changes

#### 2.1 Enable fp8 KV Cache with Calibration
**Priority**: HIGH
**Expected Impact**: More concurrent sequences or longer context without OOM

**Details:**
- tcclaviger uses `--kv-cache-dtype fp8` with `--kvcalibration`
- We currently use auto/bf16 KV cache
- fp8 KV cache halves KV memory usage, allowing more sequences or longer context
- Requires calibration step to compute per-head scales

**Implementation Steps:**
1. Add `--kv-cache-dtype fp8` to our compose file
2. ~~Implement or port the `--kvcalibration` tool~~ — **NOT NEEDED.** The R9700 branch measured fp8 KV as mantissa-bound: e4m3's 3-bit mantissa fixes the ~2.65% RMS per element, identically at scale 1.0 and at an amax-calibrated scale, so per-head calibration buys nothing (review §"fp8 KV cache -- mantissa-bound, and free"). KV amax 8-20 (K) / 4-77 (V), ~1% subnormal, none saturated.
3. Benchmark with fp8 KV cache vs bf16 to measure:
   - Memory usage reduction
   - Throughput change
   - Quality impact (perplexity comparison)
4. If quality impact is acceptable, enable by default

**Risk**: Low — the branch measured bf16-vs-fp8 KV PPL inside the noise (KL 0.003-0.006 over 32k-115k); bf16 KV would halve capacity for no measurable gain.

#### 2.2 Compare AITER Settings
**Priority**: HIGH
**Expected Impact**: Unknown — could improve or degrade performance
**Status**: MOVED to the End A/B Battery (it is an either/or env choice, not an integration)

**Details:**
- tcclaviger: `VLLM_ROCM_USE_AITER=0`
- Ours: `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`
- AITER provides optimized attention and GEMM kernels for ROCm
- tcclaviger's `VLLM_ROCM_USE_AITER=0` suggests they prefer their custom `clav_*` extensions over AITER

**Implementation Steps:**
1. Benchmark current setup with `VLLM_ROCM_USE_AITER=1` (baseline)
2. Benchmark with `VLLM_ROCM_USE_AITER=0` to compare
3. Benchmark with `VLLM_ROCM_USE_AITER=1` but `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0`
4. Analyze which configuration gives best throughput and latency
5. Update compose file with optimal settings

**Risk**: Low — just env var changes, easy to revert.

### Phase 3: Medium-Impact Changes

#### 3.1 Adopt the R9700 prebuilt TunableOp table for the skinny fp8 GEMMs
**Priority**: HIGH (replaces the earlier sweep-from-scratch plan)
**Expected Impact**: +5.1% sampled (measured on the branch); MTP GEMMs 1.30 → 0.85 ms, lm_head 2.47 → 2.32 ms, step 41.8 → 39.9 ms, acceptance identical
**Status**: MOVED UP — the branch already did this work and the versions match

**Details:**
- The branch's `fp8_tune.py` sweeps the six skinny fp8 shapes at every M the V2 runner uses (1..12) → a 78-entry per-shape table; hipBLASLt's heuristic gives the MTP drafter's N=5120 GEMMs a 16x128 tile (40 workgroups on 64 CUs, no split) and the lm_head 64x64; tuned solutions are 16x16.
- Load read-only: `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv`.
- Prebuilt table extracted to `aijuus/refs/r9700-tp1/tunableop-skinny0.csv`; its validators are **gfx1201 / torch 2.11.0 / HIP 714 / hipBLASLt 100401 — the same versions our image ships**, so it should load as-is. The table carries validators and is ignored on an image change, so a mismatch fails safe (defaults return).

**Implementation Steps:**
1. Mount `aijuus/refs/r9700-tp1/tunableop-skinny0.csv` into the container and set the three `PYTORCH_TUNABLEOP_*` env vars.
2. Confirm the table loads (not silently ignored) and the MTP GEMM/lm_head timings drop.
3. If any shape misses, run the branch's `fp8_tune.py` on our stack to regenerate.
4. Keep the sweep-from-scratch route only as a fallback.

**Risk**: Low — table is version-gated and read-only; worst case it is ignored.

#### 3.2 Evaluate flash_attn Integration
**Priority**: MEDIUM
**Expected Impact**: Unknown — could improve attention performance

**Details:**
- tcclaviger has `flash_attn 2.8.3` installed
- We don't have flash_attn
- tcclaviger has `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- We use R4D attention and AITER unified attention

**Implementation Steps:**
1. Install flash_attn 2.8.3 in our image
2. Benchmark with flash_attn enabled vs our current R4D/AITER attention
3. If flash_attn is faster, consider switching or using as fallback
4. If not faster, leave as optional

**Risk**: Low — optional feature.

#### 3.3 Optimize CUDA Graph Capture Sizes
**Priority**: MEDIUM
**Expected Impact**: Better GPU utilization for varying batch sizes

**Details:**
- tcclaviger uses `cudagraph_capture_sizes: [4,8,12,16,20,24,28,32]`
- Ours: sized for `SEQS*(SPEC+1)=136` (MAX_NUM_SEQS=8, SPEC=8)
- tcclaviger's approach captures multiple sizes for better batching flexibility

**Implementation Steps:**
1. Analyze our current CUDA graph capture configuration
2. Consider adopting tcclaviger's multi-size capture approach
3. Benchmark with different capture size configurations
4. Update compose file with optimal configuration

**Risk**: Low — just configuration change.

### Phase 4: Quality-of-Life / Safety Features

#### 4.1 Implement DRY Repetition Penalty
**Priority**: LOW
**Expected Impact**: Improved output quality (reduced repetition)

**Details:**
- tcclaviger has server-wide DRY repetition penalty with per-request overrides
- Located in `patches/dry_sampler`

**Implementation Steps:**
1. Fetch tcclaviger's vLLM fork to get the DRY sampler implementation
2. Port the DRY repetition penalty logic
3. Add server-wide default with per-request override support
4. Test with various prompts to measure quality improvement

**Risk**: Low — affects output quality, not performance.

#### 4.2 Implement Degenerate-Loop Detection
**Priority**: LOW
**Expected Impact**: Improved output quality (prevents degenerate loops)

**Details:**
- tcclaviger has server-wide detection of degenerate output loops
- Located in `patches/degen_detect`

**Implementation Steps:**
1. Fetch tcclaviger's vLLM fork to get the degenerate-loop detection implementation
2. Port the detection logic
3. Add server-wide detection with configurable thresholds
4. Test with various prompts to measure effectiveness

**Risk**: Low — affects output quality, not performance.

### Phase 5: Infrastructure / Long-Term

#### 5.1 Fetch and Inspect tcclaviger's Repos
**Priority**: HIGH (prerequisite for other phases)
**Expected Impact**: Enables all other implementations

**Details:**
- tcclaviger's libr4d: `codeberg.org/StillDeadcode/libr4d` (already cloned to `/tmp/kilo/libr4d-tcclaviger`)
- tcclaviger's vLLM fork: needs to be located (not publicly available)
- Developer mentions "deadcodes int2 verifier/predictor" and "davethas pixie dust" — need to understand these

**Implementation Steps:**
1. Analyze cloned libr4d repo for kernel implementation details
2. Locate and clone tcclaviger's vLLM fork (ask developer for access)
3. Analyze the full patch set for additional optimizations
4. Understand "deadcodes int2 verifier/predictor" and "davethas pixie dust"
5. Document findings for future reference

**Risk**: None — just research.

#### 5.2 Evaluate ROCm Version Upgrade
**Priority**: LOW (long-term)
**Expected Impact**: Potential performance improvements from newer ROCm

**Details:**
- tcclaviger uses ROCm 10.0
- We use ROCm 7.14
- ROCm 10.0 may have improved kernels and optimizations

**Implementation Steps:**
1. Research ROCm 10.0 changes and improvements
2. Evaluate compatibility with our model and setup
3. Plan upgrade path if beneficial
4. Benchmark before and after upgrade

**Risk**: High — major version change may introduce compatibility issues.

## End A/B Battery (run only AFTER everything above is implemented)

Nothing here is a "just take it" — these are either/or or workload-dependent choices, so they
are deliberately deferred to a single battery on a frozen, fully-integrated build, with one
variable at a time.

| # | A/B | Arms | Why deferred |
|---|-----|------|--------------|
| B1 | **SPEC depth** | SPEC 4 vs 8 (vs 5) | We run 8; the branch found SPEC 4 best under sampling (3: 80.8, 4: 84.5, 5: 83.7) but 8 best greedy. Workload-dependent. |
| B2 | **Head implementation** | Phase 1.0 pruned-int2 vs Phase 1.1 pruned-W4 | Same axis (vocab prune); decide which is faster/better once both exist. |
| B3 | **MXFP4 + GPTQ drafter** | fp8 drafter vs MXFP4-RTN vs MXFP4-GPTQ (branch `RADIANCE_MTP_MXFP4[_FILE]`, `mtp_refit.py`/`mtp_gptq.py`) | Heavier; needs the branch's paro quant tooling and a calibration pass. |
| B4 | **AITER settings** | `VLLM_ROCM_USE_AITER=1` vs `0`, unified-attn on/off | Env either/or (Phase 2.2). |
| B5 | **flash_attn / capture sizes / DRY / degen** | as in Phases 3.2/3.3/4.x | Independent optional features. |
| B6 | **Heavy branch wins** | libr4d GDN exact-decay rx9x/rx9z; rot4 prefill producers; vision-tower + fp8-embed KV-pool offload | Separate, larger changes; not part of MTP Phase 1. |

**Integrate directly (do NOT put in the battery):** Phase 0.5 (correctness gates), Phase 1.0
(lossless head surface), Phase 2.1 no-calibration, Phase 3.1 TunableOp table — all proven
lossless/byte-exact and complementary, so gating them behind an A/B only wastes wall-clock.

## Implementation Order

1. **Phase 0**: Decision point (old vs new approach) — done (Option B)
2. **Phase 0.5**: vLLM 0.29 correctness gates (V2-hook audit + `patch_aot_envkey`) — before any RADIANCE_* A/B
3. **Phase 5.1**: Fetch tcclaviger's vLLM fork (prerequisite for Phase 1.1)
4. **Phase 1.0**: Draft-head surface port (vocab prune + exactset + fused head) — cheapest gain, produces the keep list
5. **Phase 1.1**: Reduced vocab draft head (consumes the Phase 1.0 keep list)
6. **Phase 2.1**: Enable fp8 KV cache (no calibration)
7. **Phase 3.1**: Adopt the R9700 prebuilt TunableOp table
8. **Phase 3.2**: Evaluate flash_attn integration
9. **Phase 3.3**: Optimize CUDA graph capture sizes
10. **Phase 4.1**: Implement DRY repetition penalty
11. **Phase 4.2**: Implement degenerate-loop detection
12. **Phase 5.2**: Evaluate ROCm version upgrade (long-term)
13. **End A/B Battery** (B1-B6) on the frozen integrated build

## Success Metrics

| Metric | Current | Target (after all phases) |
|--------|---------|--------------------------|
| Combined decode (single GPU) | 79.9 t/s | 85+ t/s (+5% from reduced vocab) |
| Throughput @ 8 concurrent (2 GPU) | 313.9 t/s | 400+ t/s (from fp8 KV + AITER optimization) |
| TTFT p50 | 88-123 ms | <80 ms (from CUDA graph optimization) |
| Max concurrent sequences | 8 | 16+ (from fp8 KV cache) |
| Draft lm_head latency | ~1000 us (bf16) | ~260 us (reduced vocab W4) |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Reduced vocab quality degradation | Benchmark perplexity before/after; keep acceptance rate monitoring |
| `draft_keep_file` generation complexity | Use representative benchmark prompts; regenerate periodically |
| W4 integration porting from Qwen4Exp to Qwen3_5MTP | Test thoroughly; keep old 2-bit Triton draft head as fallback |
| fp8 KV cache quality degradation | Run calibration; compare perplexity; keep bf16 as fallback |
| AITER setting changes break compatibility | Benchmark each configuration; keep current settings as fallback |
| ROCm upgrade breaks compatibility | Thorough testing; keep ROCm 7.14 as fallback |
| CUDA graph changes increase memory usage | Monitor GPU memory usage; adjust capture sizes as needed |

## Open Questions

1. Where is tcclaviger's vLLM fork located? (Not publicly listed on Docker Hub)
2. What is the exact impact of reduced vocab draft head on our specific model (Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp)?
3. What is the quality impact of fp8 KV cache on our model?
4. Does AITER unified attention provide benefits that tcclaviger's custom extensions don't?
5. What is the performance impact of ROCm 10.0 vs 7.14 on our setup?
6. What is "deadcodes int2 verifier/predictor" mentioned by the developer?
7. What is "davethas pixie dust" mentioned by the developer?
8. How does the reduced vocab approach handle tokens not in `draft_keep_file`?
9. What is the optimal `CLAV_DRAFT_HEAD_REPLICATE` setting for our 2-GPU data-parallel setup?

## Relevant Files

- `/home/juup/radiance-vllm-mxfp4/aijuus/WORKLOG.md` — running dated work log (decisions, changes, verification)
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_keep/qwen38-draft-vocab-49152.txt` — R9700 branch 49,152-id draft vocab (seed only; UNION with our collector output)
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/REVIEW-LOG.md` — R9700 branch engineering review (rounds 1-7)
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/tunableop-skinny0.csv` — prebuilt TunableOp table (Phase 3.1)
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/patch_aot_envkey.py` — 0.29 second-compile-cache env-key fix (Phase 0.5)
- `/home/juup/radiance-vllm-mxfp4/aijuus/model-registry.json` — per-model serving knobs; MTP entry now carries the interim `kv_cache_memory` pin
- `/home/juup/radiance-vllm-mxfp4/aijuus/TCCLA-VLLM-MTP-RESEARCH.md` — Research findings
- `/home/juup/radiance-vllm-mxfp4/aijuus/TCCLA-VLLM-MTP-PLAN-A-FALLBACK.md` — Plan A fallback documentation (old DAVETHA_DRAFTER_QUANT approach)
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml` — Current deployment config
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/050-fp8-mtp.patch` — MXFP4 body + FP8 drafter checkpoint conversion
- `/home/juup/radiance-vllm-mxfp4/aijuus/r4d_lib.py` — ctypes loader for r4d.so exposing W4A16 GEMM functions
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_w4_lmhead.py` — int4 group-128 draft lm_head implementation (old 29.05.2 approach)
- `/home/juup/radiance-vllm-mxfp4/aijuus/collect_tokens.py` — Token ID collector for generating draft_keep_file (Plan B)
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/092-radiance-kernels.patch` — radiance_kernels.py runtime hooks (`_install_token_collector`, `_install_draft_w4`); replaces 090/091/old-092/094
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/072-kv-cache-029.patch` — kv-cache/*.py 0.29 port (previously uncovered)
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_keep/merge.py` — merges `rank*.json` collector output into `keep.json`
- `/home/juup/radiance-vllm-mxfp4/aijuus/kv-offload/ops/entrypoint.sh` — runtime overlay (radiance_*.py + aijuus/ + configs from /patches)
- `/home/juup/radiance-vllm-mxfp4/radiance_kernels.py` — vLLM plugin hook entry point (092-radiance-kernels.patch; overlaid from /patches at runtime)
- `/home/juup/radiance-vllm-mxfp4/radiance_drafthead.py` — existing 2-bit Triton draft head (baseline for comparison)
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/062-radiance-drafthead.patch` — our current 2-bit Triton draft head implementation
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/060-radiance-draft.patch` — dynamic draft scheduling patch
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/ocp-mtp8.betterbench.html` — Our MTP8 32k 2-GPU benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8-205k.betterbench.html` — Our MTP8 205k benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8.betterbench.html` — Our MTP8 single-GPU 32k benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/tc-dflash24.betterbench.html` — ThinkingCap dflash24 benchmark
- `/tmp/kilo/libr4d-tcclaviger/` — cloned tcclaviger libr4d repo (contains `r4d_gemm_w4a16_nt_m64.hip`)
- `/tmp/kilo/draft_w4_lmhead.py` — extracted tcclaviger int4 draft lm_head implementation (source reference, old 29.05.2)
- `/tmp/kilo/r4d_lib.py` — extracted tcclaviger ctypes loader for r4d.so (source reference)
- `/tmp/kilo/qwen4_exp_mtp.py` — extracted tcclaviger Qwen4Exp MTP model integration (reference, old 29.05.2)
- `/tmp/kilo/mtp_speculator.py` — extracted tcclaviger MTP speculator (reference)
- `/tmp/kilo/mtp_speculator_29.05.12.py` — extracted tcclaviger 29.05.12 MTP speculator (59 lines, `share_mtp_topk_indices` feature)
- `/tmp/kilo/speculator_base_29.05.12.py` — extracted tcclaviger 29.05.12 base speculator (611 lines)
- `/tmp/kilo/autoregressive_speculator_29.05.12.py` — extracted tcclaviger 29.05.12 autoregressive speculator (1232 lines, fork-local `mtp_confidence_exit`)
- `/tmp/kilo/draft_w4_lmhead_29.05.12.py` — extracted tcclaviger 29.05.12 draft_w4_lmhead.py (398 lines, new reduced-vocab approach)
- `/tmp/kilo/qwen4_exp_mtp_29.05.12.py` — extracted tcclaviger 29.05.12 Qwen4Exp MTP model integration (970 lines, contains W4 draft head integration, `set_skip_topk`, `compact_topk_indices`)
- `/tmp/kilo/qwen3_5_mtp_29.05.12.py` — extracted tcclaviger 29.05.12 Qwen3_5 MTP model (329 lines, NO W4 integration)
- `/tmp/kilo/qwen3_next_mtp_29.05.12.py` — extracted tcclaviger 29.05.12 Qwen3Next MTP model (249 lines, NO W4 integration)
- `/tmp/kilo/keep_observed_qfn.json` — extracted tcclaviger 29.05.12 draft_keep_file (52,380 tokens, for Qwen4Exp)
- `/tmp/kilo/gdn_hip_29.05.12/` — extracted tcclaviger 29.05.12 gdn_hip package (native HIP GDN ops including `gdn_verify_r` MTP spec-verify kernel)
- `/tmp/kilo/clav_attn_29.05.12/` — extracted tcclaviger 29.05.12 clav_attn package (native unified attention kernel)
- `/tmp/kilo/clav_ar_29.05.12.py` — extracted tcclaviger 29.05.12 clav_ar (P2P-BAR all-reduce, 857 lines)
- `/tmp/kilo/clav_ag_29.05.12.py` — extracted tcclaviger 29.05.12 clav_ag (P2P-BAR all-gather, 448 lines)
- `/tmp/kilo/app_src_29.05.12/` — extracted tcclaviger 29.05.12 /app/src (mischip/q4hc, plehip HIP source files)
- `tcclaviger/vllm:29.05.12` (Docker image) — contains new reduced-vocab approach in `qwen4_exp/amd/mtp.py`, `draft_w4_lmhead.py` (updated), new clav_* packages, HIP kernel packages in /app/
- `/opt/vllm/lib/python3.12/site-packages/r4d.so` — our r4d library (has `r4d_gemm_w4a16_nt_m64` entry point)

## External References

- https://hub.docker.com/r/tcclaviger/vllm
- https://blog.robai.net/Qwen3.8-Flash-Next-MXFP4-TP4-MTP4-29.05.7/
- https://blog.robai.net/vllmdocs/
- https://codeberg.org/StillDeadcode/libr4d
