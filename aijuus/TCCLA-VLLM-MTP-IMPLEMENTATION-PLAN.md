# tcclaviger/vllm MTP Optimization Implementation Plan

> **SUPERSEDED (2026-09-29):** the drafter-quant axis in this document — Option A
> `DAVETHA_DRAFTER_QUANT` and Option B "reduced-vocab W4 draft head" (`draft_keep_file` +
> `Qwen3_5MTPW4`) — is **DROPPED**. tcclaviger removed that MTP path; our B2 A/B showed no clear
> win (see WORKLOG). What we actually ADOPTED from the R9700 branch is the **torch-level
> `RADIANCE_DRAFT_VOCAB` prune on our existing int2 head** (Phase 1.0, +5-6% greedy/sampled,
> lossless), plus the prebuilt TunableOp table (3.1) and the 0.29 correctness gates (0.5).
> Phase 1.1 and the B2 arm are removed; the W4 modules, `merge.py`, `keep-union.json` and
> `PLAN-A-FALLBACK.md` were deleted from the tree.

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
- **STATUS: FALLBACK PLAN — REMOVED 2026-09-29** (doc deleted; drafter-quant axis dropped)
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
**Status**: COMPLETE — `RADIANCE_DRAFT_VOCAB` + `RADIANCE_DRAFT_EXACTSET` + `RADIANCE_DRAFT_FUSED`
implemented in `radiance_drafthead.py` and GPU-validated. FUSED is byte-identical to the unfused
exact-set path and inert without the exact set; since the MTP entry keeps EXACTSET off (tau-gate,
finding #3), FUSED is inert here until the tau-gate confidence is decoupled. Wired into the MTP
registry entry (`RADIANCE_DRAFT_VOCAB=…/keep-union.txt`). Keep file built by
`aijuus/draft_keep/build_vocab.py` (union of the branch seed and our collector output).
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

### Phase 1.1: Reduced Vocab Draft Head (Option B) — REMOVED (superseded)

**Status**: REMOVED 2026-09-29. The W4 reduced head is dropped (tcclaviger removed that MTP path;
our B2 A/B showed no clear win — W4 ~86/~85 with acceptance ~3-4 vs int2 ~88/~80 with acceptance
~5-6). Deleted: `aijuus/qwen3_5_mtp_w4.py`, `aijuus/draft_w4_lmhead.py`, `aijuus/r4d_lib.py`,
`aijuus/draft_keep/merge.py`, `aijuus/draft_keep/keep-union.json`, `aijuus/TCCLA-VLLM-MTP-PLAN-A-FALLBACK.md`
and the `_install_draft_w4` wiring. `RADIANCE_DRAFT_VOCAB` on the int2 head (Phase 1.0) is kept.

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
**Status**: DONE (2026-09-29) — table copied to `/cache/tunableop/skinny0.csv` and loaded
(`reading tuning results from …`). Smoke bench: greedy flat, sampled 64.5 → 72.3 t/s (+12% on the
test prompt), acceptance unchanged. VERBOSE removed.

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
| B2 | ~~Head implementation~~ **REMOVED** | int2-vs-W4 dropped 2026-09-29 (superseded; int2 kept) | W4 path deleted from the tree |
| B3 | **MXFP4 + GPTQ drafter** | fp8 drafter vs MXFP4-RTN vs MXFP4-GPTQ (branch `RADIANCE_MTP_MXFP4[_FILE]`, `mtp_refit.py`/`mtp_gptq.py`) | Heavier; needs the branch's paro quant tooling and a calibration pass. |
| B4 | **AITER settings** | `VLLM_ROCM_USE_AITER=1` vs `0`, unified-attn on/off | Env either/or (Phase 2.2). |
| B5 | **flash_attn / capture sizes / DRY / degen** | as in Phases 3.2/3.3/4.x | Independent optional features. |
| B6 | **Heavy branch wins** | libr4d GDN exact-decay rx9x/rx9z; rot4 prefill producers; vision-tower + fp8-embed KV-pool offload | Separate, larger changes; not part of MTP Phase 1. |

**Integrate directly (do NOT put in the battery):** Phase 0.5 (correctness gates), Phase 1.0
(lossless head surface), Phase 2.1 no-calibration, Phase 3.1 TunableOp table — all proven
lossless/byte-exact and complementary, so gating them behind an A/B only wastes wall-clock.

### Battery results (log)
- **B1 SPEC depth**: SPEC 4 **worse** than 8 on our workload (greedy 86.5 vs 88.1, sampled 77.6 vs
  80.2 t/s; acceptance length ~3 vs ~5-6). Kept `spec_tokens=8`.
- **B2 W4 head**: **NOT adopted → REMOVED.** W4 (~85-87 greedy / ~83-87 sampled, acceptance ~3-4)
  showed no clear win over int2+vocab (~88 / ~80, acceptance ~5-6); lower acceptance. The W4 path
  was then deleted from the tree as superseded (tcclaviger dropped that MTP path). int2 kept.
- **Tuning — SPEC/captures**: denser capture ladder (3.3) adopted: greedy 92.4 / sampled 88.1 vs
  88.1 / 80.2.
- **Tuning — `RADIANCE_DRAFT_TAU`**: 0.28 worse than 0.20 (86.6 / 83.5 vs 92.4 / 88.1) → kept 0.20.
- **Tuning — `RADIANCE_SKINNY_GEMM=all`**: armed, pending.
- **DRY (quality, opt-in)**: `RADIANCE_DRY_MULTIPLIER` (unset/0 = off; e.g. `0.8` enables;
  optional `RADIANCE_DRY_BASE`/`_ALLOWED_LENGTH`/`_RANGE`). A/B on loop-prone prompts (repetition
  should drop) **and** normal prose/code (no over-penalization); decode t/s is ~neutral — DRY is a
  quality guard, not a speedup.
  **Baseline (DRY off, 2026-09-29)**: loop-phrase rep 0.523 / 217 tok / stop; loop-sentence rep 0.941
  / 176 tok (degen); loop-count rep 0.047 / 509 tok; normal-code rep 0.431; normal-prose rep 0.341;
  t/s greedy 87.6 / sampled 90.3. Probe: repeat inducing prompts + t/s (see WORKLOG).
  **ARM RESULT (DRY 0.8, range 2048) — NOT ADOPTED.** Quality is marginal (loop-phrase rep 0.523 →
  0.451, but loop-count 0.047 → 0.105 and normal-code 0.431 → 0.511; the forced loop is already
  caught by degen either way) and **sampled t/s regressed 90.3 → 76-83** (DRY adds per-step work on
  the rejection-sampler path). Revert: clear `RADIANCE_DRY_MULTIPLIER`/`RADIANCE_DRY_RANGE`.
- B3/B4: pending (B4 AITER neutral → reverted).

### Remaining work status (2026-09-29)
- **3.3 capture ladder**: **ADOPTED.** Clean re-measure: greedy 92.4 (was 88.1), sampled 88.1 (was
  80.2) → +4.9% / +9.9%.
- **B4 AITER on/off**: ARMED (`VLLM_ROCM_USE_AITER=0`); measure vs the 3.3 baseline.
- **3.2 flash_attn**: not installed. On gfx1201 the **CK backend cannot build** (Wave32 vs
  CK's Wave64); only the **Triton** backend works (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), and in
  vLLM it mainly accelerates **ViT** attention (1.8-6.5x TTFT for multimodal). Our body uses the
  purpose-built R4D attention → low expected value; optional attention-backend A/B.
- **B3 MXFP4+GPTQ drafter**: NOT pursued — needs the R9700 branch's paroquant plugin + a GPTQ
  calibration `.pt`; overlaps the W4 head we dropped.
- **4.2 degen**: **DONE (2026-09-29)** — `patch_degen.py` (runtime overlay, wired into the
  entrypoint; defaults `max_period 100 / min_repeats 6 / min_span 128`, `--degen-max-period 0`
  disables). Validated: fires on a period-1 run at ~129 tokens, period-4 at ~132, never on a
  non-periodic sequence; finishes with `finish_reason "repetition"`.
- **4.1 DRY**: **DONE (2026-09-29)** — `patch_dry.py` (runtime overlay, wired into the entrypoint
  after `patch_degen.py`) + the `dry.py` module copied by the entrypoint from
  `aijuus/refs/tcclaviger-vllm-29.05.12/vllm/v1/sample/ops/dry.py`. Tested in-container: all touched
  modules import, `SamplingMetadata.dry_params` field present, `SchedulerConfig`/`EngineArgs` dry
  fields (0.0/0.75/2/-1/None), `apply_dry` runs, `DryParams.from_any(None) is None`, `--dry-*` flags
  registered. **Default-off** (`dry_multiplier` 0.0 → no behaviour change until enabled). The port
  touches the sampler / rejection-sampler hot path plus input-processor / GPU-input-batch plumbing —
  refer to the change set below if a future vLLM bump needs it re-applied:
  1. **new** `vllm/v1/sample/ops/dry.py` (391 lines) — `DryParams` (with `from_any`/`enabled`),
     `apply_dry`, `build_breaker_seqs`, `ramp_to_base`, `DRY_DEFAULT_BREAKERS`.
  2. `vllm/v1/sample/penalties.py` — import `DryParams, apply_dry`; `apply_all_penalties(..., dry_params=None, apply_classic=True)`.
  3. `vllm/v1/sample/metadata.py` — import `DryParams`; field `dry_params: dict[int, DryParams] | None = None`.
  4. `vllm/v1/sample/sampler.py` — `apply_logits_processors` counts `bool(sampling_metadata.dry_params)`; `apply_penalties` reads `dry_params`/`apply_classic` and passes them.
  5. `vllm/v1/sample/rejection_sampler.py` — `has_dry`; force `repeat_indices`; build `dry_by_row` (req→row); `apply_penalties(..., dry_by_row)`.
  6. `vllm/sampling_params.py` — `enable_dry/dry_multiplier/dry_base/dry_allowed_length/dry_range/dry_sequence_breakers/_dry_params`.
  7. `vllm/v1/engine/input_processor.py` — `__init__` `_dry_breaker_cache`; `_dry_xarg`/`_as_bool`/`_resolve_dry_params` (xargs > SamplingParams > `--dry-*` defaults); call after `update_from_tokenizer`.
  8. `vllm/v1/worker/gpu_input_batch.py` — import `DryParams`; `self.dry_params` dict init/add/remove/swap/move; two `or bool(self.dry_params)` needs-token-ids gates; pass `dry_params=` in metadata.
  9. `vllm/config/scheduler.py` — `dry_multiplier` (0.0), `dry_penalty_ramp` (0.75), `dry_allowed_length` (2), `dry_range` (-1), `dry_sequence_breakers` (None).
  10. `vllm/engine/arg_utils.py` — `EngineArgs` dry fields, `--dry-*` CLI options, and the `SchedulerConfig(...)` kwargs.
  Optionally A/B later (e.g. `--dry-multiplier 0.8`).
- **5.1 tcclaviger repos**: DONE — public radiance repo cloned; the current "davetha" path is the
  DFlash2 drafter int4 projections under `RADIANCE_FAST_DRAFT`, not the MTP quant we removed.
- **5.2 ROCm 10**: RESEARCHED — see below; deferred as a major upgrade.

### ROCm 10 research (5.2, 2026-09-29) — staying on vLLM 0.29
- **ROCm 10.0.0** (Aug 2026): HIP **10.0.0**, LLVM **24.0.0**, rocBLAS **5.6.0**, hipBLASLt **1.4.1**,
  CK **1.2.0**; gfx1201/RDNA4 supported. Validated frameworks: PyTorch 2.11-2.13, **vLLM 0.27.0**
  (older than our 0.29), Python 3.14. The "3.3x inference / 2.4x training over ROCm 7" headline is
  from **ROCm.AI / Hyperloom** auto-optimization on **Instinct**, not a raw SDK/gfx1201 gain.
- **What actually changed for gfx1201**: refreshed gfx1201 SystemDB (tuned hipBLASLt find/perf
  entries), a hipBLASLt **local optimizer**, CK a8w8 GEMM tuning (MI355X), **HIP graph replay**
  reduces inter-kernel gaps for graphs interleaving async allocs, gfx12 Winograd, hipMemcpy2D /
  hipEventRecord wins. AITER PR #3330 added RDNA4 scalar fallbacks + ck-tile targets.
- **Known gfx1200/1201 caution**: default *batched* GEMM via hipBLASLt can regress vs non-batched
  (workaround `ROCBLAS_USE_HIPBLASLT=0`); a Tensile `MT64x64x64_ISA1201` OOB kernel bug exists on
  some ROCm 7.2 builds.
- **Why we defer (staying on 0.29)**: our radiance stack is hand-written gfx1201 kernels (libr4d,
  `radiance_mxfp4_fp8.so`) AOT-built with **ROCm 7.14 hipcc**, plus fork-local vLLM/AITER patches.
  A ROCm 10 move means rebuilding libr4d + the HIP module + vLLM and re-validating every patch;
  our TunableOp table is version-validated and would need regeneration. Net gfx1201-specific gain
  is unproven; effort/risk high. Revisit only if a vLLM 0.29-class stack is validated on ROCm 10.
- **wvSplitK (PR #34709)**: NOT a lever for us — it is **already in vLLM 0.29.0 and active**
  (`use_skinny` accepts `on_gfx1x()`; `VLLM_ROCM_USE_SKINNY_GEMM` default True; `wvSplitK_hf`
  wave32 kernels present in `_rocm_C.abi3.so`). Applies only to unquantized bf16 linears.

### Deep dive: tcclaviger MTP/kernel comparison (2026-09-29)
Compared the public `tcclaviger/vllm-radiance` repo and the `tcclaviger/vllm:29.05.12` image
against our tree:
- **Radiance repo**: ours is a **strict superset** — no `def`/`class` present in theirs that is
  missing from ours (`radiance_draft*.py`, `radiance_gdn/gemm/attn/kernels`).
- **vLLM spec_decode** (`eagle.py`, `medusa.py`, `ngram_proposer.py`): effectively identical
  (docstring/whitespace only).
- **MTP patches**: `patch_mtp_loopbreak.py`, `patch_mtp_mm_mask.py`, `patch_skinny_gemm.py`
  identical; `patch_radiance_fusion.py` (13) and `patch_gdn_metadata.py` (23) differ (our fork
  edits, not missing features).
- **wvSplitK**: already upstream/enabled (see above). **R4D attention**: already in use.
- **Shard-local draft confidence** (42% less per-step all-reduce): TP>1 only → N/A (we serve TP=1
  x2 instances).

**Two concrete missed tunings** (from their documented sweeps):
1. **`RADIANCE_DRAFT_TAU`**: tcclaviger bakes **0.35** for the bf16 head and **0.28 with
   `RADIANCE_FAST_DRAFT=1`** (+5.3% over 0.35). Ours was **0.20**. **RESULT: 0.28 is WORSE on our
   workload — greedy 86.6 / sampled 83.5 vs 0.20's 92.4 / 88.1 (−6.3% / −5.2%). Kept 0.20.** Our
   deeper 0.20 is the right value here (tcclaviger's +5.3% was vs *their* 0.35 baseline, not ours).
2. **`RADIANCE_SKINNY_GEMM=all`**: adds the ULP-level shapes (notably GDN `in_proj_ba`, 48x/step,
   28.5→3.6us) — **+3.5% tokens/s** in their DFlash2 case, no acceptance cost, but a bf16-ULP
   change can move acceptance under spec decode. **RESULT: REGRESSED on our MTP — greedy 92.6 (flat)
   but sampled 84.5 vs 88.1, acceptance down to 27-40% (mean accept ~3.2-4.2 vs ~5-6). Reverted to
   `1`.** The ULP-level shapes hurt MTP drafting acceptance here.

### Vocab A/B (draft keep-set)
One flip = `RADIANCE_DRAFT_VOCAB` in the MTP registry entry (+ redeploy). Arms:
- **seed-49k** (baseline, `keep-union.txt` 49,159): greedy 90.9 / sampled 85.2, accept 46.7%, mean accept 4.73.
- **full** (`RADIANCE_DRAFT_VOCAB=""` → 248,320-row int2 head, 0.33 GiB): **RESULT greedy 85.6 / sampled 85.0, accept 53.8%, mean accept 5.30.** The prune is confirmed worthwhile — full raises acceptance but the 8x coarse pass costs greedy t/s; sampled is a wash. Keep a pruned head.
- **freq variants** (ranked on the `betterbench/corpus/v1` workload, 21.8k corpus tokens / 3.2k unique → freq ranking padded by id):
  `keep-24k-freq.txt` (24,576), `keep-49k-freq.txt` (49,152), `keep-96k-freq.txt` (98,304),
  `keep-union-freq.txt` (65,327 = seed ∪ freq-49k).
Run one arm at a time; measure `aijuus/tools/mtp-bench.py` + acceptance.

### How to run an arm
One variable at a time. Each arm = change one env knob in `coolify-compose-2gpu.yml` (or the
Coolify UI env), redeploy, then measure with the same harness:

    # inside a vLLM container (or any host with the port reachable)
    python3 /patches/aijuus/tools/mtp-bench.py --url http://localhost:8000 --reps 3 --metrics

- Harness: `aijuus/tools/mtp-bench.py` (greedy + sampled, multi-prompt mean; dependency-free).
- Arm knobs:
  - **B1 SPEC depth**: `speculative_config.num_speculative_tokens` (8 -> 4). Also retime
    `RADIANCE_DRAFT_SCHEDULE` (it is keyed to SPEC=8).
  - **B2 head**: `DAVETHA_DRAFTER_QUANT=1` (int2+vocab vs W4 reduced head).
  - **B3 drafter quant**: `RADIANCE_MTP_MXFP4[_FILE]` (if/when ported).
  - **B4 AITER**: `VLLM_ROCM_USE_AITER=0` (and unified-attn off).
- Baseline (2026-09-29, int2+vocab, TunableOp, SPEC 8): greedy mean ~88, sampled mean ~80
  (2 reps x 400 tok; single-prompt numbers are noisy — use the mean).

## Implementation Order

1. **Phase 0**: Decision point — SUPERSEDED 2026-09-29 (drafter-quant axis dropped)
2. **Phase 0.5**: vLLM 0.29 correctness gates (V2-hook audit + `patch_aot_envkey`) — before any RADIANCE_* A/B
3. **Phase 5.1**: Fetch tcclaviger's vLLM fork (prerequisite for Phases 4.1/4.2)
4. **Phase 1.0**: Draft-head surface port (vocab prune + exactset + fused head) — cheapest gain, produces the keep list
5. **Phase 2.1**: Enable fp8 KV cache (no calibration)
6. **Phase 3.1**: Adopt the R9700 prebuilt TunableOp table
7. **Phase 3.2**: Evaluate flash_attn integration
8. **Phase 3.3**: Optimize CUDA graph capture sizes
9. **Phase 4.1**: Implement DRY repetition penalty
10. **Phase 4.2**: Implement degenerate-loop detection
11. **Phase 5.2**: Evaluate ROCm version upgrade (long-term)
12. **End A/B Battery** (B1, B3-B6) on the frozen integrated build

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
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml` — Current deployment config
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/050-fp8-mtp.patch` — MXFP4 body + FP8 drafter checkpoint conversion
- `/home/juup/radiance-vllm-mxfp4/aijuus/collect_tokens.py` — token-id collector feeding the `RADIANCE_DRAFT_VOCAB` keep set
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
