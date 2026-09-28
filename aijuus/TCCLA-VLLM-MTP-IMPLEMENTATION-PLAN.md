# tcclaviger/vllm MTP Optimization Implementation Plan

## Executive Summary

Based on the research findings from tcclaviger's vLLM Docker image and blog posts, this plan outlines all beneficial optimizations to implement into our Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp setup.

**Key findings:**
- We're ~2× better per GPU than tcclaviger (79.9 t/s vs 40.8 t/s per GPU)
- tcclaviger scales better with concurrency (643.7 t/s @ 16 vs our 313.9 t/s @ 8)
- Biggest missing optimization: `DAVETHA_DRAFTER_QUANT` (int4 draft lm_head, 3.85x speedup)
- We already have: dynamic MTP (confidence + n-gram), R4D attention, AR quantization, confidence-gated early exit

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

## Implementation Plan

### Phase 1: High-Impact, Low-Risk Changes

#### 1.1 Port DAVETHA_DRAFTER_QUANT (int4 draft lm_head)
**Priority**: CRITICAL
**Expected Impact**: +4..+8% tok/s net (3.85x speedup on draft lm_head)

**Details:**
- Located in tcclaviger image at: `vllm/model_executor/kernels/draft_w4_lmhead.py`
- int4 group-128 lm_head for MTP DRAFT loop only, on gfx1201
- Uses libr4d's `r4d_gemm_w4a16_nt_m64` kernel
- Two ways to enable:
  1. Checkpoint ships `mtp.lm_head.weight_q4` / `weight_scale` / `weight_zero` (packed per rank at load)
  2. `DAVETHA_DRAFTER_QUANT=1` — quantizes the shared bf16 head online at warmup
- Measured speedup (R9700 gfx1201, N=124160 K=2560, cold, in HIP graph):
  - M=1: 3.85x (1006.0 us -> 261.3 us)
  - M=2: 3.82x
  - M=4: 3.82x
  - M=5: 3.81x
  - M=20: 3.45x
- Production result: step 41.7 -> 38.9 ms, acceptance -3..-9% relative, net +4..+8% tok/s
- Quality: 9.99% relative Frobenius error at g128 with clip search
- Activation is f16 (not bf16) — kernel widens 4-bit code to f16

**Implementation Steps:**
1. Fetch tcclaviger's vLLM fork to get the full `draft_w4_lmhead.py` implementation
2. Port the int4 quantization logic to our radiance draft head (`radiance_drafthead.py`)
3. Add `DAVETHA_DRAFTER_QUANT` env var toggle (default 0 for safety)
4. Implement online quantization at warmup (option 2) as the initial approach
5. Benchmark with and without to measure actual impact on our model
6. If successful, consider pre-quantizing the checkpoint (option 1) for faster startup

**Risk**: Low — only affects draft loop, not target model. Acceptance may drop slightly (-3..-9%) but net tok/s increases.

#### 1.2 Enable fp8 KV Cache with Calibration
**Priority**: HIGH
**Expected Impact**: More concurrent sequences or longer context without OOM

**Details:**
- tcclaviger uses `--kv-cache-dtype fp8` with `--kvcalibration`
- We currently use auto/bf16 KV cache
- fp8 KV cache halves KV memory usage, allowing more sequences or longer context
- Requires calibration step to compute per-head scales

**Implementation Steps:**
1. Add `--kv-cache-dtype fp8` to our compose file
2. Implement or port the `--kvcalibration` tool to compute per-head scales
3. Run calibration on representative prompts
4. Store calibration scales in the checkpoint or a separate file
5. Benchmark with fp8 KV cache vs bf16 to measure:
   - Memory usage reduction
   - Throughput change
   - Quality impact (perplexity comparison)
6. If quality impact is acceptable, enable by default

**Risk**: Medium — fp8 KV cache may introduce quality degradation. Calibration is required to minimize this.

#### 1.3 Compare AITER Settings
**Priority**: HIGH
**Expected Impact**: Unknown — could improve or degrade performance

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

### Phase 2: Medium-Impact Changes

#### 2.1 Implement TunableOp GEMM Sweep
**Priority**: MEDIUM
**Expected Impact**: GEMM performance improvement (unknown magnitude)

**Details:**
- tcclaviger has optional `CLAV_TUNABLEOP_SWEEP=1` for startup GEMM tuning
- Sweeps GEMM configurations at startup to find optimal parameters
- `CLAV_TUNABLEOP_CACHE_DIR` for caching results
- `CLAV_TUNABLEOP_WITH_OFFLOAD=1` for sweep with expert offload

**Implementation Steps:**
1. Fetch tcclaviger's vLLM fork to get the TunableOp sweep implementation
2. Port the sweep logic to our radiance GEMM (`radiance_gemm.py`)
3. Add `RADIANCE_TUNABLEOP_SWEEP` env var toggle (default 0)
4. Implement result caching to avoid re-sweeping on every startup
5. Benchmark with and without sweep to measure impact

**Risk**: Low — optional feature, default off.

#### 2.2 Evaluate flash_attn Integration
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

#### 2.3 Optimize CUDA Graph Capture Sizes
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

### Phase 3: Quality-of-Life / Safety Features

#### 3.1 Implement DRY Repetition Penalty
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

#### 3.2 Implement Degenerate-Loop Detection
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

### Phase 4: Infrastructure / Long-Term

#### 4.1 Fetch and Inspect tcclaviger's Repos
**Priority**: HIGH (prerequisite for other phases)
**Expected Impact**: Enables all other implementations

**Details:**
- tcclaviger's libr4d: `codeberg.org/StillDeadcode/libr4d`
- tcclaviger's vLLM fork: needs to be located

**Implementation Steps:**
1. Clone `codeberg.org/StillDeadcode/libr4d` to inspect kernel implementation
2. Locate and clone tcclaviger's vLLM fork
3. Analyze the full patch set for additional optimizations
4. Document findings for future reference

**Risk**: None — just research.

#### 4.2 Evaluate ROCm Version Upgrade
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

## Implementation Order

1. **Phase 4.1**: Fetch tcclaviger's repos (prerequisite)
2. **Phase 1.1**: Port DAVETHA_DRAFTER_QUANT (highest impact)
3. **Phase 1.2**: Enable fp8 KV cache with calibration
4. **Phase 1.3**: Compare AITER settings
5. **Phase 2.1**: Implement TunableOp GEMM sweep
6. **Phase 2.2**: Evaluate flash_attn integration
7. **Phase 2.3**: Optimize CUDA graph capture sizes
8. **Phase 3.1**: Implement DRY repetition penalty
9. **Phase 3.2**: Implement degenerate-loop detection
10. **Phase 4.2**: Evaluate ROCm version upgrade (long-term)

## Success Metrics

| Metric | Current | Target (after all phases) |
|--------|---------|--------------------------|
| Combined decode (single GPU) | 79.9 t/s | 85+ t/s (+5% from DAVETHA) |
| Throughput @ 8 concurrent (2 GPU) | 313.9 t/s | 400+ t/s (from fp8 KV + AITER optimization) |
| TTFT p50 | 88-123 ms | <80 ms (from CUDA graph optimization) |
| Max concurrent sequences | 8 | 16+ (from fp8 KV cache) |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| DAVETHA_DRAFTER_QUANT quality degradation | Benchmark perplexity before/after; keep acceptance rate monitoring |
| fp8 KV cache quality degradation | Run calibration; compare perplexity; keep bf16 as fallback |
| AITER setting changes break compatibility | Benchmark each configuration; keep current settings as fallback |
| ROCm upgrade breaks compatibility | Thorough testing; keep ROCm 7.14 as fallback |
| CUDA graph changes increase memory usage | Monitor GPU memory usage; adjust capture sizes as needed |

## Open Questions

1. Where is tcclaviger's vLLM fork located? (Not publicly listed on Docker Hub)
2. What is the exact impact of DAVETHA_DRAFTER_QUANT on our specific model (Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp)?
3. What is the quality impact of fp8 KV cache on our model?
4. Does AITER unified attention provide benefits that tcclaviger's custom extensions don't?
5. What is the performance impact of ROCm 10.0 vs 7.14 on our setup?

## Relevant Files

- `/home/juup/radiance-vllm-mxfp4/aijuus/TCCLA-VLLM-MTP-RESEARCH.md` — Research findings
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml` — Current deployment config
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/050-fp8-mtp.patch` — MXFP4 body + FP8 drafter checkpoint conversion
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/ocp-mtp8.betterbench.html` — Our MTP8 32k 2-GPU benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8-205k.betterbench.html` — Our MTP8 205k benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8.betterbench.html` — Our MTP8 single-GPU 32k benchmark
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/tc-dflash24.betterbench.html` — ThinkingCap dflash24 benchmark

## External References

- https://hub.docker.com/r/tcclaviger/vllm
- https://blog.robai.net/Qwen3.8-Flash-Next-MXFP4-TP4-MTP4-29.05.7/
- https://blog.robai.net/vllmdocs/
- https://codeberg.org/StillDeadcode/libr4d
