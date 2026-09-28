# tcclaviger/vllm MTP Optimization Research

## Executive Summary
- **We're ~2× better per GPU** than tcclaviger (79.9 t/s vs 40.8 t/s per GPU)
- **tcclaviger scales better with concurrency** (643.7 t/s @ 16 vs our 313.9 t/s @ 8)
- **Biggest missing optimization**: `DAVETHA_DRAFTER_QUANT` (int4 draft lm_head, 3.85x speedup) — NOT in our image
- **We already have**: dynamic MTP (confidence + n-gram), R4D attention, AR quantization, confidence-gated early exit
- **tcclaviger has**: fp8 KV cache, TunableOp GEMM sweep, flash_attn, clav_* extensions, DRY repetition penalty, degenerate-loop detection
- **Key difference**: tcclaviger uses ROCm 10.0, we use ROCm 7.14

## Objective
Research tcclaviger's vLLM Docker image and blog posts for MTP optimizations applicable to our Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp setup, and assess how they relate to the async-verify design investigation.

## tcclaviger/vllm Image Details
- **Base**: vLLM 0.27/0.29 tree, ROCm 10.0, torch 2.11, Python 3.14, gfx1201 (Radeon R9700)
- **Key library**: Ships libr4d (RDNA4 HIP kernel library) from codeberg.org/StillDeadcode/libr4d
- **Docker Hub**: https://hub.docker.com/r/tcclaviger/vllm

## tcclaviger Benchmark (Qwen3.8-Flash-Next-MXFP4-TP4-MTP4)
- Combined decode: 163.3 t/s
- ITL 1% low: 136.6 t/s
- TTFT p50: 58 ms
- Throughput @ 16 concurrent: 643.7 t/s
- Source: https://blog.robai.net/Qwen3.8-Flash-Next-MXFP4-TP4-MTP4-29.05.7/

## Our Setup
- **Model**: Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp
- **MTP**: SPEC=8
- **TP**: 1 (per GPU)
- **Architecture**: 2 GPUs data-parallel via model-router LB
- **Compose**: `coolify-compose-2gpu.yml`
- **Image**: `stilldeadcode/vllm-radiance:0.9.3`
- **Key settings**: MAX_NUM_SEQS=8, MAX_NUM_BATCHED_TOKENS=4096, KV_OFFLOAD_GIB=12, KV_OFFLOAD_DISK_DIR=/kvcache/blocks

## Our Benchmark Baselines
| Benchmark | Combined Decode | Update p99 | TTFT p50 | @ 8 concurrent | Prefill |
|-----------|----------------|------------|----------|----------------|---------|
| mtp8 (32k, single GPU) | 79.9 t/s | 62.1 ms | 123 ms | - | - |
| ocp-mtp8 (32k, 2 GPU) | 79.9 t/s | 62.1 ms | 123 ms | 163.1 t/s | 2,375 t/s @ 32k |
| mtp8-205k (205k, 2 GPU) | 80.4 t/s | 62.4 ms | 88 ms | 313.9 t/s | 2,368 t/s @ 64k |
| tc-dflash24 (ThinkingCap, 65k) | 115.5 t/s | 47.6 ms | 66 ms | 131.4 t/s | - |

## Per-GPU Comparison
- tcclaviger: 163.3 / 4 GPUs = 40.8 t/s per GPU
- Ours: 79.9 / 1 GPU = 79.9 t/s per GPU
- **We're ~2× better per GPU**

## Concurrency Scaling Comparison
- tcclaviger: 643.7 t/s @ 16 concurrent (16 GPUs total? or 4 GPUs with TP=4?)
- Ours: 313.9 t/s @ 8 concurrent (2 GPUs data-parallel)
- tcclaviger scales better with concurrency

## ASYNC-VERIFY-DESIGN.md Verdict
- Async verify not viable
- Verify N+1 depends on draft N (cross-step data dependency)
- V2/dflash path has no host loop
- V1/mtp path has host loop but overlap is unsound

## Docker Image Inspection (tcclaviger/vllm:latest)

### Image Metadata
- **Created**: 2026-08-17T09:02:45.677319+00:00
- **Base**: Ubuntu 26.04
- **Size**: 14.5 GB on disk, 4.23 GB content
- **Image ID**: ef99b3d07c3f

### Environment Variables
```
PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
VIRTUAL_ENV=/opt/vllm
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm
HIP_PLATFORM=amd
VLLM_TARGET_DEVICE=rocm
PYTORCH_ROCM_ARCH=gfx1201;gfx1200
HIP_ARCHITECTURES=gfx1201
AMDGPU_TARGETS=gfx1201
GPU_ARCHS=gfx1201
HSA_ENABLE_IPC_MODE_LEGACY=0
HSA_NO_SCRATCH_RECLAIM=1
HIP_FORCE_DEV_KERNARG=1
VLLM_ROCM_USE_AITER=0
PYTORCH_NVML_BASED_CUDA_CHECK=1
TVM_FFI_CACHE_DIR=/opt/tvm_ffi_cache
VLLM_LOGGING_COLOR=1
SAFETENSORS_FAST_GPU=1
TOKENIZERS_PARALLELISM=false
PYTHONDONTWRITEBYTECODE=1
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
CLAV_ATTN_AUTOTUNE=1
SP=/opt/vllm/lib/python3.14/site-packages
```

### Key Installed Packages
- `vllm 0.29.0.dev0+g2bdbbc8080` (dev version)
- `torch 2.11.0+rocm10.0`
- `triton 3.6.0`
- `flash_attn 2.8.3`
- `conch-triton-kernels 1.2.1`

### Custom clav_* Extensions (tcclaviger's fork-local libraries)
- `clav_ag_ext 0.0.1` — attention group extension
- `clav_ar_ext 0.0.1` — autoregressive extension (AR quantization)
- `clav_attn 0.1.0` — custom attention backend
- `clav_conv1d 0.1.0` — custom conv1d
- `clav_hc 0.1.0` — hidden state copy
- `clav_memcpy 0.1.0` — custom memcpy
- `clav_pleconv 0.1.0` — PLE conv (Flash-Next specific)
- `clav_reshape_cache 0.1.0` — cache reshaping
- `clav_silu_quant 0.1.0` — SiLU quantization
- `clav_state_copy 0.1.0` — state copy
- `gdn_hip 0.1.0` — GDN (Gated Delta Network) HIP kernel

### libr4d Location
- `/app/r4dhip/r4d.so` — prebuilt RDNA4 HIP kernel library

### r4d-related Files in vLLM
- `vllm/distributed/device_communicators/r4d_all_reduce.py` — distributed communication
- `vllm/v1/attention/backends/r4d_attn.py` — R4D attention backend
- `vllm/model_executor/layers/fused_moe/experts/r4d_mxfp4_moe.py` — MoE experts
- `vllm/model_executor/layers/fused_moe/experts/r4d_w4a16_moe.py` — MoE experts
- `vllm/model_executor/layers/fused_moe/router/r4d_route_router.py` — router
- `vllm/model_executor/layers/quantization/r4d_dsfp8_linear.py` — quantization
- `vllm/model_executor/kernels/linear/mxfp4/r4dhip.py` — linear kernels
- `vllm/model_executor/kernels/mhc/r4d_dsv4.py` — MHC kernels
- `vllm/model_executor/kernels/r4d_lib.py` — r4d library wrapper
- `gdn_hip/r4d_core.py` — GDN core

### MTP Speculator Implementation
- Located at: `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`
- Extends `AutoRegressiveSpeculator`
- Key feature: `share_mtp_topk_indices` — shares top-k indices between MTP steps to avoid recomputing them
- Uses `load_eagle_model` to load the draft model
- Has hooks for prefill and multi-step decode lifecycle

### Spec Decode Methods Available
- `mtp/speculator.py` — MTP speculator
- `dflash/speculator.py` — DFlash speculator
- `dflash2/speculator.py` — DFlash2 speculator
- `multi_module_mtp/speculator.py` — Multi-module MTP
- `autoregressive/speculator.py` — Base autoregressive speculator
- `dspark/speculator.py` — DSpark speculator
- `eagle/speculator.py` — Eagle speculator
- `gemma4/speculator.py` — Gemma4 speculator

### Fork-Local Features (tcclaviger's patches)
1. **`patches/mtp_confidence_exit`** — confidence-gated early exit
   - `draft_confidence_threshold` — stops drafting once product of draft-pick probabilities falls below threshold
   - Only supported with autoregressive drafters (method='mtp' or eagle family)
   - Runs one graph per draft step with host check after each

2. **`patches/mtp_hf_overrides`** — draft == target checkpoint
   - When the draft IS the target checkpoint, apply HF config overrides

3. **`patches/expert_offload`** — MoE expert offload with device-side expert cache
   - Uses libr4d's `moe_lru_*` kernels for device-side LRU eviction
   - Not applicable to our non-MoE model

4. **`patches/dry_sampler`** — DRY repetition penalty
   - Server-wide repetition penalty with per-request overrides

5. **`patches/degen_detect`** — degenerate-loop detection
   - Server-wide detection of degenerate output loops

6. **`patches/qwen4_exp`** — Qwen4 experimental features
   - PLE offload process transport
   - Blocked-weight default for quant_fp8

7. **`patches/think_state_sync`** — reasoning state sync
   - Reasoning closed in the prompt

8. **`patches/ep_no_comm`** — EP without communication
   - MoE expert parallelism without communication overhead

### DAVETHA_DRAFTER_QUANT Implementation
- Located at: `vllm/model_executor/kernels/draft_w4_lmhead.py`
- int4 group-128 lm_head for MTP DRAFT loop only, on gfx1201
- Two ways to enable:
  1. Checkpoint ships `mtp.lm_head.weight_q4` / `weight_scale` / `weight_zero` (packed per rank at load)
  2. `DAVETHA_DRAFTER_QUANT=1` — quantizes the shared bf16 head online at warmup
- Uses libr4d's `r4d_gemm_w4a16_nt_m64` kernel
- **Measured speedup** (R9700 gfx1201, N=124160 K=2560, cold, in HIP graph):
  - M=1: 3.85x speedup (1006.0 us -> 261.3 us)
  - M=2: 3.82x speedup
  - M=4: 3.82x speedup
  - M=5: 3.81x speedup
  - M=20: 3.45x speedup
- **Production result**: step 41.7 -> 38.9 ms, acceptance -3..-9% relative, net +4..+8% tok/s
- Quality: 9.99% relative Frobenius error at g128 with clip search
- Activation is f16 (not bf16) — kernel widens 4-bit code to f16

### R4D Attention Backend
- Located at: `vllm/v1/attention/backends/r4d_attn.py`
- `R4D_ATTN=1` is **DEFAULT OFF** in this image (unlike radiance where it's on)
- Hand-written HIP kernels built around transposed score matrix S^T = K.Q^T
- Wave32 WMMA fragment gives each lane exactly one query row, softmax is lane-private
- **Measured**: prefill attention 1.65x, +14.6% end-to-end prefill at 64K context
- Constraints:
  - head_dim 256, paged block size 16, 6 query heads per KV head
  - causal decoder attention, bf16 query, bf16 or fp8_e4m3 KV cache
  - no sliding window, no sinks, no soft cap, no alibi, no per-(token,head) scales
- Subclasses Triton backend — any batch R4D cannot take falls back to Triton kernel

### Dynamic Speculative Decoding
- Located at: `vllm/v1/spec_decode/dynamic/utils.py`
- `num_speculative_tokens_per_batch_size` — batch-size schedule for dynamic speculative token count
- Format: `[(range_start, range_end, num_speculative_tokens), ...]`
- Builds a dense lookup table: `batch_size -> K`
- Example: `[(1, 16, 3), (32, 128, 2)]` maps batch sizes 1-16 to K=3, 17-31 to K=3 (carried forward), 32-128 to K=2

### Entrypoint Tools
- `--tune` — RDNA4 quant GEMM config tuner
- `--quantize` — MXFP4-16 quantizer
- `--format rfa|rfi|mxfp4|composite` — unified quantizer CLI
- `--calib` — activation-aware quantization (ships in :dev tag only)
- `--kvcalibration` — FP8 KV-cache scale calibration
- `--test --ppl` — WikiText-2 perplexity test
- `--map-experts` — per-(layer,expert) activation-importance map
- `--flags` — print local flags
- `--recipes` — print measured serve recipes

### Env Toggles (from entrypoint)
- `VLLM_PLE_MLOCK=0` — do not mlock pinned host PLE table
- `VLLM_PLE_OFFLOAD_HOST_SLOTS=N` — host slots per PLE pipeline (4)
- `VLLM_DISABLE_RDNA4_FP8_KERNEL=1` — fall back to generic w8a8 Triton kernel
- `VLLM_GDN_SIDECACHE_DEBUG=1` — verbose GDN side-cache logging
- `VLLM_ATTN_AUTOTUNE_DUMP_DIR=DIR` — dump attention autotuner results
- `NO_AMD_GDN_HIP=1` — DISABLE native HIP GDN path (ON by default)
- `VLLM_GDN_USE_RECURRENT=1` — route GDN prefill to scalar recurrent kernel
- `CLAV_TUNABLEOP_SWEEP=1` — opt in to startup TunableOp GEMM sweep
- `CLAV_TUNABLEOP_CACHE_DIR` — cache directory for TunableOp results
- `CLAV_TUNABLEOP_WITH_OFFLOAD=1` — enable TunableOp sweep with expert offload

### Flash-Next Recipe (tp2-expert-mem-60gb-ple-cache-8gb)
```
--tensor-parallel-size 2
--max-model-len 262144
--max-num-seqs 16
--max-num-batched-tokens 4096
--kv-cache-dtype fp8
--gpu-memory-utilization 0.95
--enable-expert-offload
--expert-offload-mem 60
--ple-nvme-offload
--ple-nvme-dir /app/pleoffload
--ple-cache-gb 8
--ple-cache-reuse true
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
--compilation-config '{"cudagraph_capture_sizes": [4,8,12,16,20,24,28,32], "max_cudagraph_capture_size": 32, "inductor_compile_config": {"combo_kernels": false, "benchmark_combo_kernel": false}}'
```

### Our Image Inspection (stilldeadcode/vllm-radiance:0.9.3)

#### Image Metadata
- **Size**: 13.7 GB on disk, 3.96 GB content
- **Image ID**: 45694209177a
- **Base**: Ubuntu 24.04

#### Environment Variables
```
PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
VIRTUAL_ENV=/opt/vllm
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm
HIP_PLATFORM=amd
VLLM_TARGET_DEVICE=rocm
PYTORCH_ROCM_ARCH=gfx1201
RADIANCE_GFX_ARCH=gfx1201
HIP_ARCHITECTURES=gfx1201
AMDGPU_TARGETS=gfx1201
GPU_ARCHS=gfx1201
SAFETENSORS_FAST_GPU=1
TOKENIZERS_PARALLELISM=false
TRITON_CACHE_AUTOTUNING=1
PYTHONDONTWRITEBYTECODE=1
RADIANCE_USE_R4D=1
RADIANCE_USE_R4D_AR=1
RADIANCE_USE_R4D_AR_QUANT=1
RADIANCE_PRESHUFFLE=1
RADIANCE_FUSE_RMS_QUANT=1
RADIANCE_DYNAMIC_DRAFT=1
RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4
RADIANCE_DRAFT_TAU=0.35
RADIANCE_RUN_BWTEST=1
RADIANCE_VERSION=0.6.2
```

#### Key Installed Packages
- `vllm 0.27.1`
- `torch 2.11.0+rocm7.14`
- `triton 3.6.0+git7c56a5e4`
- `amd-aiter 0.1.17`
- `conch-triton-kernels 1.2.1`
- No `flash_attn` (tcclaviger has flash_attn 2.8.3)
- No `clav_*` extensions (tcclaviger has many)

#### Radiance Custom Modules
- `radiance_kernels.py` — kernel installation entry point
- `radiance_gdn.py` — GDN kernels
- `radiance_drafthead.py` — draft head
- `radiance_amdsmi.py` — AMD SMI integration
- `radiance_w4.py` — W4 quantization
- `radiance_vit_attn.py` — ViT attention
- `radiance_r4d_attn.py` — R4D attention
- `radiance_gemm.py` — GEMM kernels
- `radiance_draft.py` — dynamic MTP draft controller
- `radiance_allreduce.py` — allreduce
- `radiance_draft_gpu.py` — GPU kernels for draft controller

#### RADIANCE_DYNAMIC_DRAFT Implementation
- Located at: `/opt/vllm/lib/python3.12/site-packages/radiance_draft.py`
- **Per-request, per-slot controller** for MTP self-speculation
- Three actions per slot: keep drafting, take verbatim n-gram continuation, or stop and verify
- **Confidence-gated depth**: draft while running product of top-1 confidences stays >= TAU (default 0.28)
- **Batch-size schedule**: caps MTP forwards at given concurrency (default: 1:8,2:7,4:6,8:5,16:4)
- **Lossless**: only changes how many tokens are drafted, outputs cannot change
- **Local-vocabulary draft sampling**: avoids full-vocab all-gather across TP group
- **All hot-path work is on-device** (Triton confidence capture + n-gram matcher)
- **Env knobs**:
  - `RADIANCE_DYNAMIC_DRAFT` — 1=on (default), 0=off -> byte-identical stock MTP
  - `RADIANCE_DRAFT_SCHEDULE` — "bs:max_depth,..." batch-size MTP-forward ceiling
  - `RADIANCE_DRAFT_TAU` — confidence-product stop threshold (default 0.28)

### Key Differences from Our Setup
| Feature | tcclaviger | Ours |
|---------|------------|------|
| vLLM version | 0.29.0.dev0+g2bdbbc8080 | 0.27.1 |
| torch | 2.11.0+rocm10.0 | 2.11.0+rocm7.14 |
| ROCm | 10.0 | 7.14 |
| Python | 3.14 | 3.12 |
| Base OS | Ubuntu 26.04 | Ubuntu 24.04 |
| VLLM_ROCM_USE_AITER | 0 | 1 |
| AITER Unified Attention | N/A | 1 |
| R4D_ATTN | OFF (default) | ON (R4D_ATTN_FP8=3) |
| KV cache dtype | fp8 (with calibration) | auto/bf16 (no fp8) |
| MTP spec tokens | 3 (Flash-Next recipe) | 8 |
| CUDA graph capture | [4,8,12,16,20,24,28,32] | sized for SEQS*(SPEC+1)=136 |
| Expert offload | Yes (MoE) | N/A (non-MoE) |
| PLE NVMe offload | Yes (Flash-Next) | N/A |
| Drafter quant | DAVETHA_DRAFTER_QUANT=1 (int4) | Not present |
| Dynamic spec decode | Yes (batch-size schedule) | Yes (RADIANCE_DYNAMIC_DRAFT, confidence + n-gram) |
| Confidence early exit | Yes (draft_confidence_threshold) | Yes (RADIANCE_DRAFT_TAU) |
| DRY repetition penalty | Yes | Not present |
| Degenerate-loop detection | Yes | Not present |
| TunableOp GEMM sweep | Optional (CLAV_TUNABLEOP_SWEEP=1) | Not present |
| flash_attn | 2.8.3 | Not present |
| clav_* extensions | Yes (11 libraries) | No |

### RADIANCE_DYNMTP Status
- **NOT FOUND** in tcclaviger/vllm image
- Our equivalent is `RADIANCE_DYNAMIC_DRAFT` which is more sophisticated:
  - tcclaviger: batch-size schedule only (num_speculative_tokens_per_batch_size)
  - Ours: confidence-gated + n-gram matching + batch-size schedule
- Our dynamic draft is ON by default (RADIANCE_DYNAMIC_DRAFT=1)

## Pending Investigation
1. **DAVETHA_DRAFTER_QUANT** — found in tcclaviger image (int4 draft lm_head, 3.85x speedup), NOT in our image. Needs porting to our vLLM build.
2. **fp8 KV cache** — tcclaviger uses `--kv-cache-dtype fp8` with `--kvcalibration`. We don't use fp8 KV cache. Evaluate impact on our 2-GPU data-parallel setup.
3. **AITER settings** — tcclaviger has `VLLM_ROCM_USE_AITER=0`, we have `VLLM_ROCM_USE_AITER=1` with `UNIFIED_ATTENTION=1`. Compare performance impact.
4. **TunableOp GEMM sweep** — tcclaviger has optional `CLAV_TUNABLEOP_SWEEP=1`. Evaluate for performance improvement.
5. **flash_attn** — tcclaviger has flash_attn 2.8.3, we don't. Evaluate if it would help.
6. **DRY repetition penalty** — tcclaviger has server-wide DRY. Evaluate if useful for our setup.
7. **Degenerate-loop detection** — tcclaviger has server-wide detection. Evaluate if useful.
8. **Fetch tcclaviger's libr4d codeberg repo** (codeberg.org/StillDeadcode/libr4d) to inspect actual kernel implementation.
9. **Fetch tcclaviger's vLLM fork** to inspect the full patch set.
10. **ROCm version** — tcclaviger uses ROCm 10.0, we use ROCm 7.14. Evaluate if upgrading would help.

## Confirmed Findings
- Our `RADIANCE_DYNAMIC_DRAFT` is more sophisticated than tcclaviger's dynamic spec decode (confidence + n-gram vs batch-size only)
- We already have confidence-gated early exit (RADIANCE_DRAFT_TAU)
- We already have batch-size schedule (RADIANCE_DRAFT_SCHEDULE)
- We already have R4D attention (R4D_ATTN_FP8=3)
- We already have AR quantization (RADIANCE_USE_R4D_AR_QUANT=1)
- tcclaviger's `DAVETHA_DRAFTER_QUANT` (int4 draft lm_head) is NOT in our image — this is the biggest missing optimization
- tcclaviger's fp8 KV cache is NOT in our image
- tcclaviger's TunableOp GEMM sweep is NOT in our image

## Relevant Files
- `/home/juup/radiance-vllm-mxfp4/aijuus/ASYNC-VERIFY-DESIGN.md`
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml`
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/050-fp8-mtp.patch`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/ocp-mtp8.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8-205k.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/tc-dflash24.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-ft.betterbench.html`

## Next Steps
1. **Port DAVETHA_DRAFTER_QUANT** to our vLLM build — biggest potential speedup (3.85x on draft lm_head)
2. **Evaluate fp8 KV cache** — could allow more concurrent sequences or longer context without OOM
3. **Compare AITER settings** — test `VLLM_ROCM_USE_AITER=0` vs our current `=1` with `UNIFIED_ATTENTION=1`
4. **Evaluate TunableOp GEMM sweep** — `CLAV_TUNABLEOP_SWEEP=1` for GEMM performance improvement
5. **Fetch tcclaviger's libr4d repo** — inspect kernel implementation for potential improvements
6. **Fetch tcclaviger's vLLM fork** — inspect full patch set for additional optimizations
7. **Benchmark our current setup** with `RADIANCE_DYNAMIC_DRAFT` enabled vs disabled to measure its impact

## External References
- https://hub.docker.com/r/tcclaviger/vllm
- https://blog.robai.net/Qwen3.8-Flash-Next-MXFP4-TP4-MTP4-29.05.7/
- https://blog.robai.net/vllmdocs/
- https://codeberg.org/StillDeadcode/libr4d
