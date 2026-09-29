# tcclaviger/vllm MTP Optimization Research

## Executive Summary
- **We're ~2× better per GPU** than tcclaviger (79.9 t/s vs 40.8 t/s per GPU)
- **tcclaviger scales better with concurrency** (643.7 t/s @ 16 vs our 313.9 t/s @ 8)
- **MAJOR CHANGE in 29.05.12**: `DAVETHA_DRAFTER_QUANT` (int4 draft lm_head) was **REMOVED** on 2026-09-27
- **New approach in 29.05.12**: Reduced vocab draft head — cuts only observed/needed vocab rows from checkpoint's bf16 lm_head at load time using `draft_keep_file` (JSON of observed vocab IDs)
- **New env vars in 29.05.12**: `CLAV_DRAFT_HEAD` (default "reduced"), `CLAV_DRAFT_HEAD_REPLICATE` (default "0")
- **W4 draft head integration only in `qwen4_exp/amd/mtp.py`** — NOT in `qwen3_5_mtp.py` (our model uses Qwen3_5MTP/Qwen3NextMTP)
- **Developer comment**: ".12 onwards has a version of deadcodes int2 verifier/predictor adopted modifed and optimized then uses some of davethas pixie dust to reuse decisions that were validated. Sometimes is just much easier to reuse existing kernels in fp8hip."
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

## Docker Image Inspection (tcclaviger/vllm:29.05.12) — LATEST

### Image Metadata
- **Created**: 2026-09-28 (pushed to Docker Hub)
- **Base**: Ubuntu 26.04
- **Size**: 4.2 GB
- **vLLM**: 0.29.0.dev0
- **Python**: 3.14
- **torch**: 2.11.0+rocm10.0

### Environment Variables (29.05.12)
```
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
CLAV_DRAFT_HEAD=reduced
CLAV_DRAFT_HEAD_REPLICATE=0
SP=/opt/vllm/lib/python3.14/site-packages
```

### Key Changes from 29.05.2 (latest) to 29.05.12
1. **DAVETHA_DRAFTER_QUANT REMOVED** (removed 2026-09-27)
   - The full-vocab warmup-time int4 pack approach is gone
   - `draft_w4_lmhead.py` still exists but the env var is no longer referenced
   - Docstring in `draft_w4_lmhead.py` confirms removal

2. **New: Reduced vocab draft head**
   - `CLAV_DRAFT_HEAD` env var (default "reduced")
   - Cuts only observed/needed vocab rows from checkpoint's bf16 lm_head at load time
   - Uses `draft_keep_file` — JSON file containing observed vocab IDs
   - No full-vocab all-gather in `get_top_tokens()` — only (value, id) pairs
   - Auto layout selection: replicated vs split based on device count
   - Automatic at load time (no env var needed beyond `CLAV_DRAFT_HEAD`)

3. **New: `CLAV_DRAFT_HEAD_REPLICATE`** (default "0")
   - Controls whether reduced head is replicated across devices or split

4. **W4 draft head integration location**
   - ONLY in `qwen4_exp/amd/mtp.py` — NOT in `qwen3_5_mtp.py`
   - Our model uses `Qwen3_5MTP`/`Qwen3NextMTP` classes, NOT `Qwen4Exp`
   - `qwen3_5_mtp.py` in 29.05.12 uses standard `ParallelLMHead` + `LogitsProcessor` (no W4 integration)

5. **New packages in 29.05.12**
   - `clav_ag`, `clav_ar`, `clav_attn`, `clav_conv1d`, `clav_hc`, `clav_memcpy`, `clav_pleconv`, `clav_reshape_cache`, `clav_silu_quant`, `clav_state_copy`, `gdn_hip`

### Fork HIP Kernel Packages Baked In (Developer Info)

| Package | Source | What it does |
|---------|--------|--------------|
| gdnhip | gdnhip/csrc/gdn_prefill_r.hip | Qwen GDN linear-attention: fused prefill, decode, and MTP spec-verify. Replaces fla-Triton. |
| hipattn (clav_attn) | hipattn/csrc/unified_clav_attn_16.hip, qsa_scorer.hip, qsa_sparse_attn.hip | Native full attention plus QSA sparse attention and scorer. Replaces TRITON_ATTN. Built with CLAV_P_FP16=1. |
| clav_allreduce | clav_allreduce/csrc/clav_ar.hip, clav_ar_hip.hip | P2P-BAR all-reduce for TP groups. Runs ahead of PyNCCl. |
| clav_allgather | clav_allgather/csrc/clav_ag.hip, clav_ag_hip.hip | P2P-BAR all-gather. Replaces RCCL for sizes it wins on. |
| rfhip | rfhip/csrc/rfi_gemm.hip, rfi8_gemm.hip, rfa_gemm.hip, rfi_fused.hip, rfi_rotate.hip | RFI/RFA quantized GEMM and rotation kernels. Built with -DRFI_DIRECT_A=1 -DRFI_SPLITK=8. |
| plehip | plehip/csrc/ple_direct.hip, ple_stream_ops.hip, ple_nvme.cpp, ple_rocr.cpp | Qwen4Exp PLE n-gram row gather from NVMe over PCIe. |
| mischip (7 packages) | mischip/*/csrc/*.hip | batch_memcopy, fused_state_copy, causal_conv1d (fwd + update, plus RTC), reshape_and_cache_flash (plus RTC), silu_mul_fp8, q4hc (HyperConnection fused), pleconv. |
| fp8hip | fp8hip/csrc/fp8hip_capi.hip includes fp8hip_gemm.hip and fp8hip_moe.hip | Block-scaled w8a8 FP8 GEMM, dense and MoE. Plain ctypes .so at /app/fp8hip/libfp8hip_gemm.so. |
| parohip | parohip/csrc/radiance_paroquant.hip + par_kernels.h | ParoQuant int4 x fp8 W4A8 kernels. pybind .so at /app/parohip/. |
| r4dhip (libr4d) | /mnt/dest/unification/libr4d/*.hip (external context, NOT in repo) | Big kernel family: paged attention h256, GDN chunk scan, all-reduce wires, bf16 skinny GEMM, mxfp4a8/dsfp4a8/dsfp8a8/w4a16/w4a8 dense + MoE GEMMs, QSA index prep, mHC for DSV4, dflash conv, PLE dequant, MoE LRU. Ships as /app/r4dhip/r4d.so. |

### Extracted HIP Kernel Package Analysis (29.05.12)

**gdn_hip** (extracted to `/tmp/kilo/gdn_hip_29.05.12/`):
- Native HIP GDN ops for gfx1201 (RDNA4)
- Four ops (since 2026-08-08 deprecation): `gdn_prefill_r`, `gdn_prefill_r2`, `gdn_decode_r`, `gdn_verify_r`
- **`gdn_verify_r`** is the fused MTP spec-verify kernel — directly relevant to MTP optimization
- Also has `gdn_prefill_state_r2` (FORK-LOCAL CED)
- Head dims other than 128 fall through to fla-Triton
- Loads AOT-compiled `gdn_hip_C.cpython-314-x86_64-linux-gnu.so`

**clav_attn** (extracted to `/tmp/kilo/clav_attn_29.05.12/`):
- Native unified attention kernel replacing TRITON_ATTN
- Single op: `unified_attn_16` with tuning parameters (block_m, tile, nwarps, segments, partition, qslice)
- Supports bf16 and fp8 KV cache (k_descale, v_descale, q_descale parameters)
- Sweep ranges: UNIFIED_TILE=(16,32,64), UNIFIED_NWARPS=(1,2,4), UNIFIED_SEGMENTS=(1,16,32,64,128,256)
- FA1 vs FA2-v2: FA1 nwarps splits KV columns of one 16-row m-tile (max 4 waves/SIMD); FA2-v2 nwarps IS the row count (BLOCK_M = nwarps*16), 8 is default
- Loads AOT-compiled `clav_attn_C.cpython-314-x86_64-linux-gnu.so`

**clav_ar** (extracted to `/tmp/kilo/clav_ar_29.05.12.py`, 857 lines):
- One-shot P2P-BAR custom all-reduce for RDNA4 TP groups, TP = 2/4/8
- Derived from radiance TP=2 design (`reference/radiance_allreduce.py`), generalized over rank count
- PUSH topology: every rank pushes input into slot[my_rank] of every peer's scratch over PCIe BAR
- Reduce order FIXED (rank 0..ws-1) for deterministic output
- Double-buffered by seq parity for cudagraph compatibility
- Env: CLAV_AR, CLAV_AR_MAX_KB, CLAV_AR_NT, CLAV_AR_WORDS_PER_BLOCK, CLAV_AR_MIN_NB, CLAV_AR_MAX_NB
- fp8 payload quant NOT carried over yet (additive, belongs behind its own gate)

**clav_ag** (extracted to `/tmp/kilo/clav_ag_29.05.12.py`, 448 lines):
- One-shot P2P-BAR custom all-gather for RDNA4 TP groups, TP = 2/4/8
- Sibling of clav_ar, same PUSH topology and handshake
- Exact wire by default, Q8 opt-in (CLAV_AG_Q8=1) for large gathers (0.625x fabric bytes)
- Q8: layered wire (L1 top-1 exact / L2 RMS bound / L3 raw escape), bf16/fp16 only
- Env: CLAV_AG, CLAV_AG_MAX_KB, CLAV_AG_NT, CLAV_AG_Q8, CLAV_AG_QUANT_MIN_KB, CLAV_AG_Q8_BOUND

**mischip/q4hc** (extracted to `/tmp/kilo/app_src_29.05.12/mischip/q4hc/csrc/`):
- HyperConnection fused kernel for Qwen4Exp (NOT relevant to our Qwen3.8-3.6-27B model)
- `hc_fused.hip` (616 lines): fused GatedResidual.{mix, combine_and_mix}
- `hc_fused_wmma.hip`: WMMA variant
- Shapes: residual [N, D], block_output [N, H], injection [N, HC], N <= 16, HC <= 4, D <= 10240

**plehip** (extracted to `/tmp/kilo/app_src_29.05.12/plehip/csrc/`):
- PLE n-gram row gather from NVMe over PCIe for Qwen4Exp (NOT relevant to our model)
- `ple_direct.hip`, `ple_stream_ops.hip`, `ple_nvme.cpp`, `ple_rocr.cpp`

**Prebuilt .so files in /app/**:
- `/app/fp8hip/libfp8hip_gemm.so` (196 KB) — block-scaled w8a8 FP8 GEMM
- `/app/parohip/radiance_paroquant_kernel.so` (5.1 MB) — ParoQuant int4 x fp8 W4A8 kernels
- `/app/r4dhip/r4d.so` (10.9 MB) — big kernel family including w4a16 GEMM

### Developer Comment on .12+ Changes
> ".12 onwards has a version of deadcodes int2 verifier/predictor adopted modifed and optimized then uses some of davethas pixie dust to reuse decisions that were validated. Sometimes is just much easier to reuse existing kernels in fp8hip."

**Interpretation:**
- Deadcode's int2 verifier/predictor was adopted, modified, and optimized
- Uses some of davetha's "pixie dust" to reuse validated decisions
- Sometimes reuses existing kernels in fp8hip instead of creating new ones
- This suggests the new approach is a hybrid: deadcode's int2 + davetha's decision reuse + fp8hip kernel reuse

### draft_w4_lmhead.py (29.05.12) — Updated Implementation
- Still contains int4 group-128 quantization logic (`pack_w4a16`, `gemm_w4a16`, `PackedW4`)
- But the `DAVETHA_DRAFTER_QUANT` env var is no longer used
- Now uses `CLAV_DRAFT_HEAD` and `draft_keep_file` for reduced vocab
- `get_top_tokens()` avoids full-vocab all-gather (only (value, id) pairs)
- Auto layout selection (replicated vs split) based on device count

### MTP Model Files in 29.05.12
- `qwen4_exp/amd/mtp.py` — contains W4 draft head integration (reduced vocab)
- `qwen3_5_mtp.py` — standard ParallelLMHead + LogitsProcessor (NO W4 integration)
- Our model uses Qwen3_5MTP/Qwen3NextMTP classes → need to port W4 integration from qwen4_exp/amd/mtp.py

### Key Implications for Our Setup
- We're TP=1 per GPU (data-parallel), so all-gather bottleneck doesn't apply to us
- Reduced vocab is still beneficial: smaller DRAM traffic for lm_head forward
- Need to generate `draft_keep_file` for our model (observe token usage during benchmark run)
- Need to port W4 draft head integration from `qwen4_exp/amd/mtp.py` to our `Qwen3_5MTP`/`Qwen3NextMTP` classes

## Docker Image Inspection (tcclaviger/vllm:latest) — HISTORICAL (29.05.2)

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
| Feature | tcclaviger 29.05.12 | tcclaviger 29.05.2 | Ours |
|---------|---------------------|-------------------|------|
| vLLM version | 0.29.0.dev0 | 0.29.0.dev0+g2bdbbc8080 | 0.27.1 |
| torch | 2.11.0+rocm10.0 | 2.11.0+rocm10.0 | 2.11.0+rocm7.14 |
| ROCm | 10.0 | 10.0 | 7.14 |
| Python | 3.14 | 3.14 | 3.12 |
| Base OS | Ubuntu 26.04 | Ubuntu 26.04 | Ubuntu 24.04 |
| VLLM_ROCM_USE_AITER | 0 | 0 | 1 |
| AITER Unified Attention | N/A | N/A | 1 |
| R4D_ATTN | OFF (default) | OFF (default) | ON (R4D_ATTN_FP8=3) |
| KV cache dtype | fp8 (with calibration) | fp8 (with calibration) | auto/bf16 (no fp8) |
| MTP spec tokens | 3 (Flash-Next recipe) | 3 (Flash-Next recipe) | 8 |
| CUDA graph capture | [4,8,12,16,20,24,28,32] | [4,8,12,16,20,24,28,32] | sized for SEQS*(SPEC+1)=136 |
| Expert offload | Yes (MoE) | Yes (MoE) | N/A (non-MoE) |
| PLE NVMe offload | Yes (Flash-Next) | Yes (Flash-Next) | N/A |
| Drafter quant | Reduced vocab (CLAV_DRAFT_HEAD=reduced) | DAVETHA_DRAFTER_QUANT=1 (int4) | Not present |
| draft_keep_file | Yes (JSON of observed vocab IDs) | No | No |
| get_top_tokens | No full-vocab all-gather (only (value, id) pairs) | Full-vocab all-gather | Full-vocab all-gather |
| Auto layout selection | Yes (replicated vs split) | No | N/A (TP=1) |
| W4 draft head integration | qwen4_exp/amd/mtp.py only | draft_w4_lmhead.py | Not present |
| Dynamic spec decode | Yes (batch-size schedule) | Yes (batch-size schedule) | Yes (RADIANCE_DYNAMIC_DRAFT, confidence + n-gram) |
| Confidence early exit | Yes (draft_confidence_threshold) | Yes (draft_confidence_threshold) | Yes (RADIANCE_DRAFT_TAU) |
| DRY repetition penalty | Yes | Yes | Not present |
| Degenerate-loop detection | Yes | Yes | Not present |
| TunableOp GEMM sweep | Optional (CLAV_TUNABLEOP_SWEEP=1) | Optional (CLAV_TUNABLEOP_SWEEP=1) | Not present |
| flash_attn | 2.8.3 | 2.8.3 | Not present |
| clav_* extensions | Yes (11 libraries) | Yes (11 libraries) | No |

### RADIANCE_DYNMTP Status
- **NOT FOUND** in tcclaviger/vllm image
- Our equivalent is `RADIANCE_DYNAMIC_DRAFT` which is more sophisticated:
  - tcclaviger: batch-size schedule only (num_speculative_tokens_per_batch_size)
  - Ours: confidence-gated + n-gram matching + batch-size schedule
- Our dynamic draft is ON by default (RADIANCE_DYNAMIC_DRAFT=1)

## mtstanfield/vllm-mxfp4 `r9700-tp1` branch — MTP findings (2026-09-29)

Source: `github.com/GGZ14/vllm-mxfp4` compare `main...mtstanfield:vllm-mxfp4:r9700-tp1`
(4 commits ahead, 8 behind, 219 files). Base `1e82407` + two weeks of single-R9700
(gfx1201, 32 GB) TP=1 tuning. Engineering log: `r9700/docs/vllm-radiance-review-20260927.md`
(823 lines, rounds 1-7). Branch cloned to `/tmp/kilo/mts-r9700`.

**Model parity**: their "Qwen3.8-27B PARO-MXFP4 v2" and our
`Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ` are both `qwen3_5`, `vocab_size 248320`,
`hidden 5120`. Their draft-vocab artifact is directly compatible with our tokenizer.

### A. Draft-vocab prune — simplest, immediately usable (our Phase 1 alternative)
- Mechanism `RADIANCE_DRAFT_VOCAB=<file, one token id per line>` in `radiance_drafthead.py`
  (see `r9700/review-20260927/patch_draftvocab.py`): `index_select` the kept rows into a
  sub-head, run the *existing* int2 coarse pass + exact rerank on the sub-matrix, fill the
  rest of the row `-inf`. Drafter-side only — the target verifies with its own head, so
  output cannot change; only acceptance/speed move.
- **They ship a ready 49,152-id list for exactly our vocab:
  `local/qwen38-draft-vocab-49152.txt` (min 0, max 248076, unique 49152).** Usable as-is to
  seed our draft vocab / `draft_keep_file`.
- Measured (SPEC 4): greedy 113.3 -> 120.0 t/s @8k (**+6%**, byte-identical text);
  sampled overall 76.3 -> 80.2 (**+5%**); prefill unchanged. SPEC 8 greedy is faster
  (149.2) but sampled loses 7% overall -> **they stay at SPEC 4**.
- Contrast with tcclaviger Option B (our current Phase 1): tcclaviger cuts rows + W4 kernel
  at load; mts prunes vocab at the torch level with **no new kernel**. Lower-risk, and it can
  be validated on our rig now with the shipped list, de-risking/paralleling the W4 port.

### B. Fused draft head (`RADIANCE_DRAFT_FUSED=1`)
- For the exact-set vocab path: the int2 kernel masks padding rows, sums its x groups from the
  tile it already loads, skips the coarse-score write, and the rerank writes its bf16 logits
  straight into the `-inf` full-vocab row. 15 -> 6 launches per drafter pass.
- Byte-identical drafts/logits (`dh_check.py`, m=1..16); sampled 90.5 -> 91.0, omp windows
  80.8 -> 81.3. Adopted. Ref: `r9700/review-20260927/radiance_drafthead.fused.diff`.

### C. `RADIANCE_DRAFT_EXACTSET=1`
- Only the exactly-reranked candidates are eligible for sampled drafts (the rest of the int2
  row is coarse 2-bit scores a sampled draft could otherwise pick). Lossless: overall
  80.4 -> 81.0; greedy unaffected. Rejected: `RADIANCE_DRAFT_TOPKP=1` (draft nucleus costs
  time and drops tokens the target's top-20 accepts).

### D. TunableOp table for the skinny fp8 GEMMs (+5%) — our Phase 3.1
- `fp8_tune.py` sweeps the 6 shapes at every M the V2 runner uses (1..12) -> 78-entry table;
  loaded read-only via `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
  PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv`.
- MTP GEMMs 1.30 -> 0.85 ms, lm_head 2.47 -> 2.32 ms, step 41.8 -> 39.9 ms; sampled overall
  80.4 -> 84.5 (**+5.1%**), acceptance identical, text identical.
- Table carries validators (torch/HIP/hipBLASLt/rocBLAS/gfx) and is ignored on image change;
  prebuilt table `r9700/prebuilt/tunableop/skinny0.csv` is **gfx1201 / torch 2.11.0 / HIP 714 /
  hipBLASLt 100401 — same versions as our build**, so it may drop in directly.
- Requires the AOT-cache env-key fix (see F) or an env-variant boot can load the wrong pieces.

### E. MXFP4 drafter + GPTQ refit — our W4 draft head, validated
- `RADIANCE_MTP_MXFP4=1` (`local/patch_paroquant_install.py`): the drafter's 5 linears are
  requantized at load fp8 per-channel -> MXFP4 and served by the body's W4A8 kernel through
  one opaque op `radiance::mtp_mxfp4_linear`. Drafter 405 -> 213 MiB; step -5.2%; sampled
  85.9 -> 88.6 (**+3.1%**); acceptance -2..-6% (json most); distribution exact.
- GPTQ (`r9700/quant/mtp-gptq/mtp_gptq.py`) fits codes to the drafter's own calibration H
  (custom-quant v2 `gptq_mxfp4`); recovers ~45% of the acceptance loss (held-out omp
  +3.4% vs fp8, +2.4% vs RTN). Production uses GPTQ. Codes ship as a 215 MiB file with a sha1
  fingerprint of the fp8 weights they were fitted to (`RADIANCE_MTP_MXFP4_FILE`).
- `r9700/quant/paro-mxfp4-v2/mtp_refit.py` also self-distills the MTP head against the
  *quantized* body (it was trained on bf16 hidden states) — explains their higher acceptance
  vs llama.cpp's stock MTP tensors. This is tcclaviger's 4-bit-draft axis, done MIT-style.

### F. vLLM 0.29 gotchas that directly affect us
- **"vLLM 0.29 runs the V2 model runner — Radiance's V1 hooks are inert"** (review §191). Our
  runtime hooks must target the V2 runner path.
- **Second compile cache ignores the environment** (`torch_compile_cache/<hash>/rank_R_D/
  {backbone,eagle_head}`, keyed on [env, config, traced code, compiler], loaded by piece
  index): env-selected graph variants can load another variant's pieces and die in inductor
  (`copy_misaligned_inputs`). Fix `patch_aot_envkey.py` appends the sorted `RADIANCE_*` env to
  the piecewise hash factors — **needed before trusting any RADIANCE_* A/B on 0.29**.
- `patch_gdn_lazy.py` (+117/-24): lazy GDN is default-off (corrupts multi-turn chat).
- **fp8 KV is mantissa-bound and free**: per-head `--kvcalibration` buys nothing (e4m3's 3-bit
  mantissa fixes the ~2.65% RMS); bf16 KV would halve capacity for no measurable quality.
  -> **our Phase 2.1 can skip calibration.**
- SPEC depth under sampling: SPEC 3 80.8, **SPEC 4 84.5**, SPEC 5 83.7. We run SPEC 8 -> worth
  an A/B for sampled workloads (greedy is not comparable across depths).
- `RADIANCE_VERIFY_HEAD` int2 target verify head: exact for sampled only when top_k <= RERANK/4
  (RERANK >= 80 for top_k 20) plus a 0.33 GiB head; not adopted there.

### G. Other (non-MTP) wins in the branch
- libr4d GDN exact-decay kernels `rx9x`/`rx9z`: the stock e^80 clamp is a real PPL deviation
  (code +0.5-0.6%); rx9z prefill +1.4-1.7% at 100k+, bit-exact.
- Prefill conflict-free multi-row rotation producer ("rot4") and split-token per-token
  producers: +1.4-1.7% prefill, byte-exact.
- Round 7 VRAM -> KV pool: vision-tower UVA offload + fp8 embed UVA -> pool 273,333 ->
  375,633 tokens.
- Start -> finalist overall: **+19.5% sampled, +18% omp decode, -18.7% step @8k, prefill
  +13.8-15.7%**.

### Immediate actions for us
1. Validate the vocab-prune gain **now** with their 49,152-id list (A) — no rebuild, low-risk;
   feeds and de-risks the Phase 1 W4 port.
2. Port/verify `RADIANCE_DRAFT_EXACTSET` (C) and the fused draft head (B) if our
   `radiance_drafthead.py` shares the structure.
3. Evaluate their prebuilt TunableOp table (D) — matching torch/HIP/hipBLASLt versions.
4. Adopt the AOT env-key fix (F) before trusting any RADIANCE_* A/B on 0.29.
5. Phase 2.1: enable fp8 KV **without** calibration.
6. Consider the MXFP4+GPTQ drafter path (E) as the higher-ceiling variant of Phase 1.

## Pending Investigation
1. **Reduced vocab draft head** — NEW in 29.05.12. Needs `draft_keep_file` generation (observe token usage during benchmark run) and porting W4 integration from `qwen4_exp/amd/mtp.py` to our `Qwen3_5MTP`/`Qwen3NextMTP` classes. Confirmed: `qwen3_5_mtp.py` (329 lines) and `qwen3_next_mtp.py` (249 lines) in 29.05.12 have NO W4 draft head integration.
2. **DAVETHA_DRAFTER_QUANT** — REMOVED in 29.05.12. Old approach (int4 draft lm_head, 3.85x speedup) is no longer the recommended path. Decision pending: keep old approach (patches 090/091 already created) or adopt new reduced-vocab approach.
3. **fp8 KV cache** — tcclaviger uses `--kv-cache-dtype fp8` with `--kvcalibration`. We don't use fp8 KV cache. Evaluate impact on our 2-GPU data-parallel setup.
4. **AITER settings** — tcclaviger has `VLLM_ROCM_USE_AITER=0`, we have `VLLM_ROCM_USE_AITER=1` with `UNIFIED_ATTENTION=1`. Compare performance impact.
5. **TunableOp GEMM sweep** — tcclaviger has optional `CLAV_TUNABLEOP_SWEEP=1`. Evaluate for performance improvement.
6. **flash_attn** — tcclaviger has flash_attn 2.8.3, we don't. Evaluate if it would help.
7. **DRY repetition penalty** — tcclaviger has server-wide DRY. Evaluate if useful for our setup.
8. **Degenerate-loop detection** — tcclaviger has server-wide detection. Evaluate if useful.
9. **Fetch tcclaviger's libr4d codeberg repo** (codeberg.org/StillDeadcode/libr4d) to inspect actual kernel implementation.
10. **Fetch tcclaviger's vLLM fork** to inspect the full patch set.
11. **ROCm version** — tcclaviger uses ROCm 10.0, we use ROCm 7.14. Evaluate if upgrading would help.
12. **Deadcode's int2 verifier/predictor** — Developer mentions this was adopted, modified, and optimized in .12+. Need to understand what this is and how it relates to our setup.
13. **Davetha's "pixie dust"** — Developer mentions this is used to reuse validated decisions. Need to understand what this refers to.
14. **Fork HIP kernel packages** — Developer confirms HIP kernels are "baked in more" in 29.05.12. Extracted and analyzed: gdn_hip (MTP spec-verify kernel `gdn_verify_r`), clav_attn (unified attention), clav_ar (P2P-BAR all-reduce), clav_ag (P2P-BAR all-gather), mischip/q4hc (Qwen4Exp HyperConnection, not relevant), plehip (Qwen4Exp PLE, not relevant). Still need to analyze: rfhip (RFI/RFA quantized GEMM), fp8hip (block-scaled w8a8 FP8 GEMM), parohip (ParoQuant W4A8), and the full r4dhip kernel family.
15. **QSA index sharing** — MTPSpeculator in 29.05.12 has `share_mtp_topk_indices` feature for reusing QSA top-k indices across MTP steps. Only applicable to QSA models (Qwen4Exp), not our model. Confirmed not applicable.

## Confirmed Findings
- Our `RADIANCE_DYNAMIC_DRAFT` is more sophisticated than tcclaviger's dynamic spec decode (confidence + n-gram vs batch-size only)
- We already have confidence-gated early exit (RADIANCE_DRAFT_TAU)
- We already have batch-size schedule (RADIANCE_DRAFT_SCHEDULE)
- We already have R4D attention (R4D_ATTN_FP8=3)
- We already have AR quantization (RADIANCE_USE_R4D_AR_QUANT=1)
- **DAVETHA_DRAFTER_QUANT REMOVED in 29.05.12** — the old int4 draft lm_head approach is no longer the recommended path
- **New approach in 29.05.12**: Reduced vocab draft head using `draft_keep_file` (JSON of observed vocab IDs)
- **W4 draft head integration only in `qwen4_exp/amd/mtp.py`** — NOT in `qwen3_5_mtp.py` or `qwen3_next_mtp.py` (our model uses Qwen3_5MTP/Qwen3NextMTP)
- **New env vars**: `CLAV_DRAFT_HEAD` (default "reduced"), `CLAV_DRAFT_HEAD_REPLICATE` (default "0")
- **get_top_tokens() avoids full-vocab all-gather** in 29.05.12 (only (value, id) pairs)
- **Auto layout selection** (replicated vs split) in 29.05.12
- tcclaviger's fp8 KV cache is NOT in our image
- tcclaviger's TunableOp GEMM sweep is NOT in our image
- Our r4d.so (`/opt/vllm/lib/python3.12/site-packages/r4d.so`, 1.9MB) ALREADY contains `r4d_gemm_w4a16_nt_m64` C entry point
- Our current draft head: 2-bit quantized with Triton kernel (`radiance_drafthead.py`); tcclaviger: 4-bit with HIP kernel
- **Fork HIP kernel packages baked in (developer confirmed)**: gdnhip (GDN linear-attention fused prefill/decode/MTP spec-verify), hipattn/clav_attn (native full attention + QSA sparse), clav_allreduce/clav_allgather (P2P-BAR collective ops), rfhip (RFI/RFA quantized GEMM), plehip (PLE n-gram gather), mischip (7 packages: batch_memcopy, fused_state_copy, causal_conv1d, etc.), fp8hip (block-scaled w8a8 FP8 GEMM), parohip (ParoQuant int4 x fp8 W4A8), r4dhip/libr4d (big kernel family including w4a16 GEMM)
- **MTPSpeculator (29.05.12)**: Extends AutoRegressiveSpeculator, adds `share_mtp_topk_indices` for QSA index sharing across MTP steps (step 0 computes, steps 1+ reuse). Only applicable to QSA models like Qwen4Exp.
- **AutoRegressiveSpeculator (29.05.12)**: 1232 lines, has fork-local `patches/mtp_confidence_exit` (confidence-gated early exit), lifecycle hooks (on_prefill_begin/end, on_multi_step_decode_begin/end), CUDA graphs for prefill and decode
- **Qwen3_5MTP (29.05.12)**: 329 lines, NO W4 draft head integration, NO QSA index sharing — standard MTP implementation
- **Qwen3NextMTP (29.05.12)**: 249 lines, NO W4 draft head integration, NO QSA index sharing — standard MTP implementation
- **Qwen4ExpMTP (29.05.12)**: 970 lines, implements `set_skip_topk` and `compact_topk_indices` for QSA index sharing, has W4 draft head integration (reduced vocab approach)

## Relevant Files
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_keep/qwen38-draft-vocab-49152.txt` — mts r9700-tp1 49,152-id draft vocab for our exact vocab (248320); extracted 2026-09-29
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/REVIEW-LOG.md` — mts r9700-tp1 engineering review (rounds 1-7), extracted 2026-09-29
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/tunableop-skinny0.csv` — mts prebuilt TunableOp table (gfx1201, torch 2.11.0, HIP 714, hipBLASLt 100401)
- `/home/juup/radiance-vllm-mxfp4/aijuus/refs/r9700-tp1/patch_aot_envkey.py` — reference fix for the 0.29 second compile cache ignoring RADIANCE_* env
- `/tmp/kilo/mts-r9700/` — full clone of `mtstanfield/vllm-mxfp4@r9700-tp1`
- `/home/juup/radiance-vllm-mxfp4/aijuus/ASYNC-VERIFY-DESIGN.md`
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml`
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/050-fp8-mtp.patch`
- `/home/juup/radiance-vllm-mxfp4/aijuus/r4d_lib.py` — ctypes loader for r4d.so exposing W4A16 GEMM functions
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_w4_lmhead.py` — int4 group-128 draft lm_head implementation (old 29.05.2 approach)
- (removed 2026-09-28) `aijuus/patches/090-davetha-draft.patch` and `091-draft-w4-install.patch` — superseded by the runtime overlay; both `radiance_kernels.py` hooks now live in `aijuus/patches/092-radiance-kernels.patch`
- `/home/juup/radiance-vllm-mxfp4/radiance_kernels.py` — vLLM plugin hook entry point (`092-radiance-kernels.patch`; also overlaid from `/patches` at runtime)
- `/home/juup/radiance-vllm-mxfp4/radiance_drafthead.py` — existing 2-bit Triton draft head (baseline for comparison)
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/062-radiance-drafthead.patch` — our current 2-bit Triton draft head implementation
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/060-radiance-draft.patch` — dynamic draft scheduling patch
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/ocp-mtp8.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8-205k.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/mtp8.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/tc-dflash24.betterbench.html`
- `/home/juup/radiance-vllm-mxfp4/aijuus/bench/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-ft.betterbench.html`
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
- `/tmp/kilo/gdn_hip_29.05.12/` — extracted tcclaviger 29.05.12 gdn_hip package (native HIP GDN ops including `gdn_verify_r` MTP spec-verify kernel)
- `/tmp/kilo/clav_attn_29.05.12/` — extracted tcclaviger 29.05.12 clav_attn package (native unified attention kernel)
- `/tmp/kilo/clav_ar_29.05.12.py` — extracted tcclaviger 29.05.12 clav_ar (P2P-BAR all-reduce, 857 lines)
- `/tmp/kilo/clav_ag_29.05.12.py` — extracted tcclaviger 29.05.12 clav_ag (P2P-BAR all-gather, 448 lines)
- `/tmp/kilo/app_src_29.05.12/` — extracted tcclaviger 29.05.12 /app/src (mischip/q4hc, plehip HIP source files)
- `tcclaviger/vllm:29.05.12` (Docker image) — contains new reduced-vocab approach in `qwen4_exp/amd/mtp.py`, `draft_w4_lmhead.py` (updated), new clav_* packages, HIP kernel packages in /app/
- `/opt/vllm/lib/python3.12/site-packages/r4d.so` — our r4d library (has `r4d_gemm_w4a16_nt_m64` entry point)

## Next Steps
1. **Decision**: Choose between old DAVETHA_DRAFTER_QUANT approach (patches 090/091 ready) vs new reduced-vocab approach from 29.05.12. Code analysis complete — W4 draft head integration only in Qwen4ExpMTP, not in Qwen3_5MTP/Qwen3NextMTP.
2. **If choosing new approach**: Generate `draft_keep_file` by observing token usage during benchmark run, then port W4 draft head integration from `qwen4_exp/amd/mtp.py` to `Qwen3_5MTP`/`Qwen3NextMTP`
3. **If choosing old approach**: Build Docker image with patches 090/091 applied (`aijuus/apply.sh` then docker build), deploy, and benchmark DAVETHA_DRAFTER_QUANT=0 vs 1
4. **Analyze remaining HIP kernel packages**: rfhip (RFI/RFA quantized GEMM), fp8hip (block-scaled w8a8 FP8 GEMM), parohip (ParoQuant W4A8), full r4dhip kernel family
5. **Evaluate gdn_verify_r** — the fused MTP spec-verify kernel from gdn_hip could replace our current MTP verify path for performance improvement
6. **Evaluate clav_attn** — native unified attention kernel could replace our current attention backend for performance improvement
7. **Evaluate clav_ar/clav_ag** — P2P-BAR collective ops could improve our 2-GPU data-parallel communication
8. **Evaluate fp8 KV cache** — could allow more concurrent sequences or longer context without OOM
9. **Compare AITER settings** — test `VLLM_ROCM_USE_AITER=0` vs our current `=1` with `UNIFIED_ATTENTION=1`
10. **Evaluate TunableOp GEMM sweep** — `CLAV_TUNABLEOP_SWEEP=1` for GEMM performance improvement
11. **Fetch tcclaviger's vLLM fork** — inspect full patch set for additional optimizations (especially deadcode's int2 verifier/predictor and davetha's "pixie dust")
12. **Benchmark our current setup** with `RADIANCE_DYNAMIC_DRAFT` enabled vs disabled to measure its impact

## External References
- https://hub.docker.com/r/tcclaviger/vllm
- https://blog.robai.net/Qwen3.8-Flash-Next-MXFP4-TP4-MTP4-29.05.7/
- https://blog.robai.net/vllmdocs/
- https://codeberg.org/StillDeadcode/libr4d
