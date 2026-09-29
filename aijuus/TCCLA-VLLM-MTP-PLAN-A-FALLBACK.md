# Plan A: DAVETHA_DRAFTER_QUANT (Fallback)

## Status
**FALLBACK PLAN** — Only use if Plan B (reduced-vocab approach) fails. Do not mix with Plan B work.

> **SUPERSEDED PATCHING (2026-09-28):** the overlay no longer bakes these modules. The W4 draft
> head is wired at runtime by `radiance_kernels._install_draft_w4()` (gated on
> `DAVETHA_DRAFTER_QUANT=1`), and `aijuus/kv-offload/ops/entrypoint.sh` overlays
> `draft_w4_lmhead.py` / `r4d_lib.py` / `qwen3_5_mtp_w4.py` from `/patches` at container start.
> Patches `090-davetha-draft.patch` and `091-draft-w4-install.patch` were **deleted**; the
> `radiance_kernels.py` changes are captured by `092-radiance-kernels.patch`.

## Overview
Full-vocab int4 draft lm_head quantization using libr4d's `r4d_gemm_w4a16_nt_m64` HIP kernel. This was tcclaviger's approach in 29.05.2 but was **REMOVED in 29.05.12** (2026-09-27).

## Why Fallback
- tcclaviger removed this approach, suggesting they found issues
- New reduced-vocab approach (Plan B) is the current recommended path
- Keep this as a safety net if Plan B proves difficult or underperforms

## Implementation Status
**READY TO DEPLOY** — wired at runtime (no baked patches):

### Wiring
- `radiance_kernels._install_draft_w4()` (captured by `092-radiance-kernels.patch`), gated on `DAVETHA_DRAFTER_QUANT=1`: remaps the registry to `aijuus.qwen3_5_mtp_w4.Qwen3_5MTPW4`, registers the W4 op, and points `draft_keep_file` at `RADIANCE_DRAFT_KEEP_FILE`.
- `aijuus/kv-offload/ops/entrypoint.sh` overlays `draft_w4_lmhead.py` → `vllm/model_executor/kernels/`, `r4d_lib.py` and `qwen3_5_mtp_w4.py` → `$SP`.

### Files
- `aijuus/r4d_lib.py`: ctypes loader for r4d.so exposing `r4d_gemm_w4a16_nt_m64`, `r4d_gemm_w4a16_nt_m64_max_m`, `r4d_gemm_w4a16_nt_m64_group`
- `aijuus/draft_w4_lmhead.py`: int4 group-128 draft lm_head with `pack_w4a16`, `gemm_w4a16`, `install()`, `PackedW4` dataclass, clip-grid quantization, launch config selection

### Compose Changes
- `aijuus/coolify-compose-2gpu.yml`: `DAVETHA_DRAFTER_QUANT=${DAVETHA_DRAFTER_QUANT:-0}` added to both vllm-0 and vllm-1 services

### Integration
- `draft_w4_lmhead.install()` wraps drafter `load_weights` to quantize lm_head post-load
- Patches `LogitsProcessor._apply_head` to use W4A16 GEMM for draft head inference

## How to Activate (if needed)
1. Set `DAVETHA_DRAFTER_QUANT=1` (and `RADIANCE_DRAFT_KEEP_FILE` if not using the default) in the compose env
2. Redeploy — no rebuild (runtime overlay)
3. Deploy with `DAVETHA_DRAFTER_QUANT=1` in compose env
4. Benchmark: compare `DAVETHA_DRAFTER_QUANT=0` vs `1`

## Expected Performance
- Measured by tcclaviger (R9700 gfx1201, N=124160 K=2560): 3.85x at M=1, 3.45x at M=20
- Production: step 41.7→38.9ms, acceptance -3..-9% relative, net +4..+8% tok/s

## Known Limitations
- Full-vocab quantization (quantizes all 124160 vocab rows)
- Requires full-vocab all-gather in `get_top_tokens()` (doesn't apply to our 2-GPU data-parallel setup)
- tcclaviger removed this approach (reason unknown — possibly accuracy or performance issues at scale)

## Risks
- tcclaviger removed this for a reason — unknown what issues they encountered
- May not scale well with larger vocab or higher concurrency
- No layout selection (always full vocab on every rank)

## Reverting Plan A
Plan A is now runtime-gated (`DAVETHA_DRAFTER_QUANT=1`), so reverting is a config change, not a rebuild:
1. Set `DAVETHA_DRAFTER_QUANT=0` in the compose env (or drop it)
2. Redeploy — with the flag off the overlay keeps the stock bf16 head

## Relevant Files
- `/home/juup/radiance-vllm-mxfp4/aijuus/patches/092-radiance-kernels.patch` (runtime hooks: `_install_token_collector`, `_install_draft_w4`)
- `/home/juup/radiance-vllm-mxfp4/aijuus/kv-offload/ops/entrypoint.sh` (runtime overlay)
- `/home/juup/radiance-vllm-mxfp4/aijuus/r4d_lib.py`
- `/home/juup/radiance-vllm-mxfp4/aijuus/draft_w4_lmhead.py`
- `/home/juup/radiance-vllm-mxfp4/aijuus/coolify-compose-2gpu.yml`
- `/home/juup/radiance-vllm-mxfp4/radiance_kernels.py`
- `/home/juup/radiance-vllm-mxfp4/radiance_drafthead.py` (existing 2-bit Triton draft head baseline)
