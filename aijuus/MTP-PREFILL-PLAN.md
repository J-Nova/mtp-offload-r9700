# MTP PREFILL PLAN

Scope: raise prompt-processing (prefill) throughput of the MTP blend on vLLM 0.29 / RDNA4 / TP=1.
**Excluded by decision:** the offload, power-cap, and TP=2 A/Bs (items 2/3/4 of the earlier list).
This plan covers kernel-level prefill work (item 5) and upstream PR ports as overlays.

Metric: BetterBench prefill sweep (`--quick` off, depths 2k/8k/16k/32k/64k) on vllm-0. Current
baseline (results/mtp-vllm-0.json): **1813 / 1995 / 2117 / 2050 / 1909 PP t/s**. Reference TP=1
(README, native MXFP4): 2737 / 2720 / 2778 / 2651 / 2480. Target: close the kernel-attributable part.

Baseline accounting already established:
- MTP adds a **draft-prefill pass** (speculator `prefill_cudagraph_manager` / `_prefill`,
  `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:136-187`). Not measured in isolation yet.
- Offload store is ~18% of a cold single-prompt prefill (measured), async but awaited at finish.
- Quant dispatch: `radiance_rmsquant.py:9-16` — 8,512 quant launches at prefill = **4.2% of wall**.

---

## Task index (resume here in a fresh chat)

Status legend: **READY** = actionable now, no blockers · **BLOCKED** = needs a reload/A-B you own ·
**VERDICT** = analysis done, only a portable action remains · **PARKED** = deliberately out of scope ·
**N/A** = not applicable to our stack.

| ID | Task | Status | Deps | Expected | Key files / refs |
|----|------|--------|------|----------|------------------|
| A1 | Fold remaining per-token quants into epilogues | **VERDICT (largely already fused)** | none | remaining ceiling is small, not 4.2% | see note below; `radiance_mxfp4.py:923`, fp8-stream |
| A2 | R4D prefill geometry h256 + KV-block analyze | **READY (needs reload)** | reload per arm | unknown; measure first | libr4d attn table, `radiance_kernels.py:118`, `kv-profiles.tsv` |
| A3 | MXFP4 prefill GEMM (TN4 sweep / split-K) | BLOCKED | reload per arm | diminishing (~+10% @M8192 documented) | `radiance_mxfp4.py` |
| A4 | GDN prefill | PARKED | profiling first | low | `radiance_gdn*.py` |
| B1 | #58114 PLE — verdict + portable D2H-sync audit | **VERDICT (nothing to port)** | none | none found | `gdn_attn.py` already CPU-side |
| B2 | #58845 MLA qlnorm — verdict + redundant-call audit | **VERDICT** | none | audit only | our attention forward |
| B3 | #57951 long-prefill-alone scheduler overlay | BLOCKED | C decision | small; needed only if threshold≠0 | new `patch_sched_long_prefill_alone.py` |
| B4 | #54440 rDNA h256 backend finding | BLOCKED | reload | A/B R4D vs AITER attn at h256 | `R4D_ATTN=0` switch |
| B5 | #39060 speculative/sparse prefill | PARKED | research track | biggest TTFT, high cost | scheduler + model runner |
| C | `long_prefill_token_threshold` policy | **VERDICT** | B3 if adopted | keep 0 unless p99 ITL bad | `config/scheduler.py` |
| D | Measurement & validation harness | **READY** | per change | — | BetterBench sweep + 18k probe + GSM8K |
| E1 | Suffix-only invalidation for volatile MTP/draft KV | **READY** | none | decode/multi-turn; enables MTP-KV offload | offload/prefix path (see `ADAPTIVE-KV-STREAMING.md`) |
| E2 | reserved/device-ready/host-ready/committed publication | **READY** | none | targets the ~18% prefill store await | offload connector `wait()` path |
| E3 | Host-spilled GDN rollback (2-slot stage) | READY | research | unblocks `RADIANCE_GDN_LAZY=0` | spec/GDN snapshots |
| E4 | Phase arena (prefill ws → decode KV) | PARKED | excluded by decision | their +4.98% decode | — |
| E5 | cross-token prefetch / sparse feedback | PARKED | decode-only | +2.26% / +0.63% decode | — |
| E6 | PDL early-completion fence lesson | N/A | — | CUDA-only, no ROCm analogue | — |

**Recommended resume order:** A2 (R4D prefill geometry sweep) / B4 (R4D-vs-AITER h256 A/B) → D → E2 → E1 → B3/C → A3 → B5.
Closed 2026-09-30: **A1** (fusion already on; optional confirmation A/B only) and **B1** (no D2H sync to remove).

---

## Workstream A — kernel-level

### A1. Fold the remaining per-token quants — VERDICT: largely already fused (2026-09-30)
The 8,512-prefill / 14,288-decode figures in `radiance_rmsquant.py:9-16` are the **pre-fusion
motivation**, not remaining work. The activation quant is emitted in the *traced* region
(`radiance_mxfp4.py:923` `_traced_quant`, gated by `RADIANCE_MXFP4_HOIST_QUANT=1` +
`RADIANCE_MXFP4_TRACED_QUANT=1`), which is exactly what lets vLLM's `AiterRMSNormDynamicQuantPattern`
fold it into the preceding `rms_norm`. Our env has that whole chain on
(`FUSE_RMS_QUANT=1`, `FP8_STREAM=1`, `GDN_NORM_QUANT=1`) and boot reports
`fp8 stream installed: 64 mid epilogues, 63 down streams, 64 act epilogues, 48 gdn norm+quant epilogues`.

Remaining **unfusable-by-that-pattern** quants (input is not an rms output) and their true size:
- attention output → `o_proj` input: ~16 full-attention layers per step (small).
- KV-cache quant if fp8 KV is materialized outside the attention kernel (needs confirmation).
- drafter/MTP head quants (mostly bf16 / int2 head; likely none).
→ Ceiling is single-digit percent, not 4.2%. Do **not** prioritize A1.

Confirmation A/B (optional, one reload): `RADIANCE_FUSE_RMS_QUANT=0`+`RADIANCE_FP8_STREAM=0` vs on —
if that costs >>2% prefill, the fusion is already earning its keep and A1 is closed.

### A1b. Fold the attention-output → o_proj quant (only if the above A/B shows headroom)

### A2. Prefill attention (R4D, h256 gqa6 fp8kv)
We run `ATTN=R4D` (`serve-mxfp4.sh:784`), so the AITER `_PREFILL_2D_BY_HEAD` tune table
(`radiance_kernels.py:118`) does **not** apply — that table backs the AITER unified-attention path only.
Levers:
1. Tune the **R4D prefill** launch geometry for h256/large-q (BLOCK_M / TILE / num_warps / waves) — the
   R4D selection table (see `RADIANCE_R4D_REPORT=1` boot table) currently has no h256 prefill-specific
   entry; add one and sweep, as was done for head_size 512 (BLOCK_M 32 worth -30% TTFT @32K).
2. Reduce per-attention kernel calls (principle from upstream #58845): audit the R4D prefill path for a
   redundant rope/quant/projection launch that can be folded.
3. KV block size: at TP=1 attention block is **880**, forced down by the fp16 ssm page under
   `mamba_cache_mode=align`. Analyze (do not assume): whether a larger attention block (fewer, larger
   pages) helps prefill paging, and the ssm-dtype tradeoff (fp16 vs fp32 page) needed to allow it. This
   is a profile first, then a config experiment. See `kv-profiles.tsv` + SINGLE_GPU_PROFILE notes.
Files: libr4d prefill attn kernel + its selection table, `patch_unified_attention_lds.py` (correctness
clamps stay), `radiance_attn*`.

### A3. MXFP4 prefill GEMM (diminishing)
A-tiled (`RADIANCE_MXFP4_A_TILED_MIN_M=513`) + TN4 (2048) + EPIFAST are the tuned path. Remaining:
- Sweep `RADIANCE_MXFP4_TN4_MIN_M` per prefill M (documented +10% at M=8192); confirm the current 2048 is
  optimal for the swept chunk sizes.
- Evaluate a persistent / split-K prefill GEMM variant against A-tiled. Low priority; measure before
  investing.

### A4. GDN prefill (low)
Fused chunk-scan + WMMA solve + conv BLOCK_N 1024 already in. Little left without a new kernel; revisit
only if profiling shows GDN scans dominating at short depth.

---

## Workstream B — upstream PRs (deep dive + verdict + port)

### B1. #58114 `[Perf][Qwen3.8] Reduce PLE metadata construction overhead`
**Verdict: NOT applicable to our checkpoint.** The PR touches `vllm/models/qwen4_exp/*` (PLE —
per-layer n-gram embeddings, Qwen3.8 **2.4T Flash-Next**). Our checkpoint is arch
`Qwen3_5ForConditionalGeneration`: `config.json` has only the multimodal wrapper keys
(`text_config`/`vision_config`/`language_model_only`), and the weight index has **no** PLE/ngram tensors
(only `embed_tokens`, vision patch/pos, `mtp.pre_fc_norm_embedding`). So our "Qwen3.8-27B" is qwen3_5,
not qwen4_exp. Consistent with repo note `TCCLA-VLLM-MTP-RESEARCH.md:184`.
**Portable idea (worth mining):** the diff (a) removes a **prefill-time CPU round-trip** from the
Mamba/short-conv metadata builder — it deletes `query_start_loc_cpu` /
`compute_causal_conv1d_metadata` from `short_conv_attn.py` and threads a precomputed
`query_start_loc_p` instead, explicitly "to avoid a device-to-host synchronization"; (b) adds
`needs_causal_conv1d_metadata=False` so a builder skips the conv metadata entirely.
Action: audit our prefill metadata builder(s) for a `compute_causal_conv1d_metadata(...)` /
`query_start_loc_cpu` D2H sync on the prefill path and, if present, overlay the same elimination.

### B2. #58845 `[GLM5.3] Skip qlnorm for MHA, 4.4~7.7% E2E TTFT`
**Verdict: arch N/A** (deepseek_v32 MLA — it skips the `ql_nope = bmm(q_nope, W_UK_T)` absorbed
projection when the MHA path is taken). We run GQA h256, no MLA/absorption.
**Portable ideas:** (a) the *category* — a redundant per-attention projection/kernel call removed for a
specific path; audit our attention forward for the same; (b) their capture guard
`cudagraph_runtime_mode == CUDAGraphMode.FULL or torch.cuda.is_current_stream_capturing()` to skip work
during graph capture — relevant if any of our overlay work runs redundantly during capture.

### B3. #57951 `[Scheduler] Soften long prefill token threshold`
**Verdict: portable, small.** It makes `long_prefill_token_threshold` **not** cap a lone request (nobody
to starve) — only caps when >1 request is in flight, hoisted to a local:
```python
long_prefill_token_threshold = (self.scheduler_config.long_prefill_token_threshold
    if len(self.running)+len(self.waiting)+len(self.skipped_waiting) > 1 else 0)
```
Port as `patch_sched_long_prefill_alone.py` (scheduler thread; adds a local + two call-site swaps,
idempotent, gated). Only meaningful if we adopt a nonzero threshold (see Workstream C).

### B4. #54440 `[ROCm] Rank ROCM_ATTN by whether its custom kernel applies on RDNA`
**Verdict: directly informative, but not a drop-in.** On gfx1x, ROCM_ATTN's custom paged-attention HIP
kernel exists only at head_size 128; at other head sizes it falls back to Triton, and the PR demotes it
below TRITON_ATTN because at head_size 256 the ranking winner is slower at depth. **We are at head_size
256** — exactly the case. But we serve with the **R4D** backend, not ROCM_ATTN, so the ranking change
doesn't apply; the actionable part is the finding: at h256 on RDNA the "obvious" backend is not the
fast one. Action: A/B our R4D prefill vs AITER unified attention at h256 for the prefill sweep (we have
`R4D_ATTN=0` as a switch), to confirm R4D really wins prefill at h256 — never re-verify this by
assumption.

### B5. #39060 `Speculative Prefill — draft-assisted sparse prefill for TTFT`
**Verdict: open feature request, not merged; high reward, high cost.** Algorithm: score prompt chunks
(e.g. 32-token) by importance (attention pattern), prefill only the top-k% tokens into the target, keep
original positions via manual RoPE patching, then decode normally. Orthogonal to chunked/disagg prefill
and shares the draft model with spec decode.

PoC path (only if A/Bs show prefill dominates and quality allows 90–100% retention):
1. Prototype offline: score chunks with the MTP head's attention on a sample of prompts; measure
   accuracy vs retention (their cited 0.2 keep-pct, 90–100% baseline retention).
2. If quality holds, integrate as an overlay: scheduler picks a chunk subset + position map; model
   runner prefills the subset with position-preserving RoPE; verify GSM8K/BetterBench quality gates.
3. This is the biggest single TTFT reducer in the list but needs model-level changes — treat as a
   separate research track, not a quick patch.

---

## Workstream C — `long_prefill_token_threshold` deep dive

Current value: **0 (disabled)**. Semantics: for chunked prefill, a pending prefill is capped to this many
tokens per step even if the token budget is larger — it exists so one long prefill doesn't consume the
whole `max_num_batched_tokens` (4096) and starve co-scheduled decodes/other prefills.

- With MTP, `draft_slots = max_num_new_slots_for_drafting = 0` for single-module MTP
  (`.../config/speculative.py:1786-1825`), so `input_budget ≈ max_num_batched_tokens = 4096`; the
  threshold caps against that.
- Effect of a nonzero threshold: **lower TTFT for long prompts is lost** (the prompt is split into
  threshold-sized steps), while **ITL under mixed load improves** (decodes aren't starved). It is a
  TTFT↔ITL trade decided by the concurrency mix.
- Our Shape: LB runs interleaved chat at up to ~16 concurrent; a single long prompt at low concurrency
  benefits from 0. The current BetterBench lvl-1 is a lone request, where threshold is irrelevant — and
  #57951 makes that explicit.
- Recommendation: **keep 0** unless p99 ITL under mixed load is a problem. If we ever set it, land B3
  first (so it only caps when >1 request), and pick the value as `~max_num_batched_tokens/2` (2048) and
  measure TTFT/ITL on the concurrency sweep, not the lone-prompt sweep.

---

## Workstream D — measurement & validation

Per change, one vllm-0 reload:
1. BetterBench prefill sweep + the single ~18k probe (for store timing).
2. Kernel changes: report launches/step (`RADIANCE_MXFP4_MHIST=1` or `RADIANCE_STEP_TRACE=N`) alongside
   PP t/s, so a win is attributable to dispatch reduction, not noise.
3. Quality gate: GSM8K (`aijuus/bench-eval.py` / BetterBench quality) on any change touching attention,
   GEMM, or quant (these are numerics-adjacent).
4. Acceptance criteria: PP t/s per depth; a change ships only if it is +>=2% at >=2 depths and neutral
   elsewhere. A1 target: recover most of the 4.2% quant dispatch. A2 target: measure first.

## Workstream E — llama.cpp Adaptive-KV-Streaming borrowables (cross-ref)
Deep-dive: `aijuus/ADAPTIVE-KV-STREAMING.md` (llama.cpp V2 `RaymondHuang210129/llama.cpp-adaptive-kv-streaming`,
README + MTP roadmap + device-memory infra). ggml/CUDA-internal code does not port; the mechanisms do:
- **E1 suffix-only invalidation for volatile MTP/draft KV** — our boot log already excludes
  `EAGLE/MTP draft attention groups [8]` from offload "due to volatility"; theirs keeps MTP as logical
  layer 17 and truncates only the rejected suffix (prefix rebased without re-transfer). Transferable to
  our offload/prefix path; highest near-term value.
- **E2 reserved/device-ready/host-ready/committed publication** — the design that removes our measured
  ~18% prefill offload-store await: let KV durability lag attention/response, advance a host-committed
  frontier only over contiguous generation-matched host-ready ranges.
- **E3 host-spilled recurrent rollback** (2-slot GPU stage -> pinned host snapshots) — the shape of a
  correct+cheap fix for our disabled lazy GDN snapshots (`RADIANCE_GDN_LAZY=0`).
- **E4 phase arena** — reclaim idle prefill workspace for decode KV (their 6.3c: +4.98% decode). Excluded
  here by decision; documentation-level follow-up.
- **E5 cross-token prefetch / sparse feedback** — decode-only, single-digit gains (6.6 +2.26%,
  6.7a +0.63%); park. **E6** PDL lesson is CUDA-only (no ROCm analogue).
- Calibration: their own adaptive-KV path pays **-4% to -23% prefill** for stock arithmetic; this repo
  does not solve prefill — it is a decode/long-context system.

## Risks / rollback
- **Deploy env precedence (2026-09-30):** Coolify's UI environment overrides the compose
  `${VAR:-default}` expressions. Evidence: after recreating vllm-0, my *newly added* compose lines
  (`RADIANCE_DRAFT_STATS=0`, `RUN_BWTEST=0`, `R4D_REPORT=0`) applied, but `RADIANCE_OFFLOAD_INSTRUMENTATION`,
  `_LOOKUP_METRICS`, `_DEBUG_INSTRUMENT`, `_TIER_REPORT`, `_PROMOTION_WALLCLOCK`, `_MISS_DEFERRAL_METRICS`
  and `COLLECT_TOKENS` stayed `1` — they are Coolify-managed. Deploy flag changes must be made in the
  Coolify env (or the var removed there), not only in `aijuus/coolify-compose-2gpu.yml`.
- Every change re-keys the AOT cache (fresh compile). Revert = drop the env gate / overlay patch.
- Attention/GEMM/quant changes are numerics-adjacent: keep the GSM8K gate.
- A2 KV-block change interacts with `mamba_cache_mode` and the KV pin: re-run `calibrate-kv.sh` if the
  block size changes.
- #39060 (speculative prefill) is a research track, not a near-term patch.

## Order
1. A2 (R4D prefill geometry sweep) + B4 (R4D vs AITER h256 prefill A/B) — the remaining prefill lever.
2. D measurement harness (launch/PP attribution).
3. E2 offload publication split (targets the ~18% store await) → E1 suffix-only invalidation.
4. B3 + C — scheduler overlay, only if a nonzero threshold is adopted.
5. A3 sweep; B5 research track.
6. A1 confirmation A/B (optional) and A1b (o_proj quant fold) only if it shows headroom.
