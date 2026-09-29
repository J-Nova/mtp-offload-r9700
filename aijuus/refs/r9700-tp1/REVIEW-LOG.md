# vLLM-Radiance review and A/B (2026-09-27)

Static review of vLLM-Radiance (finalist PARO-MXFP4 v2, 0.29 prebake image, libr4d b9e42ab-rx9, single-GPU profile) using
what the llama.cpp HIP work taught about gfx1201, then measured A/B of every lead on the R9700 at 250 W. Rig: container
`vllm-exp` on :1252 (`deploy-vllm-exp.sh` = the next-wrapper with NAME/PORT overridable; the watchdog and omp never see it),
driven through the finalist launcher (`WRAPPER=deploy-vllm-exp.sh bash deploy-vllm-qwen38-paro.sh`). Scripts and diffs:
`vllm-patches/review-20260927/`.

## Summary

| lead | result | recommendation |
|---|---|---|
| GDN chunk-scan midpoint split + e^80 clamp (libr4d) | real deviation: code PPL +0.5-0.6% vs the FLA reference (wiki -0.3%); our exact-decay kernel matches the reference, no measurable speed cost | adopt `R4D_KEY=b9e42ab-rx9x`; offer the fix upstream (libr4d has no LICENSE -- user's call) |
| Prefill GDN falls back to FLA on the fp16 state | WRONG lead: the main prefill path converts the state to fp32 and runs libr4d's scan; FLA only takes declined steps. 16-bit-state scan = no speed change | nothing to adopt (rx9x carries the entry points anyway) |
| MTP draft head reads the whole vocabulary | the launcher already runs the int2 head (FAST_DRAFT=1); our 48k draft vocabulary on top: +5% sampled, +6-7% greedy, byte-identical greedy text | adopt `FAST_DRAFT=0` + `RADIANCE_DRAFT_VOCAB` |
| SPEC 8 | +24-37% greedy, but sampled -7% overall (code/prose lose, json wins) and the pool drops 273,333 -> 240,908 tokens | stay at SPEC 4 |
| fp8 conversions / KV scale / grid sizing | activation quant clamps to +-448; attention P bounded; KV at scale 1.0 saturates (no NaN); no occupancy-API sizing | nothing to do |

## 1. GDN chunked scan: the clamp is a measurable deviation

`r4d_gdn_chunk_scan_k128_v128_c64_bf16` splits the decay e^{g_i-g_j} at the chunk midpoint so each half needs e^{span/2}; spans
past ~176 overflow, so the developer clamps each half at e^80 (PERFORMANCE.md "The three overflows", "the clamp bounds the
damage; it does not remove the cause"). Tokens whose distance from the midpoint exceeds 80 get their diagonal term and their
weight into the carried state attenuated.

**rx9x** (`patch_rx9x.py`, built at `persist/libr4d-029/b9e42ab-rx9x`): exact decay with every weight a factor <= 1
(gv = e^{g_last-g_t} on the state path, sA = scale * e^{g_i-g_j} per causal element, S0 scaled by e^{g_last}, V' staged
unscaled and rescaled after O), plus `_f16state` / `_bf16state` entry points registered ahead of the fp32 row.

Synthetic check (`gdn_check.py`: 48 v heads, 2,048 tokens, heads 0-11 with g in [-6, 0), vs an fp32 recurrent reference):

| build | o NMSE, high-span heads | final state NMSE, high-span | other heads | kernel us |
|---|---:|---:|---:|---:|
| stock rx9 (clamp) | 1.5e-1 | 0.997 | 1e-5 | 262 |
| rx9x fp32 / fp16 / bf16 state | 9.5e-6 | 2.6e-6 / 2.6e-6 / 5.5e-6 | 1e-5 | 309 / 305 / 304 |

Served PPL (`p7-vllm-ppl.sh`: KV_MEM 4e9, MAXLEN 65536, wikitext-2 test 60 x 2048, code sample 30 x 2048; FLA reference =
`RADIANCE_GDN_SCAN_OFF=1`, a local switch that sends the whole layer to the stock path):

| state | FLA (reference) | rx9x exact | stock clamp |
|---|---|---|---|
| fp32 (profile OFF) | 5.9864 / 2.3052 | 5.9810 / 2.3037 | 5.9659 / 2.3163 |
| fp16 (prod profile) | 5.9757 / 2.3037 | 5.9846 / 2.3065 | **5.9592 / 2.3170** (= prod) |

Exact tracks the reference within 0.1-0.15%; the clamp moves code PPL +0.5-0.6% and wiki -0.3%, i.e. it changes the model's
behaviour rather than improving it. Prefill speed with rx9x: unchanged (2,912 vs 2,939 @8k, 2,583 vs 2,589 @48k, 2,139 vs
2,150 @112k -- within noise).

## 2. Draft head

The launcher defaults `FAST_DRAFT=1` for MTP too: prod already scores drafts with an int2 copy of the fp8 lm_head (0.33 GiB,
coarse pass + exact rerank of 32). `RADIANCE_DRAFT_VOCAB=<ids>` (radiance_drafthead.py, local) runs that same machinery on the
48k-row sub-matrix (0.07 GiB) and fills the rest of the row with -inf; the target verifies with its own head, so output cannot
change.

Greedy (`oai_bench_greedy`, same prompt, 600 tokens):

| arm | @8k tok/s (ms/step) | @100k | tok/step @8k |
|---|---:|---:|---:|
| A prod (int2 full head, SPEC 4) | 113.3 (42.0) | 91.1 | 4.76 |
| B stock fp8 head | 99.5 (47.9) | 86.0 | 4.76 |
| **C 48k vocab** | **120.0 (39.7)** | **97.1** | 4.76, same text |
| D 48k vocab, SPEC 8 | 149.2 (49.6) | 125.1 | 7.41 |
| E int2 full, SPEC 8 | 142.9 (59.1) | 119.5 | 8.45 |

Sampled (`spec_sampled.py`, production sampling from the server, 8 runs x 600 tokens per prompt):

| arm | code | prose | json | overall |
|---|---:|---:|---:|---:|
| A prod | 75.2 | 63.6 | 97.2 | 76.3 |
| **C 48k vocab** | **80.2** | **68.0** | 97.6 | **80.2** |
| D 48k vocab, SPEC 8 | 71.6 | 59.0 | 106.9 | 74.5 |

Prefill is identical across A-E.

## 3. Same-harness comparison with llama.cpp HIP (production, n-max 7, 48k draft vocab)

| | code | prose | json | greedy @8k (oai_bench_greedy) |
|---|---:|---:|---:|---:|
| vLLM prod / + 48k vocab | 75.2 / 80.2 | 63.6 / 68.0 | 97.2 / 97.6 | 113.3 / 120.0 |
| llama.cpp n-max 7 / 4 / 3 | 42.3 / 40.9 / 41.3 | 41.4 / 40.6 / 39.1 | 62.0 / 59.3 / 49.9 | 44.1 |

Server-side timings agree with the client-side stream timing. On sampled chat decoding vLLM is ~1.6-1.9x faster: per step
~42 ms at 3.1 tok/step (SPEC 4) vs llama.cpp ~64 ms at ~2.7 tok/step (acceptance ~0.45 at n-max 7). vLLM's MTP head was
self-distilled against its quantized body (paro-mxfp4-v2 mtp_refit); llama.cpp uses the GGUF's stock MTP tensors. The earlier
"llama.cpp 118 vs vLLM 77" compared bench_decode_depth (a highly predictable raw-text continuation) with oai_bench_greedy --
different prompts; retracted. n-max 7 stays llama.cpp's best setting under sampling.

## State left behind

- llama.cpp production restored (`deploy-qwen38-27b.sh`, n-max 7); `vllm-exp` removed.
- `vllm-radiance-next` tree (the finalist's launcher tree): radiance_gdn.py and radiance_drafthead.py carry the env-gated local
  changes (`*.bak-20260927` = originals; diffs here); with the envs unset and the stock libr4d they behave exactly as before.
  `local/qwen38-draft-vocab-49152.txt` added.
- To run vLLM with the recommendations: `R4D_KEY=b9e42ab-rx9x FAST_DRAFT=0
  EXTRA_ENV="<finalist list> -e RADIANCE_DRAFT_VOCAB=/patches/local/qwen38-draft-vocab-49152.txt" bash deploy-vllm-qwen38-paro.sh`.

## Round 2 -- correctness review (user: correctness first; production taken down as needed)

### Prefill attention fp8 legs (R4D_ATTN_FP8=3 in production) -- FIXED (libr4d rx9y, mode 15)

Kernel harness `attn_check.py` (24/4 heads, head_dim 256, fp8 KV, the last 256 queries of an N-key context, Gaussian
logits of spread sigma, optionally a shared component in V) vs an fp32 reference: row error f16 legs 0.2%, production
3.7-7.6% (grows with sigma). The fp8-P leg computes p = 2^(s - m_ref) with no shift and lets m_ref lag 8 octaves, so a p
below 2^-10 of the reference max rounds to 0 in the numerator -- while the denominator sums the pre-quantization p. With a
shared value component the output shrinks: norm ratio 0.980 (sigma 2) / 0.946 (sigma 3) at 128k keys.
rx9y (`patch_attnden.py`, `patch_attnsh.py`): PVDEN (denominator over the e4m3-rounded p) + PVSH (4-octave shift, 4-octave
lag). New modes 6/7/14/15. End to end (`lastpos_kl.py`, 80 next-token cut points at 32k-115k, vs mode 0):

| mode | KL | p90 | top-1 | prefill 48k / 120k |
|---|---:|---:|---:|---:|
| 3 (stock) | 0.0056 | 0.016 | 93.8% | 2,617 / 2,098 |
| 15 (fixed, production since) | 0.0048 | 0.014 | 100% | 2,615 / 2,083 |
| 14 (fixed, f16 Q) | 0.0043 | 0.009 | 98.8% | 2,438 / 1,859 |
| 0 (f16 legs) | ref | | | 2,423 / 1,800 |

### Speculative decoding -- no evidence of a bug

`spec_equiv.py`, greedy with top-20 at every token, MTP SPEC 4 vs `--speculative-config null`: per-token KL mean 2.0e-3,
p99 3e-2 while the texts agree; divergence at near-ties after 23-257 tokens (json never); no spikes after rejections, no
drift. Consistent with verify-width (M=5 vs 1) numerics.

### Prefix-cache resume -- EXACT; the differences were prefill-schedule numerics

First pass (`prefix_equiv.py`: fill the cache with L1 tokens, then L2 = L1 + 300 cached vs cold via cache_salt; hits
verified from /metrics): cached and cold differed in every configuration, top-1 same in 71/72 cases:

| config | KL mean | KL max |
|---|---:|---:|
| fp16 state, fp8 attention (prod-like) | 6.6e-3 | 1.7e-2 |
| fp32 state, fp8 attention | 1.25e-2 | 5.4e-2 |
| fp32 state, f16 attention | 1.18e-2 | 4.3e-2 |
| fp16 state, f16 attention | 7.0e-3 | 1.8e-2 (one top-1 flip) |

Localization (jobs `p9-prefix.sh`, `p9b-determinism.sh`, `p9c-confirm.sh`; fp32 state, f16 attention legs, 8,192-token
steps, 1,616-token blocks):

1. **The engine is bit-deterministic.** The same cold request twice on one boot: KL 0 at all 10 positions (R4D and stock
   GDN paths). Two boots that load the same torch.compile cache: KL 0 (S8 vs E8, D8 vs F8). A boot that runs a (partial)
   fresh compile is the exception: R8 (torch.compile 21 s) vs D8 (4.7 s, same config) KL mean 3.4e-3, max 1.3e-2.
2. **Scheduler rules (vLLM 0.29 `_mamba_block_aligned_split`).** With MTP the mamba cache hit is backed off one block
   (`use_eagle`): hit = (floor(L1/b) - 1)*b. A prefill chunk ends at a block multiple, and never runs past the last
   cacheable position L - L%b - b. So for L1 = k*b or k*b+1 and L2 = L1+1, the cached request runs exactly the cold
   request's own last prefill step, from a snapshot taken at the same boundary.
3. **Schedule-identical resumes are bit-equal.** With a per-case cache salt (fill and cached share it, no case inherits
   another's blocks): 12/12 bit-equal. Without it, 6/12 differed (KL 1.7e-5 to 7.8e-3). Those fill requests had
   themselves resumed from a snapshot an earlier case left, which was produced by a different step schedule. The cases
   that stayed exact had a chain that was new or already evicted (the salted 72k cold runs overflow the 131k pool).
4. **What is left is sensitivity to the step schedule itself.** Cold prefills of one prompt under a different chunk
   size, or a different (equally exact) GDN kernel, differ as much as cached vs cold: R8 vs R16 KL mean 6.1e-3,
   stock S8 vs S16 6.1e-3, stock vs R4D at 8192 6.6e-3 (max 2-2.5e-2, top-1 same 10/10). The chunk comparisons also
   include compile-range effects: `compile_ranges_endpoints` = the chunk size, which is why even a 7,083-token prompt
   (one step under both chunk sizes) differs. Any bit-level perturbation grows to this size by the logits. The fp8 KV
   cache and the per-token fp8 activation quantization turn ulp-level differences into whole-quantum flips, layer after
   layer. Top-1 changes only at near-ties.

Conclusion: no prefix-cache bug. Conv state, SSM state and attention KV restore exactly. A cached resume is one more
valid prefill schedule, and its output is as close to cold as any other schedule's.

Methodology consequence: compare vLLM arms only on boots that load a cached compile. A fresh-compile boot runs the
in-process graph instead of the AOT artifact it saves, and that CAN change numerics (R8 vs D8 above). It does not always
(`p9d-attn-ref.sh`: attn0, the fresh-compile mode-0 lastpos reference, is bit-identical to a cached re-run over all 80
cut points, so the fp8-attention table above stands as measured). `p7-vllm-arm.sh` now redeploys whenever torch.compile
totals more than 10 s (was 30 s; cached boots take 4-5 s, a fresh one ~21 s). Other past arms on fresh-compile boots:
P3 fp32-clamp (PPL over 90 chunks), D vocab48k-spec8 (speed only), eqB2/eqF, pR8/pR16.

### Checked and sound

fp16 GDN state storage (RTNE, fp32 accumulate); fp8 activation quantization (clamped, per-token in both the MXFP4 and
PARO-MXFP4 linears, so a token's GEMM input does not depend on what else is in the step); KV cache writes (saturating);
chunk-scan partial final chunk (invalid tokens get weight 0 on the state path, decay taken at the last valid g, loads
clamped to the last valid row); prefill kernels free of atomics (the MXFP4 split-K "last arriver" reduction sums in a
fixed order and is decode-only); prefix-cache snapshot/restore (above).

## Round 3 -- decode attention, the rest of the stack, performance

### Decode attention kernel -- sound

`decode_check.py` drives the production split-KV decode entry point (`attn_decode_h256_gqa6_fp8kv`) against an fp32
reference on the same e4m3 K/V: q_len 1 / 5 (SPEC-4 verify, causal inside the block) / 10, two sequences of different
length in one launch, contexts off the tile and block grid, the split count a captured graph bakes in (max_ctx 262,144),
forced splits 1/16/128, shuffled physical blocks, NaN-filled scratch. Row error 1.4-2.4e-3 everywhere (the bf16 output
floor), finite, nothing written past the sized scratch. Fully masked split rows store 0 (not 0/0); the combine walks
splits in fixed order; P stays in f16 with the row sum over the same f16 values, so the prefill fp8-P bug cannot occur.

One nit, measured and left: the kernel packs the folded query and the f16 split partials with `v_cvt_pkrtz`
(round-toward-zero). `decode_rtz.py` reproduces the kernel exactly with an RTZ-folded reference (norm ratio 0.9985 ->
0.9997); the residual is the partial store. Net: a 0.03% flatter softmax and a 0.04% output shrink -- harmless next to
fp8 KV noise. Round-to-nearest on those two once-per-kernel conversions would be free if libr4d is rebuilt anyway.

### vLLM 0.29 runs the V2 model runner -- Radiance's V1 hooks are inert

"Using V2 Model Runner" (auto-selected by vLLM). `radiance_draft.py` (dynamic draft depth, verbatim n-gram tails, the
loop break) hooks the V1 proposer (`SpecDecodeBaseProposer._greedy_sample`, `GPUModelRunner.propose_draft_token_ids`),
which V2 never calls. V2 drafts in `v1/worker/gpu/spec_decode`: gumbel_sample on the draft logits at the request
temperature, the pre-temperature logits cached for the rejection sampler's q (consistent: the verifier divides by the
same temperature on load). Even on V1 the controller only engaged for greedy requests.

### Draft distribution (lossless levers)

Sampled decode, `spec_sampled.py` 12 x 600 tokens x code/prose/json, production sampling. The engine is deterministic
with seed 0, so arms reproduce to the digit (eE = dC exactly).

| arm | code | prose | json | overall | acceptance |
|---|---:|---:|---:|---:|---|
| production | 78.9 | 68.9 | 98.7 | 80.4 | .522 / .425 / .719 |
| `RADIANCE_DRAFT_EXACTSET=1` (only the 32 exactly reranked draft candidates eligible; the rest of the int2 row is coarse 2-bit scores that a sampled draft can pick) | 79.8 | 69.4 | 99.3 | 81.0 | .531 / .430 / .725 |
| + `RADIANCE_DRAFT_TOPKP=1` (request top-k/top-p applied to the draft, V2 hook, capturable top-64 mask) | 78.2 | 68.1 | 97.2 | 79.5 | .528 / .428 / .720 |

TOPKP rejected (the draft's own nucleus drops tokens the target's top-20 accepts; the mask costs time). EXACTSET: small,
lossless, greedy unaffected (same text, same tokens/step).

### Decode step profile

rocprofv3's kernel trace dies in an abort loop at container shutdown (its signal handler re-catches SIGABRT forever, the
trace is never flushed); vLLM's torch profiler (`--profiler-config`, `/start_profile`) records kernels inside HIP-graph
replays and writes on stop. `bench/step_profile.py` cuts steps at `_rejection_kernel`. 8k context, greedy, SPEC 4:

| | ms/step | |
|---|---:|---|
| MXFP4 GEMM (256 launches) | 23.5 | ~13.3 GB at ~566 GB/s = 88% of peak |
| fp8 hipBLASLt (MTP drafter 4 x 5 GEMMs + target lm_head) | 7.9 | MTP at 230-430 GB/s, lm_head 2.47 ms |
| rotation + token-quant producers | 1.9 | |
| GDN fused update (48) | 1.15 | latency-bound |
| attention | 0.9 (9.5 at 135k) | ~594 GB/s at long context |
| draft head / sampler / misc | ~1.6 | |
| idle | 4.6 | ~3.4 us HIP-graph dispatch gap x ~1,300 kernels |

### Win: TunableOp table for the skinny fp8 GEMMs (+5%)

hipBLASLt's heuristic gives the MTP drafter's N=5120 GEMMs a 16x128 tile -- 40 workgroups on 64 CUs, no split -- and
the lm_head a 64x64 tile. `fp8_tune.py` runs PyTorch TunableOp over the six shapes at every M the V2 runner uses
(1..12, 16: capture sizes and padding matter -- a table with only M=1/2/5/10 changed nothing) and writes a 78-entry
table; the serve loads it read-only (`PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv`). Tuned solutions are 16x16 tiles. MTP GEMMs per draft pass
1.30 -> 0.85 ms, lm_head 2.47 -> 2.32 ms, step 41.8 -> 39.9 ms.

| | production | + table |
|---|---:|---:|
| sampled code / prose / json | 78.9 / 68.9 / 98.7 | 82.9 / 72.5 / 103.8 |
| sampled overall | 80.4 | **84.5 (+5.1%)**, acceptance identical |
| greedy 8k / 100k | 115.9 / 59.3 | 121.3 / 62.0, identical text |
| prefill | unchanged | unchanged |

The table carries validators (torch, HIP, hipBLASLt, rocBLAS versions, gfx arch): on an image change it is ignored and
the defaults return; re-run `fp8_tune.py`. `PYTORCH_TUNABLEOP_RECORD_UNTUNED` logs keys even on hits -- not evidence.

### fp8 KV cache -- mantissa-bound, and free

Census (`RADIANCE_KV_STATS`, loaded through `local/patch_local_overlay.py` because serve-mxfp4.sh only copies its own
module list): per attention layer K amax 8-20, V amax 4-77, ~1% of values subnormal, none saturated; e4m3 round trip
2.65% RMS per element at scale 1.0 and identically at an amax-calibrated scale -- the 3-bit mantissa sets it, so KV
scale calibration buys nothing. End to end, bf16 KV vs fp8 KV:

| | wiki | code | long-context KL (32k-115k) |
|---|---:|---:|---|
| fp8 KV, R4D_ATTN_FP8=15 (production) | 5.9765 | 2.3040 | 0.0055 vs bf16 KV |
| fp8 KV, f16 legs | 5.9755 | 2.3029 | 0.0038 vs bf16 KV, top-1 79/80 |
| bf16 KV | 5.9650 | 2.3053 | ref |

Inside the noise any benign perturbation produces (KL 0.003-0.006); bf16 KV would halve capacity for nothing measurable.
Production PPL now matches the FLA reference measured in round 1 (5.9757 / 2.3037).

### Also measured (round 3), rejected

- `HIP_FORCE_DEV_KERNARG=1` (kernel arguments in device memory): sampled 84.3 vs 84.5, idle 4.67 vs 4.56 ms -- the ~3.4 us
  gaps are the graph dispatcher's own cost.
- SPEC depth with the table (sampled overall): SPEC 3 80.8, **SPEC 4 84.5**, SPEC 5 83.7 (json +4%, prose -5%, pool
  270,855). Greedy is not comparable across depths: the verify width changes numerics and the greedy text diverges.
- GDN fused update grid cap (FU_MAXWG 32 -> 48) would save ~0.4 ms/step, but the cap is what keeps the grid barrier
  provably co-resident; not worth a hang risk.
- int2 target verify head (`RADIANCE_VERIFY_HEAD`, -2.2 ms/step): exact for sampled requests only when top_k <= RERANK/4,
  i.e. RERANK >= 80 for top_k 20, plus the full int2 head (0.33 GiB of pool); its exactness is the coarse pass's recall.
  Not adopted.

### Production after round 3

`deploy-vllm-qwen38-paro.sh` defaults now add the TunableOp table and `RADIANCE_DRAFT_EXACTSET=1`. Measured on production
itself (12 x 600 sampled, same prompts): code 83.8 / prose 73.0 / json 104.4 = **85.2 t/s (was 80.4, +6.0%)**,
acceptance .531 / .430 / .725, greedy 8k 121.1 t/s with the same text. KV pool unchanged at 273,333 tokens.

Remaining time in an 8k decode step (~40 ms): MXFP4 GEMM 23.5 at ~86% of DRAM peak, fp8 GEMM 5.75 (near the bandwidth
floor now), idle 4.6 (graph dispatch), rotation/quant producers 1.9, GDN update 1.2, attention 0.9, draft head 0.7.

## vLLM 0.30.0 / fork check (2026-09-27)

- Fork: GitHub main +8 commits (lazy GDN default-off because it corrupts multi-turn chat -- we already run it off; TP=1
  long-context defaults; a FAST_DRAFT draft-head fix our 48k-vocab path does not need), Codeberg main + MXFP6-PARO
  (+47% weight bytes, no quality numbers vs a reference yet), FASTBOOT.md (we already boot in ~2 min), KV offload.
  Neither remote moves off vLLM 0.29.0.
- vLLM 0.30.0: nothing gfx1201-specific; breaking changes include removed attention-metadata properties used by custom
  backends; porting the fork's ~107 string anchors + ours = multi-day. Not upgrading.
- #55450 (align-mode mamba states pinned across null gaps): backported as `local/patch_mamba_retire.py`
  (`RADIANCE_LOCAL_MAMBA_RETIRE=1`). `bench/kvusage_probe.py` through 200k / 255k-token prefills: KV usage curves
  identical with and without it at every sample (peaks 74.2% / 93.3% = the attention share; 0 preemptions); with it,
  prefix pairs 12/12 bit-equal and greedy text identical. The leak does not occur in our configuration (the previous
  step's state is freed explicitly; async scheduling off). Not adopted.
- #52228 (adaptive verification for all draft-model speculators): trims the VERIFICATION batch under a budget sized from
  profiled verify-cost curves; drafts are still all produced. Single-stream verification here is weight-bound (M=2..8
  cost the same), so it cannot gain, and it needs varlen-verify support in R4D. Not backported. The lever that fits this
  box is the opposite: stop DRAFTING early on low draft confidence (what the inert V1 controller did).

## Round 4 -- overnight deep dive (2026-09-28, 00:17-04:30 EDT)

User: "keep working on the inference performance of this model in our vLLM fork -- prefill, decode, correctness; we've got
the low-hanging fruit, time to dig in deep." Production was down all night for the rig; restored at the end with the
byte-exact wins only. New scripts/diffs in `vllm-patches/review-20260927/`: kernel gates mr_check.py / split_check.py,
par_kernels_mr.h / _mr3.h / _split.h, `*.20260928.diff` against the vllm-radiance-next originals (radiance_paroquant.hip,
radiance_paroquant.py, radiance_paroquant_mxfp4.py, patch_paroquant_install.py), the new local patches
(patch_aot_envkey.py, patch_gdn_nocopy.py, patch_gdn_warmup_sync.py), jobs p21-p30; `bench/step_seq.py`. Kernel build:
`hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) radiance_paroquant.hip` in
the 0.29 image -> persist/paroquant/radiance_paroquant_kernel-0.29-split2.so.

### What the step looks like now (tB profile, 8k, before this round)

`bench/step_seq.py` (new: one decode step's kernel sequence from a torch-profiler trace) on the round-3 trace: 39.9 ms,
1,298 kernels, ~3.5 us HIP-graph gap after every one of them (4.6 ms idle = 11%). Per GDN layer 14 kernels: triton
add+norm, rotate+quant, in_proj GEMM, the bf16 in_proj_ba GEMV, **a bf16 fill and a copy** (vLLM 0.29's GDN core op:
`core_attn_out.zero_()` and `z_out[:] = z`), GDN fused update, gated-norm+rotate+quant, out_proj GEMM, add+norm,
rotate+quant, gate_up GEMM, silu-mul+rotate+quant, down GEMM. The MTP drafter is 4 x ~1.4 ms (5 fp8 GEMMs at 470-550
GB/s + int2 head 165 us + topk 34 us + rerank 15 us), the target lm_head 2.29 ms (556 GB/s). The MXFP4 GEMMs run at
88-93% of DRAM bandwidth on the big shapes, 68% on the 6144->5120 out/o_proj.

Prefill (16k, 2 chunks): MXFP4 A-tiled GEMM 65% (~59% of the fp8 WMMA peak), per-token rotate/quant producers **17.5%**
(silu-mul->down alone 7.6%, memory-bound), attention 7.4%, GDN 4.6%, the GDN z copy 0.65%.

### Win 1 (prefill): conflict-free multi-row rotation producer -- byte-exact, ~+1% prefill

The Givens chain is LDS-bank-conflict bound (random pair records: 2 pairs per lane per round, 8 rounds), not bandwidth
bound -- a first multi-row version that shared records across rows was SLOWER (register pressure). The fix is an
"ownership" layout: `build_rot3` (host, once per linear at load) re-labels each 128-group's 8 rounds so that lane l owns
slots {l, 32+l, 64+l, 96+l} and every round is 4 perfect matchings -> no bank conflicts; tables R3/INIT live next to the
records (`par_kernels_mr3.h`, `pq_rotate_tokquant3_mr`, `pq_ew_rot_tok3_mr`). Gate `mr_check.py`: codes/scales/HS
byte-identical to the stock kernels over M 65..8192, K/N 5120-17408, P 1-3, tiled/row-major, all three ew modes.
M=8192: norm-fed rotate 681 -> 432 us (P1), 1319 -> 855 (P2), 807 -> 525 (K=6144); the ew variants ~stock (ew0 -8%
without the unread HS write). Serve switch `RADIANCE_PQM_ROT3=1` (tables looked up by the rec tensor's address inside
the opaque ops -> no graph change; a shape guard rejects a mismatched table).

| e2e, prod settings (tok/s) | prefill 8k (2 prompts) | 43k (2) | 112k / 120k | greedy 8k / 100k sha |
|---|---:|---:|---:|---|
| base (round-3 prod) | 3026 / 2948 / 2926 | 2588 / 2586 | 2140 / 2076 | 55222d73 / 86cd9c47 |
| rot3 incl. the ew producers + SKIP_HS (p25 hs) | 3090 / 2988 | 2615 | 2160 / 2097 | identical |
| **production: rot3 norm-fed only** (p28 fin2) | 3089 / 2981 / 2959 | 2604 / 2613 | 2165 / 2091 | identical |

About +1% at every depth. The ew variants and SKIP_HS (+0.3% more) are left off in production -- see the fault below.

### Win 2 (decode): split-token per-token producers -- byte-exact, sampled +1.2%

The decode-band producers ran ONE workgroup per token: at M=5 that is 5 CUs of 64, and the silu-mul->down producer
(N=17408) walks 5 rotation chains per wave in series (12.3 us, 64x per step). `par_kernels_split.h`: S = ceil(G/16)
workgroups per token each rotate 16 groups (the stock chain, unchanged), park bf16 in a scratch row, fold their fp32 amax
into a per-token atomicMax, and the LAST-arriving workgroup (atomic counter, threadfence) encodes the row and resets the
slot -- no inter-workgroup wait, so no co-residency assumption. A max is order-free, so codes/scales/HS are identical.
Gate `split_check.py`: ALL EXACT over M 1..64, every shape/mode, repeated launches, a 16384-stride gate row; isolated
M=5: silu-mul 21.4 -> 8.9 us, rotate 7.9 -> 6.4, gated norm 8.8 -> 6.7. `RADIANCE_PQM_SPLIT=1` (M <= 16).

| sampled 12x600 (seed 0) | code | prose | json | overall | acceptance |
|---|---:|---:|---:|---:|---|
| prod (round 3) | 83.8 | 73.0 | 104.4 | 85.2 | .531 / .430 / .725 |
| rot3 + SKIP_HS | 83.7 | 72.9 | 104.4 | 85.1 | identical |
| + split producers (p25 sp, quiet host) | **84.7** | **73.8** | **105.7** | **86.2 (+1.2%)** | identical |
| production after round 4 (on :1246, host busy*) | 84.3 | 73.4 | 105.0 | 85.7 (+0.6%) | identical |

\* measured at 03:25 with mario-bot (torch, 225% CPU), a Plex loudness scan and bazarr running (load avg 6-7); the
fin2 rig run at 03:12 under the same load gave 81.0 / 72.1 / 104.6 with identical acceptance -- CPU contention moves
sampled throughput by a few percent, the byte-exactness does not move.

Greedy 8k/100k 123.5 / 62.6 (was 122.3 / 62.0), same text.

### Finding: rotation "stream 1" has been dead code since it was written

`install_stream` sets each decoder layer's `forward` to the fused add+norm+rotate path -- and `radiance_arnq.install`
(the fp8-stream contract, `RADIANCE_FP8_STREAM`, serve-mxfp4.sh default 1) runs AFTER it at the end of
`radiance_gdnmerge.merge_model` and overwrites every layer's `forward` again, although on the paroquant model it
installs nothing ("fp8 stream installed: 0 mid epilogues ..."). (My first theory -- dynamo ignores instance-level forwards
-- was wrong; a synthetic test shows it honours them, and a comptime probe proved the layer forward was never traced.)
With `RADIANCE_FP8_STREAM=0` stream 1 goes live: NOT byte-exact (its fused norm reduces in a different order), prefill
-2%, and ~5% SLOWER per decode step (38.7 vs 36.8 ms at 12x600) even though the fused kernel alone is no slower than the
rotate it replaces (8.5 vs 8 us) -- not investigated further. Its greedy 100k run "accepted" 0.85 vs 0.44 only because
the rounding change sent greedy decoding into listing an arithmetic pattern; judge MTP changes on sampled prompts only.
A split variant of the fused kernel exists (`pq_add_rms_rot_tok_split`, gated exact) if anyone revisits it. Not adopted.

### Finding: vLLM 0.29's AOT compile cache ignores the environment

`torch_aot_compile/{hash}` keys on the vLLM config + the top-level forward; on load it re-checks the SOURCE of every
traced file but never the environment. Every switch of ours that changes the traced graph from env (rotation streams,
fp8 stream, GDN no-copy, ...) therefore loads the stale artifact when sources are unchanged -- A/Bs of such switches on a
warm cache measured the old graph. `local/patch_aot_envkey.py` (`RADIANCE_LOCAL_AOT_ENVKEY=1`) makes the dir
`{hash}-{sha(RADIANCE_* env)[:12]}` and seeds a new dir with the base dir's inductor cache (content-addressed), so
unchanged pieces keep their kernels and numerics (greedy shas identical across the switch).

### Tried, not adopted

- GDN glue removal (`local/patch_gdn_nocopy.py`): the `zero_()` skip (gated on EMPTY_OUT + the all-R4D path) is exact
  (greedy 8k/100k identical, +0.6% greedy); reading z in place from the projection instead of copying it is NOT exact at
  100k (cause not isolated).
- **Boot-time GPU memory fault** (open, mitigated). Between 02:08 and 02:41, boots of the configs with the no-copy /
  EMPTY_OUT switches (+ rot3 ew producers) died in the KV-profile forward with a memory fault in
  `pq_ew_rot_tok3_mr<2,...>` (the conflict-free GDN gated-norm producer) -- 6 of 8 boots, once on a config+artifact that
  had passed half an hour earlier, i.e. nondeterministic; never on configs without those switches. The kernel gate is
  clean at every shape; its addressing is data-independent (the e4m3 encoder is arithmetic); `RADIANCE_PQM_ROT3_DEBUG`
  logging of every rot3 launch (p29) shows the right table for every rec and consistent sizes. The faulting addresses were
  page-aligned buffer STARTS (0x...a00000), i.e. memory unmapped under a running kernel. Mechanism found in vLLM 0.29:
  the GDN `_warmup_prefill_kernels` runs INSIDE the profile forward (once per GDN layer) and ends with
  `torch.accelerator.empty_cache()` while the host is far ahead of the GPU -- blocks of earlier layers that are already
  free in the caching allocator but still read by queued kernels go back to the driver. Could not be confirmed by A/B:
  since 02:55 the same configs booted 16/16 clean (3 of them with `AMD_SERIALIZE_KERNEL=3`, 3 unpatched controls, 5 with
  the fix), after a shape guard on the table lookup (never fired) and a wipe of the experiment AOT dirs.
  `local/patch_gdn_warmup_sync.py` (device sync before that empty_cache, boot-only) is in production as hardening; the
  rot3 ew producers (`RADIANCE_PQM_ROT3_EW`, ~+0.3% prefill with SKIP_HS) and the zero_() skip (~+0.5% decode) stay OFF
  until the fault is understood.
- Stream 1 (above). MXFP4 for the MTP layer (would halve drafter bytes, ~4% if acceptance held) and a faster draft top-k
  (136 us/step) -- not started.

### Production after round 4

`deploy-vllm-qwen38-paro.sh` defaults add `R4_ENV`: `RADIANCE_LOCAL_AOT_ENVKEY=1`,
`RADIANCE_LOCAL_PQ_SO=radiance_paroquant_kernel-0.29-split2.so`, `RADIANCE_PQM_ROT3=1`, `RADIANCE_PQM_SPLIT=1`,
`RADIANCE_LOCAL_GDN_WARMUP_SYNC=1`. Measured on production (04:10-04:15, fresh-compile boot and a cached reboot):
greedy 8k / 100k sha 55222d73 / 86cd9c47 (identical), decode 123.1-123.9 / 62.6 t/s (was 121.1 / 62.0), prefill 3082-3089
@8k, 2162-2166 @112k (was ~3026 / 2140); sampled (03:25, R4 minus the warm-up sync, host busy) 85.7 t/s with acceptance
.531 / .430 / .725 (identical); KV pool 273,333; no memory faults in 8 production/final-config boots. Rollback: drop
`$R4_ENV` from EXTRA_ENV -- with the switches unset the plugin runs the stock producers and the stock .so.

Decode step after round 4 (p31, torch profiler, greedy, prefix-cached prompt):

| ms/step | 8k before (tB) | 8k after | 100k after |
|---|---:|---:|---:|
| wall | 39.91 | **39.44** | 48.49 |
| MXFP4 GEMM | 23.49 | 23.55 | 23.57 |
| fp8 hipBLASLt (MTP drafter + lm_head) | 6.10 | 6.13 | 6.15 |
| rotation/quant producers | 1.90 | **1.31** | 1.37 |
| GDN update | 1.16 | 1.14 | 1.18 |
| attention | 0.90 | 0.89 | **9.53** |
| draft head + sampler + copies + norms | 1.8 | 1.8 | 1.8 |
| idle (graph dispatch gaps, ~1,300 kernels) | 4.56 | 4.63 | 4.86 |

What is left is bandwidth (MXFP4 GEMM at 88-93% on the big shapes, 68% on out/o_proj), the dispatch gaps, the MTP
drafter's 4 x 1.4 ms, and at long context the decode attention.

## Round 5 -- day session (2026-09-28, 09:40-13:50 EDT): MXFP4 drafter, prefill attention, the second compile cache

User: pick the most impactful decode / prefill options and start on them (multi-day work is fine; the 250 W cap stays).
Picked: decode = the MTP drafter in MXFP4 (4 x ~1.4 ms of fp8 GEMMs per step), prefill = long-context attention (30% of
a 100k prefill). New files in `vllm-patches/review-20260927/`: `patch_rx9z.diff` (libr4d rx9y -> rx9z), `attn_bench.py`,
`dec_bench.py`, `mxdec_bench.py`, `graph_gap.py`, the drafter GPTQ tools `mtp_gptq.py` / `mtp_hc_drive.py` /
`trace_accept.py` (also in custom-quant/mtp-gptq), `patch_paroquant_install.py.20260928.diff` and `patch_aot_envkey.py`
(both updated), jobs p32-p42.

### Prefill attention: libr4d rx9z -- bit-exact, kernel -4.5%, prefill +1.4-1.7% at 100k+

`attn_bench.py` (8k-query chunk at depth N, fp8 KV, mode 15): 157 / 169 / 164 TFLOPS at 8k / 64k / 128k, ~50% of the
fp8 WMMA peak at the ~2.55 GHz this kernel holds under the cap. Ablation at 64k (73.2 ms; bits in `abl_prefill.hip`):
dropping the QK WMMAs saves 16.8 ms, the PV WMMAs 16.0, the tile staging 11.0, exp 2.5, the PVDEN row sum 2.9 -- i.e.
~37-43 ms of WMMA work and the rest VALU/staging that does NOT overlap with it (RDNA4 issues them from the same wave;
splitting the QK chain into two accumulators changes nothing, so it is issue-bound, not latency-bound). Tile 64/80/96
for the 8-bit legs: neutral; 12 warps: -8%. rocprofv3 counters are mostly zero on gfx1201 (SQ_INSTS_*).

What moved it: the option-bit re-sweep for mode 15 found prefetch 8 (+2.3%, the 16-bit legs' PF4 was never re-tuned
for the 8-bit ones), and the PVDEN row sum now runs as two `v_dot4_f32_fp8_fp8` against 1.0 after the PV WMMAs (2 VALU
instead of 4 converts + 7 adds; partial sums of <= 8 e4m3 values are exact in fp32, so the result is bit-identical).
Full builds bit-exact vs rx9y at 8k/64k/128k; kernel 73.2 -> 69.9 ms @64k, 156.3 -> 149.3 @128k.

| e2e (p32/p35, rig, prod env) | prefill 8k | 43k | 100k (119,939 tok) | 112k | greedy shas |
|---|---:|---:|---:|---:|---|
| rx9y (round 4) | 3000 | 2618 | 2097 | 2166 | 55222d73 / 86cd9c47 |
| **rx9z** | 3000 | 2644 | **2133** | **2196** | identical |

### Decode: the MTP drafter in MXFP4 -- sampled +3.1%, distribution exact

`RADIANCE_MTP_MXFP4=1` (`local/patch_paroquant_install.py`): the drafter's five linears (fp8 per-channel Quark weights)
are requantized at load to MXFP4 -- per 32-block e8m0 exponent chosen by MSE between the OCP floor and floor+1 (equal to
the brute-force per-block optimum), the body's fragment permutation applied -- and served by the body's W4A8 kernel
through ONE opaque custom op `radiance::mtp_mxfp4_linear` (per-token fp8 quant + GEMM). Load report: weight rel. error
vs fp8 10.8-11.6% (the MXFP4 norm), drafter 405 -> 213 MiB. A spec-decoding drafter only proposes; the verifier's
accept/reject keeps the output distribution exact, so the question is acceptance vs step time.

| sampled 12x600 (p33/p35/p36) | code | prose | json | overall | acceptance |
|---|---:|---:|---:|---:|---|
| fp8 drafter (round-4 prod) | 84.4 | 73.5 | 105.4 | 85.9 | .531 / .430 / .725 |
| MXFP4 drafter, qkv/o kept fp8 | 88.0 | 74.4 | 104.6 | 87.3 | .531 / .412 / .682 |
| **MXFP4 drafter (+ rx9z)** | **88.2** | **76.1** | **106.5** | **88.6 (+3.1%)** | .521 / .415 / .682 |
| same, SPEC 5 | 86.6 | 71.9 | 112.2 | 87.3 | .451 / .339 / .644 |

Step time -5.2% (~1.9 ms); acceptance -2..-6% (json most). Greedy text changes (different drafts -> different verify
windows -> schedule-dependent rounding); greedy 8k 122.1 -> 130.6 t/s, 100k 62.4 -> 63.2. SPEC 4 stays.

**GPTQ for the drafter (p39/p41).** Calibration boot (`RADIANCE_MTP_HCOLLECT`, eager, fp8 drafter): 48 windows x 4096
tokens of the omp-trace calibration split (traces.npz `calib`, the body's v2 set) with sampled continuations; every
drafter linear accumulates H = sum x^T x, split into draft-pass rows (25k) and drafter-prefill rows (195k). `mtp_gptq.py`
runs the body's v2 `gptq_mxfp4` (custom-quant/paro-mxfp4-v2) on H = (Hd/nd + Hp/np)/2. Layer-output proxy error
tr(dW H dW^T) vs the load-time RTN: 0.20-0.24x (draft rows) / 0.26-0.30x (prefill rows) on all five linears (weight
rel. error goes UP, 11 -> 14-17%, as GPTQ trades it for output error). The codes ship as a file
(`RADIANCE_MTP_MXFP4_FILE`, 215 MiB, with a sha1 fingerprint of the fp8 weights they were fitted to -- a different
checkpoint falls back to RTN, logged).

| drafter | bench 12x600 (code / prose / json) | bench accept | held-out omp windows: accept | decode t/s |
|---|---:|---|---:|---:|
| fp8 (round 4) | 85.9 (84.4 / 73.5 / 105.4) | .531 / .430 / .725 | .574 | 75.3 |
| MXFP4 RTN | 88.6 (88.2 / 76.1 / 106.5) | .521 / .415 / .682 | .547 | 76.0 |
| **MXFP4 GPTQ** | **88.9** (88.1 / 75.6 / 109.1) | .518 / .411 / .704 | **.559** | **77.9** |

The held-out windows (`trace_accept.py`: traces.npz `eval8k`, whole session files never used for calibration, 24 x
6144-token prompts + 256 sampled tokens, one at a time) are the user's own workload, and there RTN's acceptance loss is
larger than on the bench (-4.8%) and eats most of its speed gain (+0.9%); GPTQ recovers ~45% of it (+3.4% vs fp8,
+2.4% vs RTN). On the synthetic bench it is neutral (json +3% acceptance, code/prose -0.5%). Production uses GPTQ.

### Finding: vLLM 0.29 has a SECOND compile cache that ignores the environment

`torch_compile_cache/<10-char hash>/rank_R_D/{backbone,eagle_head}` (the piecewise backend's compiled pieces, keyed on
[vLLM env, config, traced code, compiler] and LOADED BY PIECE INDEX) is separate from the AOT dir round 4 fixed. A graph
variant chosen by our env (MXFP4 vs fp8 drafter, GDN no-copy, rotation streams ...) with unchanged sources got another
variant's pieces: boot died in inductor's `copy_misaligned_inputs: Expected tensors only, but got int` (every mixed
fp8/MXFP4 drafter config, and the rx9z+MXFP4 combo 3/3). It may also have contributed to the night's "boot-time GPU
memory fault" (same switches, same timeline, never reproduced once the dirs were wiped) -- but see x2 below for a more
specific suspect. `patch_aot_envkey.py` now also appends the sorted RADIANCE_* env to the piecewise hash factors. After
the fix every config booted (r5: 3/3, x1/x2 below).

### Launch gap: no free knob

`graph_gap.py`: a captured graph of 1000 tiny dependent kernels costs 2.3 us/kernel, identical under every HIP knob tried
(DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING, DEBUG_HIP_GRAPH_BATCH_SIZE, DEBUG_HIP_FORCE_GRAPH_QUEUES, ROC_SYSTEM_SCOPE_SIGNAL=0,
GPU_FLUSH_ON_EXECUTION=0, AMD_DIRECT_DISPATCH=0, DEBUG_CLR_MAX_BATCH_SIZE, HIP_FORCE_DEV_KERNARG). Only fewer kernels
help. A draft pass is ~55 kernels: ~22 model, ~23 draft-head/sampling glue, ~10 runner bookkeeping.

### Round-4 extras re-tested after the cache fix (p37): adopted except the z-in-place read

On the round-5 candidate (rx9z + MXFP4 drafter), each arm cold-compiled into its own env-keyed dirs and booted 3x:

| p37 | greedy 8k / 100k sha | 8k / 100k t/s | prefill 43k | sampled 12x600 | acceptance | boots |
|---|---|---:|---:|---:|---|---|
| r5 (p35) | 57e7b168 / da44c0cc | 130.6 / 63.2 | 2631 | 88.57 | .521 / .415 / .682 | 3/3 |
| **x1** = + ROT3_EW + SKIP_HS + GDN zero_() skip (NOCOPY=1, NOCOPY_Z=0, EMPTY_OUT) | **identical** | 130.2 / 63.4 | 2645 | **88.80** | identical | 3/3, 0 faults |
| x2 = x1 + z read in place (NOCOPY_Z=1) | 8e0cf6b7 / b2188800 | 127.8 / 66.5 | 2642 | -- | -- | 3/3, 0 faults |

x1 is exact and a little faster (+0.3% sampled, +0.5% prefill at 43k) -> in production. x2 changes the text already at
8k (and the 16-token prefill probes), so the gate VALUES the gated norm reads differ -- a copy cannot change a bf16
value. Not a stride bug: every producer path takes the row stride (`Y + m*ys` in the rot3/split/stock kernels; the
split gate tested a 16384-stride row) and the GDN core only reads the projection (its conv writes separate q/k/v).
What IS different is lifetime: the z view now crosses the GDN core op and is read after it, with the projection's
storage referenced only through that view (vLLM keeps graph outputs as weak refs, `CUDAGraphWrapper` entry.output) --
the likeliest way for the gated norm to see recycled memory. Not established (it would take a cudagraph-off A/B). It
does re-read the night's faults: all of them were in `pq_ew_rot_tok3_mr<2,...>` -- mode 2 is the gated norm, i.e. the
z consumer -- on configs where NOCOPY implied NOCOPY_Z=1, at page-aligned buffer starts. So the z-in-place read, not
(only) the stale compile pieces, is the prime suspect for those faults. Stays off; `RADIANCE_LOCAL_GDN_NOCOPY_Z=0` must be explicit (it defaults to 1 under NOCOPY).

### Decode attention at long context is at the bandwidth roof (p38) -- not a lever

p31 put decode attention at 9.5 of 48.5 ms/step at 100k, and the split law gives TP=1 only 32 splits x 4 kv heads = 128
workgroups of 1-2 waves (it was tuned at TP=2), with 10-30 VGPRs spilled -- it looked latency-bound. Measured in
isolation (`dec_bench.py`: the production entry point, graph-baked max_ctx 262,144, DRAM-fed):

| us/call (GB/s of KV) | splits 16 | **32 (law)** | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| 100k verify (q 5) | 508 (403) | **352 (582)** | 360 | 383 | 423 |
| 100k draft (q 1) | 566 | **359 (571)** | 347 (590) | 358 | 386 |
| 200k verify | 1040 | **697 (588)** | 693 | 732 | 788 |
| 8k verify | 58 | **48 (352)** | 51 | 59 | 71 |

The kernel streams KV at 90-92% of the 640 GB/s peak at depth, and 32 splits is the argmin (more splits only add
partial traffic). Per step at 112k that is ~7.9 ms of pure KV streaming (16 verify layers + 4 draft passes); the
profile's 9.5 includes the in-graph cache interference and the combines. The only byte reduction left there is the
drafter's own attention (4 x 0.36 ms at 100k): a windowed drafter attention would stay distribution-exact but would
cost acceptance exactly where long-range copying matters -- not tried.

### Decode GEMMs are at the roof too (mxdec_bench.py)

The p31 profile had the 6144->5120 out_proj/o_proj at 68% of DRAM bandwidth. In isolation (production knobs, DRAM-fed,
M = 1/5) the W4A8 decode kernel runs out/o at 531 GB/s (83%, 31.5 us), down 617, qkvz 629, gate_up 625 GB/s, the MXFP4
drafter fc 558. Forced split-K 1/2/4/8 (8 = a new instantiation): no shape improves by more than 1-4% (fc 50.0 -> 47.6 us
at ks 2), out/o not at all. So the in-graph shortfall is dispatch/ramp context (a 31 us kernel between dependent
neighbours), not the kernel -- the same wall as the launch gap.

### Where decode stands

Every big item is now at or near its bandwidth roof in isolation: MXFP4 GEMMs 83-99%, decode attention 90-92%, lm_head
fp8 ~87%. What is left is structural: ~1,250 dependent kernels x ~3.5 us of graph dispatch (11%), drafter acceptance
(fp8 .574 vs GPTQ .559 on the user's traces), and the drafter's own long-context attention (4 x 0.36 ms at 100k).

### Production after round 5

`deploy-vllm-qwen38-paro.sh` defaults: `R4D_KEY=b9e42ab-rx9z`; `R4_ENV` += `RADIANCE_MTP_MXFP4=1`,
`RADIANCE_MTP_MXFP4_FILE=/cache/mtp_gptq/qwen38-paro-v2-mtp-gptq-a05.pt` (in persist/cache-029-paro-tp1s; override with
MTP_GPTQ=), `RADIANCE_PQM_ROT3_EW=1`, `RADIANCE_PQM_SKIP_HS=1`, `RADIANCE_LOCAL_GDN_NOCOPY=1`,
`RADIANCE_LOCAL_GDN_NOCOPY_Z=0`, `RADIANCE_GDN_EMPTY_OUT=1` (also exported on the host for serve-mxfp4.sh). Measured on
production (p42, cached-compile boot after the compiling one, host load ~5):

| production | round 4 | **round 5** |
|---|---:|---:|
| sampled 12x600 (code / prose / json) | 85.9 (84.4 / 73.5 / 105.4) | **89.1** (88.4 / 75.9 / 108.7) |
| acceptance | .531 / .430 / .725 | .520 / .412 / .701 |
| held-out omp windows: decode t/s (accept) | 75.3 (.574) | **77.9** (.565) |
| greedy 8k / 100k t/s | 123.5 / 62.6 | **130.0 / 65.7** |
| prefill 8k / 43k / 100k (119,939 tok) | 2981 / 2613 / 2097 | 3001 / 2639 / **2133** |
| KV pool | 273,333 | 273,333 |

Greedy shas are new (33fab966 / 98b2b349 @8k/100k: the drafter changed, so the verify windows did); the 16-token
prefill probes match the rx9z runs (37f8ed2a / 3a67ef09 / 11205caa). 0 memory faults over the day's ~20 boots with
the adopted switches. Rollback: `R4D_KEY=b9e42ab-rx9y` and/or drop switches from R4_ENV (each is independent); the round-4 launcher is
commit 28e593c.

### Open / next

- Launch gap (11% of the step, ~1,250 kernels): the only decode lever of size left. Candidates, each ~0.3-1%: the GDN z
  copy (48/step; needs the lifetime issue above understood), the draft-head/sampling glue (~23 kernels x 4 passes),
  the add+norm -> rotate pairs (stream 1 was slower for an uninvestigated reason).
- Prefill: the MXFP4 A-tiled GEMM (65% of a 16k prefill at ~59% of the fp8 peak) -- an ablation like the attention one
  above would show how much is dequant/staging VALU vs WMMA.
- Drafter: fp8 acceptance is still .574 vs .559 on the traces; a larger calibration set, alpha toward the draft rows,
  or the pod-side mtp_refit (self-distillation to the quantized body) could close the rest.

## Round 5b -- afternoon (2026-09-28): prefill producers without output selects ("rot4")

Prefill profile of the round-5 production (p43, torch profiler, one cold 16k and one cold 100k prompt):

| share of kernel time | 16k (5.44 s) | 100k (43.6 s) |
|---|---:|---:|
| MXFP4 GEMM (A-tiled 61% / 49%) | 69.3% | 52.6% |
| rotation/quant producers | 14.9% | 11.4% |
| attention | 7.3% | 29.6% |
| GDN | 4.5% | 3.5% |
| norms / copies / rest | 4.0% | 2.9% |

The GEMM is the fork's A-tiled kernel, already ablated upstream (WMMA-only runs at 55% of its time; "the remaining
~40% over the WMMA floor is structural at this tile"). The MXFP4 drafter's prefill (5 linears per chunk on the folded
kernel, 7.4 ms each) is 1.1-1.4% of prefill -- visible, but under the p42 noise. The producers were the tractable part:
at M=8192 they move their bytes at only ~300 GB/s.

- Prefetching every group's inputs before the rotation chains (bit-exact): no gain, ew0 slower -- not latency-bound.
- Ablation: dropping the rot3 core's per-lane output selects (`pq_sel4`, 3 v_cndmask per LDS write, 12 per row per
  round; wrong output, same traffic) runs the producers 20-36% faster -> they are VALU-bound on the selects.

**rot4** (`par_kernels_mr4.h`, `build_rot4`, launch flag v4): the ownership permutation moves from the writes to the
reads. Every round lane l writes its four outputs to the fixed slots {l, 32+l, 64+l, 96+l} (bank l: conflict-free, no
selects). The round's 64 pairs are then edges between the banks holding their two elements -- a 4-regular multigraph on
32 banks, which always splits into two 2-factors (Petersen; built by orienting along Euler circuits and splitting the
out/in bipartite graph with rot3's alternating-circuit helper). In a 2-factor every bank is exactly once a tail and once
a head, so "read the tails" and "read the heads" are each conflict-free; lane l takes the edge of 2-factor A (B) whose
tail is bank l as its pair a (b). A pair read head-first swaps roles, which the table absorbs by negating the stored
sine: fmaf(c, xj, (-s)·xi) and fmaf(c, xi, -(-s)·xj) are the stock expressions for (j', i') term for term, so every
rotated value is bit-identical. Layer 0 is a fixed write too (channel 4l+k -> slot 32k+l, no INIT scatter); only the
final gather back to channel order reads through a table (once per group). Same table shapes as rot3.

`mr4_check.py` (the mr_check gate + rot4): ALL EXACT vs the stock kernels and rot3 over M 65..1000, K/N 5120-17408, P
1-3, tiled/row-major, all three ew modes. M=8192 (us):

| producer | stock | rot3 (prod) | **rot4** |
|---|---:|---:|---:|
| norm-fed rotate K=5120 P=1 / P=2 | 677 / 1325 | 418 / 867 | **298 / 627** |
| norm-fed rotate K=6144 | 807 | 518 | **357** |
| silu-mul -> down (ew0, N=17408) | 2655 | 2500 | **2222** |
| attention gate (ew1, N=6144) | 845 | 833 | **548** |
| GDN gated norm (ew2, N=6144) | 899 | 856 | **611** |
| (plain rotate K=17408, R=4 -- not a production shape) | 2477 | 1419 | 1982 |

Serve switch `RADIANCE_PQM_ROT4=1` with kernel .so `radiance_paroquant_kernel-0.29-rot4.so` (radiance_paroquant_mxfp4.py
builds rot4 tables and launches v4; without the switch nothing changes).

**End to end (p44/p45/p46, rig, round-5 env).** Prefill, same prompts, two cold runs each: 8k 2995 -> **3130** (+4.5%),
43k 2635 -> **2730** (+3.7%), 100k-class (112-124k) 2196/2111 -> **2264/2171** (+3.1/2.8%). Decode unchanged (decode-band
producers are the split kernels, untouched).

Exactness end to end took a detour worth recording. The rot4 arm's 100k greedy text (c2b0a3bf) differed from the
production run's (98b2b349) while 8k matched. Bisect: the rot4 .so with ROT4 OFF gave c2b0a3bf too; rot4 limited to
K=5120 / 6144 / 17408 gave c2b0a3bf; `kcmp.py` + `kdiff.py` (every producer entry point of the serve, real records,
M 5..8192) found the split2 and rot4 builds byte-identical; and the unchanged production env plus a dummy RADIANCE_
variable (p46: a fresh compile artifact, same kernels) gave **c2b0a3bf** as well. So rot4 is exact (every arm on a
fresh artifact agrees at 8k and 100k), and the production artifact compiled at 13:21 rounds differently at 100k from
every artifact compiled since -- compile-to-compile variance in the inductor-generated code, which the env-keyed cache
dirs expose whenever a switch changes the key. Consequences for method: compare greedy shas only between arms compiled
in the same session (or on the same artifact); a sha change after an env change is not by itself evidence against a
byte-exact kernel. (`RADIANCE_PQM_DUMP_REC=<dir>` dumps every linear's rotation records at load; `RADIANCE_PQM_ROT4_K`
limits rot4 to the listed K -- both debug switches.)

**Production after round 5b** (p42b, cached boot, 19:06): prefill 3243 @8k (first request; round 5: 3108), 2736 @43k
(2639), 2191 @120k (2133); sampled 89.1 (88.3 / 75.8 / 109.4, accept .518 / .411 / .704); held-out omp windows 78.3 t/s
(accept .559); greedy 8k 133.9 t/s sha 33fab966; 0 memory faults. The 100k greedy text is now c2b0a3bf (the
fresh-compile text above) and runs at 62.1 t/s because that text's acceptance is lower (.410 vs .448 on 98b2b349) --
greedy decode speed at depth is text-dependent; the kernels are the same.

## Round 6 (2026-09-28 evening): is the prefill GEMM at its limit?

User: "do we believe the fork author's math on the main matmul kernel?" (upstream: the A-tiled kernel's WMMA-only
ablation runs at 55% of its time, "the remaining ~40% over the WMMA floor is structural at this tile"). Measured on
this card at the 250 W cap (p47; `gemm-work/wmma_peak.hip`, `gemm_bench.py`: M=8192, DRAM-fed, sclk and power sampled
from sysfs during each timed loop):

| | TF/s | sclk | power | TF/s per GHz (x 1/132.1 = share of WMMA rate) |
|---|---:|---:|---:|---:|
| pure WMMA, 16 accumulators/wave, no memory | **375** | 2844 MHz | 247 W | 132.1 (100%) |
| MXFP4 A-tiled: gate_up 34816x5120 | 215 | 2037 | 249 | 80% |
| qkvz 16384x5120 / attn qkv 14336x5120 | 232 / 230 | 2168 / 2149 | 250 | 81% / 81% |
| out/o 5120x6144 / down 5120x17408 | 230 / 233 | 2157 / 2140 | 250 | 81% / 82% |
| hipBLASLt fp8 x fp8 (same shapes) | 161-198 | 1698-1843 | 250 | 71-83% |
| bf16 torch.mm | 103-107 | ~2030 | 250 | -- |

Three things follow. (1) The nominal "~59% of peak" understates the kernel: every shape runs at 80-82% of the WMMA
rate AT THE CLOCK IT GETS, the same fraction AMD's own tuned fp8 library reaches -- two independent implementations
landing on the same issue efficiency is good evidence that ~80% is the practical ceiling for a WMMA GEMM with its
operand traffic on RDNA4, i.e. the upstream "structural" claim holds. (2) The binding limit in our deployment is POWER:
pure WMMA sits at 2.84 GHz at the cap, the GEMM drops to ~2.1 GHz because its data movement (A from L2 per wave pair,
W dequant into LDS, fragment reads, C writes) costs energy -- 1.09 pJ/FLOP vs 0.67 for WMMA alone. hipBLASLt is slower
in absolute terms precisely because it moves 2x the weight bytes (8-bit) and clocks down to ~1.75 GHz. (3) So the
remaining headroom is energy per FLOP, not instruction scheduling: shaving the ~19% non-overlapped issue time would
mostly be eaten by a lower clock at the same 250 W. Only moving fewer bytes per FLOP would help, and the tile is already
at the accumulator (TM=TN=4 x 8 VGPRs = 128) and LDS limits upstream measured. Verdict: believe it; the ceiling
under the cap is ~energy-bound, the kernel is within ~10% of what a better data-movement design could plausibly gain,
and that is a multi-week project with an uncertain payoff. Not pursued.

### Decode fusion: where the step goes after round 5b (p48)

Torch-profiler census of one 8k decode step on the production config (`jobs/p48`, `bench/step_seq.py`): 37.4 ms wall,
32.9 ms GPU busy, **4.46 ms idle across 1,270 kernels** (~3.5 us HIP-graph dispatch gap each). MXFP4 GEMMs 25.2 ms
(~95% of DRAM bandwidth), lm_head 2.3 ms, everything else is small glue:

| phase | kernels | busy | gaps |
|---|---:|---:|---:|
| step-start bookkeeping (V2 runner) | 24 | 0.06 ms | 0.08 ms |
| 4 drafter passes (69 kernels each: 26 model, ~22 draft head + sampling, ~10 runner) | 276 | 3.10 | 1.02 |
| verify pass: GDN layer 13 kernels x 48, attention layer 22 x 16 | 964 | 27.39 | 3.32 |
| lm_head + rejection | 6 | 2.32 | 0.02 |

Fusion targets found: (1) the (inductor add+RMSNorm, our rotate+quant) pair at every norm site -- 128 per step; (2) q/k
RMSNorm + mRoPE + gate split as 8 inductor kernels per attention call (20 per step incl. the MTP layer); (3) the draft
head's 15 launches per drafter pass; (4) the GDN z copy (48); (5) the bf16 in_proj_ba GEMV (48, not attempted).

### Decode fusion results (p49-p55; production kept down across the batches)

Precise step times are torch-profiler medians over 32-34 decode steps at 8k (`jobs/p54`); throughput and acceptance are
the sampled bench (12x600, deterministic seeds) and the held-out omp windows (`trace_accept.py`, 24 windows, and in p55
cut at 2048/4096/6144 = 72 prompts x 256 sampled tokens). PPL = served wiki / code (p7-vllm-ppl.sh).

| 8k decode step | ms | kernels/step |
|---|---:|---:|
| round-5b production (p48) | 37.36 | 1270 |
| + rotation stream 1 | 36.77 (-1.6%) | 1143 |
| + stream 1 + fused draft head (**adopted**) | **36.47 (-2.4%)** | 1107 |
| + stream 1 + fused QK-norm/mRoPE | 36.10 | 1007 |
| + all three | 35.91 | 971 |

**Rotation stream 1 is live** (host `RADIANCE_FP8_STREAM=0`): radiance_arnq no longer overwrites the decoder-layer
forwards, so each norm site runs one split add+RMSNorm+rotate+quant kernel instead of inductor's add+rms plus our rotate
(128 launches per step fewer). Sampled 89.1 -> 90.5, omp windows 77.8 -> 80.8 t/s (.559 -> .575), PPL 5.977/2.304 ->
5.970/2.304. Round 4 had measured it "~5% slower" -- that boot ran from another cache dir: serve-mxfp4.sh appends the
`-tp1s` suffix only when FP8_STREAM=1, so a stream-1 boot silently lost the TunableOp table (worth +5% on its own) and
now also the GPTQ drafter file. The launcher pins `CACHE` to cache-029-paro-tp1s when FP8_STREAM=0.

**Fused draft head** (`RADIANCE_DRAFT_FUSED=1`, radiance_drafthead.py `_apply_vocab_fused`): for the exact-set vocab path
the int2 kernel masks the padding rows and sums its x groups from the tile it already loads (drops fill + cat + cast +
reduce), skips the coarse-score write, and the rerank writes its bf16 logits straight into a -inf full-vocab row
(drops fill + cast + scatter + index_put): 15 -> 6 launches per drafter pass. `dh_check.py`: identical candidate sets and
logits at m = 1..16; end to end byte-identical drafts (same greedy text, identical omp-window acceptance counts).
Sampled 90.5 -> 91.0, omp windows 80.8 -> 81.3 t/s.

**Fused QK-RMSNorm + mRoPE + gate -- not adopted.** vLLM ships this as one Triton kernel but gates it on is_cuda(); on
ROCm it does not even compile (the AMD backend's TritonAMDGPUCanonicalizePointers pass aborts on base pointers merged out
of a runtime `if is_k:`). `radiance_qknr.py` restructures it (each branch calls one @triton.jit body with its own
pointers), and v2 drops the upstream bf16 round-trip of the normalized q/k before RoPE (it mimics eager PyTorch; the
production path is inductor's, which keeps them in fp32 -- vs an fp64 reference v2 sits at the single-rounding floor,
1.62-1.69e-3, upstream 1.78-1.91e-3, eager 1.89-2.15e-3; a defaulted constexpr kernel argument broke the engine boot
under torch.compile, so the switch is a jit global). It saves 1.8% of the step, but drafter acceptance on the omp prompts
drops by about as much (24 windows: .559 -> .549, .575 -> .555; 72 prompts: .592 -> .584, target-only .590) and PPL
does not move (5.970 -> 5.972): end to end +-0.6%, i.e. nothing. Kept in the tree, env-gated
(`RADIANCE_LOCAL_FUSED_QKNR=1`, `_TARGET_ONLY=1`).

**GDN z read in place -- mechanism narrowed, parked.** Eager boots with `RADIANCE_DEBUG_ZCHECK=1`: the GDN core op leaves
the projection's z region untouched (0 changed elements in 40 prefill checks) and the in-place read gives the same text as
the copy (3k and 20k prompts, identical shas). The x2 difference is therefore specific to the compiled / CUDA-graph path
(buffer lifetime across the core op's graph split), which also fits the round-4 faults in the z consumer. Worth ~0.7%;
needs the op to keep the projection alive (e.g. declared as a mutated input) -- not attempted.

### Where we started vs the finalist (p56, 2026-09-28 evening)

Both configurations re-measured the same evening on the same rig and harness, at the 250 W cap and at the card's stock
300 W (the user lifted the cap for this run; the udev rule restores 250 W at boot). **start** = production when the
review began (09-27 morning: libr4d rx9, full-vocab int2 draft head, stock fp8 prefill-attention legs, no TunableOp;
`deploy-vllm-qwen38-paro.start.sh` = 66d6496 with round 1 undone). **finalist** = `deploy-vllm-qwen38-paro.sh` now.

| | start 250 W | finalist 250 W | start 300 W | finalist 300 W | finalist 300 W vs start 250 W |
|---|---:|---:|---:|---:|---:|
| sampled 12x600 (code / prose / json avg) | 77.3 | 91.0 | 78.8 | **92.3** | **+19.5%** |
| held-out omp prompts (72), decode t/s | 70.8 | 81.3 | 72.9 | **83.6** | **+18.0%** |
| acceptance on those prompts | .631 | .592 | .631 | .592 | |
| decode step @8k (profiler median) | 44.68 ms | 36.79 | 43.87 | **36.31** | **-18.7%** |
| decode step @100k | 53.35 ms | 45.48 | 52.44 | **44.87** | **-15.9%** |
| greedy decode @8k (600 tok) | 113.9 | 137.1 | 115.0 | **139.1** | +22% |
| prefill @8k / 43k / 100k-class t/s | 2945 / 2585 / 2072 | 3135 / 2740 / 2172 | 3178 / 2780 / 2224 | **3408 / 2970 / 2357** | **+15.7 / +14.9 / +13.8%** |
| kernels per decode step | 1330 | 1107 | 1330 | 1107 | |

The software alone is worth +17.8% sampled / +14.8% on the omp prompts / +5-6.5% prefill at the same 250 W; lifting the
cap adds only 1.4-2.8% to decode (bandwidth-bound) but 7.5-8.7% to prefill (power-bound, as the GEMM analysis above
predicts). Acceptance is LOWER than at the start (.592 vs .631): the finalist drafts from a 48k-row vocabulary with a
4-bit (GPTQ MXFP4) drafter instead of the full fp8 one, which makes each draft pass far cheaper -- the step-time win more
than pays for it. 0 memory faults in every arm.

### Production after round 6

`deploy-vllm-qwen38-paro.sh`: round 5b + `RADIANCE_FP8_STREAM=0` (host; CACHE pinned to cache-029-paro-tp1s) +
`RADIANCE_DRAFT_FUSED=1`. Verified on :1246 (p42, 23:57): greedy 8k/100k sha dd828e3e / 9718d93c (identical to the rig
finalist), sampled 92.25 (92.7 / 78.7 / 111.0), omp prompts 83.1 t/s, prefill 3484 / 2977 / 2385 @8k/43k/120k, 0 faults --
at the stock 300 W cap the user set for the measurement (the udev rule /boot/config/udev/99-amd-powercap.rules puts
250 W back at the next boot). Rollback: `RADIANCE_FP8_STREAM=1` (and CACHE unset) restores round 5b's stream; drop
`RADIANCE_DRAFT_FUSED` for the unfused draft head.

## Round 7 (2026-09-29): VRAM -> KV pool (parallel sessions)

Why: two concurrent ~112k sessions filled the 273k pool to 90% and preempted one (decode fell to ~1 t/s until it cleared).
Peak VRAM of the prod config over 8k / ~135k prefills + 2 x ~112k concurrent (p57, sysfs sampled 0.1 s; the caching allocator
keeps its high-water mark) was 30,368 / 32,624 MiB, and 2.07 GiB of weights are touched rarely or a few rows at a time:

- vision tower (0.86 GiB bf16): stock vLLM 0.29 UVA offload, `--cpu-offload-gb 1 --cpu-offload-params visual`
  (`_mark_tower_model` routes towers through the offloader; the decoder stack is untouched by the `visual` filter).
- fp8 embed table (1.18 GiB): `PQ_EMBED_UVA=1` -> `_PQFp8RowEmbeddingMethod.process_weights_after_loading` moves it to
  pinned host memory behind a UVA view; the lookup gathers ~50 KB per decode step. Not a RADIANCE_* name on purpose: the
  table's location never changes a traced graph, so it must not re-key the AOT cache (patch_aot_envkey.py).

Result (launcher `OFFLOAD=1`, now default, KV_MEM 10.1e9 -> 13.9e9): weights 16.19 -> 14.12 GiB, **pool 273,333 -> 375,633
tokens**. Greedy shas unchanged (dd828e3e 8k / 9718d93c 100k), prefill 3490 @8k / 2439 @112k, sampled 91.4 t/s vs a
vision-only arm at 92.0 / 91.0 in the same session (embed-over-PCIe cost below run-to-run noise), screenshot TTFT 0.48 ->
0.72 s (vis_lat.py, fresh image per request). Stress at 13.9e9 (stress_tok.py, exact random-token prompts): a 255,000-token
request and 2 x 165,000 concurrent (+256 each) all served, 0 preemptions, KV 88%, peak 31,585 MiB (~1 GiB spare; don't push
further without re-running p57/stress_tok). Host cost: ~2 GiB pinned RAM on Tower. Rollback: `OFFLOAD=0` (KV_MEM back to 10.1e9).
Harness note: oai_bench_greedy `--depths N` is not N tokens (100000 -> 112k, 120000 -> 158k); use stress_tok.py for exact sizes.
