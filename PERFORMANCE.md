# Performance history and provenance

The headline numbers for the current build are in the
[README](README.md#performance). This document holds the longer form: what each landed change was
worth, how it was gated, the cross-image A/B that established the stack, and the gated-delta-net
overflow bug that made a pinned libr4d build necessary.

None of this is needed to run the server.

## Contents

- [Where the speed came from](#where-the-speed-came-from)
- [Two results worth carrying forward](#two-results-worth-carrying-forward)
- [ParoQuant against MXFP4](#paroquant-against-mxfp4)
- [Provenance: the 0.5.8 -> 0.7.4 baseline](#provenance-the-058---074-baseline)
- [The gated-delta-net NaN (fixed upstream)](#the-gated-delta-net-nan-fixed-upstream)

## Where the speed came from

Each row is a separate landed change with its own gate, measured against the build immediately
before it. This is not a decomposition of one A/B, and the rows are not additive: several move the
same wall time.

| Change | Measured | Gate |
|---|---|---|
| **TP=1 lazy GDN state snapshots, 2026-09-17 — REVERTED 2026-09-17, see the note below this table** (`RADIANCE_GDN_LAZY`, now DEFAULT OFF; libr4d **rx10**, patch_gdn_lazy.py, radiance_gdn_lazy.py) -- one base state per sequence plus a per-head stash of the last step's candidate inputs, replayed on the next step (`gdn_lazy_update`), and a `gdn_lazy_materialize` kernel that stands in for vLLM's two align-mode temporal-state copies (pre-forward migration at a window shift, post-step checkpoint at a block boundary) as base + replay. A request holds **3** mamba pages per layer group instead of 2+SPEC = 9 | One R9700, MXFP4-mtpfp8, 65536 ctx, lazy vs the eager TP=1 config of 09-16: decode-step 35.0/36.9/38.4 vs 35.7/37.2/38.9 ms at 0/8k/32k (parity); prefill same-card 3104/3048/3041/2860/2647 vs 3150/3085/3070/2881/2668 PP t/s (-1.5%); concurrency agg 117/205/313/**420** vs 112/199/304/337 at 1/2/4/8, C8 TTFT p50 **0.17 s vs 2.8 s**; MAXSEQS=16: C16 **487 t/s** (TTFT p50 0.53 s), C8 unchanged. KV pool 99k vs 77k tokens at MAXSEQS 8. Shipped default (lazy, MAXSEQS 8) BetterBench --quick on card 0: combined decode **137.7 t/s**, update p99 35.7 ms, TTFT p50 ~95 ms; per-category decode chat 81 / code 145 / file_edit 166 / json 184 / math 169 / prose 79 / reasoning 121 / summarization 138; concurrency 117/208/319/416; prefill 2737/2720/2778/2651/2480. Kernel: eager16 vs lazy16 us/layer 17.5/17.5 (1x8), 68/82 (8x5), 166/117 (8x8), 180/119 (16x4). TP>=2 untouched: patch not applied, rx6 kept, cache dir unchanged | lazy vs eager harness: fp32 state bit-identical over 12 steps of random acceptance; align-mode simulation with 6 window shifts + 6 checkpoints agrees to 1e-7; 3000-token generation across boundaries clean, zero fallbacks/errors; GSM8K 250q greedy conc-8 **97.60%** at MAXSEQS 8 and **98.40%** at MAXSEQS 16 (eager side 98.40%, baseline 98.00%; the day's band is 97.6-98.4). Note the two cards differ ~12% on prefill under the same 210 W cap (card 0 runs hotter): compare prefill same-card only |
| **TP=1 (single card) pass, 2026-09-16** -- decode-kernel width cap 32768 -> 36864 (TP=1 gate_up is N=34816 and every decode call of it ran the folded prefill tile: 400 vs 158 us at M=8); fp8 residual-stream epilogues installed at TP=1 without an all-reduce (`RADIANCE_FP8_STREAM_TP1`, 64 mid + 63 down + 64 act + 48 GDN sites); libr4d **rx9** = rx6 + narrow-state GDN decode kernels so the fp16 ssm cache no longer declines every GDN layer to FLA (conv state bf16, ssm fp16); fused GDN step routed at 48 items (one sequence at H=48); silu-mul-quant epilogue widened to N<=20480 (512-thread block); launcher `GPUS=<n>` device re-indexing fix; MAXLEN 65536 default at TP=1; TP=1 KV pin measured on the NVFP4 checkpoint (calibrate-kv.sh `KV_START`, kv-profiles.tsv row `1x7551-32624`: 6.09 GiB/GPU, ~140k tokens at 65536 where the profiler itself could not reach the 2.85 GiB minimum) | BetterBench paired A/B, 200 interleaved pairs, one R9700 per side, MXFP4-mtpfp8 ckpt, 65536 ctx: decode **74.3 -> 134.8 t/s (+81%, CI +78..+85)**, update gap 63.9 -> 35.4 ms; combined decode 75.4 -> **138.7**; prefill PP t/s **same-card (card 1)** 2618/2515/2473/2348/2191 -> **3150/3085/3070/2881/2668** (+20..+24%, the baseline's GDN prefill was on the FLA fallback; the first cross-card reading of +31..+41% was inflated by card 0 running ~12% slower on prefill under the same 210 W cap); concurrency agg 65/117/188/194 -> **116/202/312/308** t/s at 1/2/4/8, C8 TTFT p50 6.3 -> 4.3 s; bench_decode_ctx 64.2/66.6/68.1 -> 35.7/37.2/38.9 ms/step at 0/8k/32k. Split-K rule re-measured on every TP=1 shape: already optimal, unchanged. TP=2 byte-identical by construction (dry-run diff: only two inert env knobs) | GSM8K 250q greedy conc-8 on the candidate: **98.40%** (246/250, 0 errors, 1 truncated) vs the baseline side 98.00% (245/250) -- same band; kernel epilogues bit-identical to the traced path (unchanged kernels); GDN fp32 path unchanged (rx9 fp32 code = rx6) |
| Decode launch-gap stack: traced quant, fp8 residual stream, fused AR epilogue (`db9dba6`) | **25.4 -> 22.66 ms/step**; decode launches 1477 -> ~1080/step; weighted single-stream +14%, conc-8 aggregate +24% | GSM8K 500q 97.8, paired sign test p=0.219; epilogue kernels bit-identical to the traced reference |
| Dynamic verify width (`f68d215`) | conc-8 steps 52-57 -> **46-47 ms**, aggregate 391-413 -> 444-461 t/s (**+11-13%**); single-stream a wash | Lossless by construction: speculative verification preserves the output distribution at any proposal length |
| fp8 QK + PV legs in prefill attention (`afa21b5`) | prefill 4442 -> 4648 t/s @ 40k, 3448 -> **3831 t/s @ 106k** (+10.8%); kernel-level 96.2 -> 150.5 TF at a hot 8k chunk | ppl 8.3708 -> 8.3707, top-1 54.09 -> 54.13%; GSM8K paired p=1.000. Upstream deleted these legs on *kernel* accuracy; the end-task gates say the error is free |
| KV cache group size by capacity, not smallest bucket (`1bf3914`) | **739,544 -> 892,799 KV tokens** (+20.7%); concurrency 2.82x -> 3.41x | Allocator-derived group size, unchanged for the n:1 layouts upstream targeted |
| Explicit KV pin over profiling | 892,799 -> **943,581 KV tokens** (+5.7%); 3.60x | Survives a full 260k-prefill sweep with no OOM. See [KV cache calibration](README.md#kv-cache-calibration) |
| int2 target verify head (`f882ea2`) | combined decode 170.0 -> **174.9 t/s** (+2.9%), all 8 categories +2.7-3.4%; conc 1/2/4 +2.8/+2.5/+1.6% | Equivalence, not a score: 24/24 seeded sampled completions byte-identical against a sequential self-consistency control |
| Epilogue store width T=512 at prefill M (`5950b38`) | 726 -> **685 us** at M>=2048 (405 -> 429 GB/s); TTFT @ 32k 7443-7878 -> 7196/7260 ms | GSM8K 97.60, in band (the 512-way striding changes per-row summation order); decode untouched and byte-identical |
| Fused GDN gated-norm + quant (`radiance::gdn_norm_quant`, `RADIANCE_GDN_NORM_QUANT=1`, default 2026-09-02) | single-stream 22.51 -> **22.32 ms/step** (-0.8%), 25.8 -> 25.1 @32k; one launch replaces two inductor kernels on each of the 48 linear-attention layers | GSM8K 500q 97.60%; BetterBench single-pass update p50 -0.2 ms in every category, tok/update neutral. Not bit-exact (silu 1 ulp), so judge on multi-prompt tok/update |
| Fragment-order weights + nontemporal decode loads (`RADIANCE_MXFP4_WPERM=1`, `RADIANCE_MXFP4_DECODE_NT=1`, defaults 2026-09-02) | single-stream 23.95 -> **22.66 ms/step** (-5.4%), 27.23 -> 25.65 @32k; BetterBench combined 195 -> 227 t/s single-pass | Acceptance byte-identical; GSM8K 500q 97.40%; prefill +0.3..+3.3% vs WPERM=0 (the A-tiled kernel is layout-neutral) |
| GDN `in_proj` single-GEMM merge (`588d5e6`) | single-stream 26.25 -> **25.50 ms/step** (-2.9%), -6.5% stacked with `WPERM=1`; removes 96 GEMM launches and 48 activation quants per forward | GSM8K 500q paired; drift is split-K reassociation only |
| GDN decode conv+recurrent fused into one launch (`9a84208`) | 25.01 -> **24.91 ms/step** (-0.4%) | **Bit-identical**: 4/4 byte-equal greedy completions |
| Decode band extended to M<=128 (`44d48f0`, opt-in at `MAXSEQS>8`) | conc-16 **549-622 t/s** at 75-79 ms steps, +30% over the pre-extension attempt | dks1 bit-identical to the folded tile at every shape and M in {72, 96, 127, 128} |

### Correction: lazy GDN snapshots were reverted the day they shipped

The row above is kept because it records what was measured, but the change is **off by default
since 2026-09-17**: lazy GDN snapshots corrupt multi-turn chat. Replies decay over turns, then
collapse into empty completions and hard repeat loops. Controlled A/B — one scripted 25-question
x 2-round conversation on the chat endpoint, temperature 0, seed 1234, the same harness on both
legs, rx10 pinned on both and the pair path forced on both, so the flag was the only variable:

| leg | turns healthy | empty replies | repeat loops | first failure |
|---|--:|--:|--:|---|
| `RADIANCE_GDN_LAZY=1` | 10/50 | 35 | 1 (198 tokens) | turn 5, 3,298 tokens of context |
| `RADIANCE_GDN_LAZY=0` | 49/50 | 0 | 0 | none |

The shipped eager path (rx9, fused-items 48 live) was gated separately: 24/25 turns healthy, 0
empty, 0 loops, conversation carried to 15,603 tokens.

**Why the original gate missed it.** It is not a long-context bug. Single-shot completions, needle
retrieval at 8k/12k/32k (lazy and eager fail on the *identical* six cases, Fisher p = 1.0) and
fp16-vs-fp32 state all read clean. The failure needs multi-turn chat: the reporting session ran
73-77% prefix-cache hits, against ~8% for every single-shot gate in this document. Judge a state
cache with a multi-turn conversation, never a needle or a single completion.

**Prime suspect, not yet proven:** `gdn_lazy_materialize` mode 1 fails open — a stash whose magic
or `base_slot` does not match replays nothing and writes an aligned checkpoint silently missing
`count` tokens, which is what a prefix hit then restores from. The stash's only validity check is
a physical block id, and block ids are recycled across turns.

**The concurrency figures in the row above do not reproduce.** They were taken before the measured
`--kv-cache-memory` pin existed. BetterBench `--quick`, same card, one R9700, current launcher,
**eager** (the new default):

| | 1 | 2 | 4 | 8 |
|---|--:|--:|--:|--:|
| aggregate t/s | 114.7 | 205.8 | 303.3 | 405.6 |
| TTFT p50 (ms) | 93.8 | 135.5 | 146.5 | 174.3 |

Prefill 2,738 / 2,721 / 2,795 / 2,666 / 2,494 PP t/s at 2k / 8k / 16k / 32k / 64k; KV pool
**140,036 tokens**. So eager reads 405.6 at conc-8 against the 337 recorded above, with a *larger*
KV pool than lazy's recorded 99k. The paired lazy leg was abandoned once the flag was reverted, so
no lazy-vs-eager delta is claimed here.

## Two results worth carrying forward

**`SPEC` depth is content-dependent, and tuning it on one content class picks the wrong default.**
A sweep on non-repetitive prose flipped the dflash default to 5. On BetterBench's weighted mix, back
to back on the same build, `SPEC=7` scores 184.3 t/s combined against `SPEC=5`'s 159.4 (+15.6%),
because code/json/file_edit run 4.7-6.0 tok/update at depth 7 and a cap at 5 truncates exactly the
high-acceptance tail the mix rewards. `SPEC=5` keeps the edge on prose-heavy content and on batch
throughput (conc-8 562 vs 544). Dynamic verify width mostly dissolves the trade-off. (`5692fed`)

**Fill the GPU before judging a tiling.** The epilogue prefetch above was justified by an M=2048
microbench on an idle, VRAM-squeezed GPU and shipped as a regression: at the real prefill shape
(M=8192) it measures 828 us against the original 726, **14% worse**, because its +40 VGPRs/thread
costs occupancy exactly when 8192 workgroups compete. (`eafcac9`, reverted in `5950b38`)

## ParoQuant against MXFP4

The second int4 format this stack serves, measured on the same box against MXFP4 production. The
path itself — format, kernels, knobs — is documented in [PAROQUANT.md](PAROQUANT.md); this is what
it is worth.

| | MXFP4 prod | ParoQuant | |
|---|---|---|---|
| GSM8K 500q | 97.8% | **97.4-98.0%** | per-token vs per-group activation scales; inside binomial noise |
| decode @ ctx25 | 22.3 ms/step | 24.19 ms/step (**23.27** since 2026-09-09: skinny gate GEMM + util 0.95, KV 854k) | -7% (-4%) |
| combined decode | 186.0 t/s | **226.2 t/s** | +22% |
| KL vs FP8 serve (top-20, wikitext / code / served) | — | MXFP4-PARO: 0.057 / 0.054 / 0.044 nats; top-1 agreement 90.5-92.7% | see PAROQUANT.md |
| conc 1/2/4/8/16 | — | 168 / 276 / 399 / **512** / 518 | |
| prefill @ 2k/8k/16k/32k/64k | — | 3782 / 3700 / 3725 / 3621 / 3450 t/s | -8.8% at 8k when it shipped, parity from ~64k |
| KV cache profile | — | 622k tokens | |

Weight-side traffic is 4.25 bits/weight in both formats (ParoQuant's asymmetric zero point rides in
the same 4-byte load as the scale), which is why they land so close at the memory-bound end. The
gap that remained was launch count, not arithmetic: at ship time the rotation was 192 separate
launches per decode step, ~2 ms of a 26 ms step.

Where the ParoQuant speed came from, each measured against the build before it:

| Change | Measured | Gate |
|---|---|---|
| A-tiled prefill GEMM + fragment-order decode + prologue v2 | prefill 3220/3144/3148/3139/2984 -> **3789/3723/3668/3630/3427 t/s** (+15-18%); 24.27 ms/step (was 25.74, -5.7%) | GSM8K 500q 97.60%; harness bit-exact incl. the bf16 scratch rounding |
| Rotation stream 1 — residual add + RMSNorm + rotate + quant as one producer kernel | 24.27 -> **24.03 ms/step**; combined decode 186.0 t/s (= MXFP4 prod); KV profile 432k -> 479k tokens | GSM8K 500q **98.00%**; 96 rotation launches per step removed |
| MXFP4-PARO fused tiled prologue (2026-09-08) | Prefill band: one workgroup per row parks the rotated row in LDS and writes the fragment-tiled A directly (`pq_rotate_tokquant<W,TILED>`, tiled stream producers), replacing pass A + tiled pass C. Prod BetterBench prefill 4770/4827/4649/4495/4273 PP t/s @2k-64k (+9%/+8%/+5%/+5%/+5% over the single-launch build, +26%/+30%/+25%/+24%/+24% over int4 PARO); decode unchanged. Two-chain interleaving in the producers: byte-exact, neutral, dark | `--bench2 tokqt` 24 shapes AT/AS byte-exact, 1.3-1.4x at M>=600; loader test all bands + stream equivalence at M=600 tiled |
| MXFP4-PARO prod decode: launch count (2026-09-08) | First prod boot 35.40 ms/step vs int4 PARO 24.19 with acceptance AND GEMM at parity -- all launch count. Fused rotate+token-quant kernel 35.40 -> 28.48; per-token stream producers (norm/silu/gate/gdn-norm + rotate + quant, shared `install_stream` dispatching on the consumer's quant method) -> 26.29; single-launch merged GEMM with in-kernel partition select -> **24.53 / 25.89 / 26.80** @ctx25/8k/32k (int4 24.19 / 25.74 / 26.70). BetterBench combined 126 -> 184 -> 200 -> 203 t/s (int4 226; tokens/update trail, not step time). Per-step linear cost at M=8: 12.7 ms int4, 9.2 ms MXFP4-PARO. Prod prefill (BetterBench sweep) 4376/4449/4423/4291/4068 PP t/s @2k-64k vs int4 3782/3700/3725/3621/3450 (+16%/+20%/+19%/+18%/+18%); KV 862k tokens at GPU_UTIL 0.95 (int4 622k) | Harness `tokq` (54 shapes) and `tokstream` (45) byte-exact vs the unfused chains; loader test: stream tuple `torch.equal` to the plain path at every site/M; single launch vs per-partition loop 1-2 ulp flips per million (split-K reassociation); GSM8K 500q with the stream 97.40% (487/500) |
| MXFP4 weights + z-lab rotations (`paroquant_mxfp4`, 2026-09-08) | GEMM inner loop 16 VALU -> **0**; A-tiled prefill enabled (layouts identical). TP=1 eager A/B vs int4 PARO, same launcher: prefill **+9.8/+8.5/+6.8/+6.2%** at 2k/8k/16k/30k (2434 vs 2217 tok/s @2k), eager decode +27% (indicative); decode traffic unchanged (4.25 bits/weight both) | loader vs independent fp32 reference at the e4m3 floor (0.009-0.014) on every module/K/M tried (K=5120/6144/17408, M=1..2048, TP=2 slices); in-serve CHECKALL with real inputs, TP=2: rel 0.0012-0.0021 on every shape/partition (bf16 output rounding floor); GSM8K 500q: one-shot pseudo 96.96% -> fine-tuned pseudo 97.20% -> **fine-tuned SERVED W4A8 97.60%** (488/500, 0 errors; int4 PARO 97.60, AMD MXFP4 97.8) |
| Rotation stream 2 — silu-mul, gated norm, attention gate producers | 24.53 -> **24.19 ms/step** (-1.4%); combined **226.2 t/s**; conc-8 500 -> 512; KV profile 479k -> **622k tokens** | Harness 24/24 bit-exact vs the unfused chain; GSM8K 97.40% |

The KV profile moving 432k -> 479k -> 622k tokens across the two stream landings is worth noting on
its own: fusing producers removed intermediates from the compiled graph, and the capacity came back
as cache.

**Rejected: rotation stream 3**, the two-rank all-reduce fused into the norm+rotate producer. It is
*correct* — bit-exact single-rank loopback, an in-serve all-reduce check passing on every call on
both ranks, GSM8K 97.40, identical sanity completions — and still slower: 25.10 vs 24.19 ms/step,
202 vs 226 t/s combined. With one workgroup per row the rotation chains serialize on a single CU
and the push uses M CUs instead of r4d's 24 blocks, costing +7 us per site. Two earlier designs
deadlocked outright, in both cases because a spin-wait on peer flags outgrew residency or changed
its slice mapping with M. The MXFP4 `exact_nq` all-reduce wins with the same structure only because
its epilogue has no rotation to serialize.

One methodology note carried over from the MXFP4 work and re-earned here: **judge layout A/Bs with
short benches.** Under the fixed 210 W per-card cap a long run and a short run of the same build do
not measure the same machine, and a GSM8K comparison run against the wrong chat template moves the
score by two points regardless of the kernel — the 97-98% band needs `qwen-fixed-v22.3.jinja`.

## Provenance: the 0.5.8 -> 0.7.4 baseline

The cross-image A/B that established the stack, kept because it is the one clean comparison this box
can make and every number in the README is measured downstream of it. Both arms ran under
`SPEC_METHOD=mtp` at `SPEC=4`, the default at the time; the shipped default is now the `dflash`
drafter.

| | 0.5.8 | 0.7.4 | |
|---|--:|--:|--:|
| prefill 7.8k | 3873 | **4387** | +13.3% |
| prefill 26k | 3445 | **4138** | +20.1% |
| prefill 104k | 2310 | **3143** | +36.1% |
| prefill 182k | 1736 | **2511** | +44.6% |
| prefill 260k | 1393 | **2089** | +49.9% |
| decode short / medium | 63.0 / 67.4 | **67.1 / 67.5** | +6.5% / +0.1% |
| WikiText-2 PPL | 8.3335 | 8.3719 | +0.46% |

All 304 linear layers run the W4A8 fp8-WMMA kernel; `aiter` is not used at all now that
`RADIANCE_MXFP4_W4A8_MIN_M` defaults to 0. The prefill gain scales with context because it is mostly
R4D's paged attention, whose share of prefill grows with sequence length.

### The decode GEMM, measured against the same 0.7.4

`RADIANCE_MXFP4_DECODE_MAX_M`, default 64, toggled against the same build with it off.

| | Off | On | |
|---|--:|--:|--:|
| single stream, ms/step | 35.06 | **32.16** | -8.3% |
| ms/step at 32k context | 36.53 | **33.47** | -8.4% |
| aggregate tok/s, 4 concurrent | 170.1 | **218.5** | +28.5% |
| aggregate tok/s, 8 concurrent | 295.0 | **353.1** | +19.7% |
| prefill, all five lengths | -- | -- | unchanged (-0.3 to -1.2%) |
| GSM8K 500q, greedy | 97.80% | 97.80% | 3/3 discordant, sign test p=1.00 |

Batched workloads gain most because at M=20-40 aiter's tuned band uses `NUM_KSPLIT=1`, which leaves
the grid underfilled, while this kernel keeps split-K. GSM8K also ran **14% faster wall**
(375.5s -> 322.7s) on slightly *more* generated tokens.

## The gated-delta-net NaN (fixed upstream)

libr4d v0.4.0 produces NaN in the gated-delta-net output on this model: WikiText-2 PPL **653586**
with the W4A8 path and no mitigation. This is why `setup-mxfp4.sh` builds libr4d at a pinned commit
rather than using the one baked into the image.

### The three overflows

All three are the same shape: an unguarded `__expf` on an inactive lane or a split-form half,
giving `0 * INF = NaN`.

1. **`kkt_solve`, padding rows.** `gi` is forced to 0 for `i >= rows` while `gb[j]` keeps its real
   negative cumsum, so `d = -gb[j]` is large POSITIVE, the opposite of the "never positive"
   invariant the code asserts, which holds only for live rows. The NaNs land in padding rows of the
   64x64 tile, and the blocked inverse merges the whole tile with WMMA, so they reach live rows.
2. **`chunk_scan`, split-form halves.** `e^{g_i-c}.e^{c-g_j}` with `cref` at the chunk midpoint
   gives each half +/-(gate span)/2; a span past ~176 sends one to +INF and the other to 0.
3. **`chunk_scan`, `V'` staging.** The dominant one, and only visible across chunks. `V' = V.gv[t]`
   is staged in **bf16**, so `gv` must leave room for `V` under bf16's 3.4e38 ceiling. Clamping at
   `e^88` still NaNs; `e^80` leaves margin.

Clamp value, measured over 208,539 WikiText-2 tokens with no other mitigation:
70 -> 8.3841, **80 -> 8.3706**, 83 -> 8.3728, reference 8.3335, stock 653586. (Those were measured
before the fold was widened; on the current build the same configuration reads 8.3719, against
8.3736 with the original fold table.)

### The fix, and what the clamp does not fix

Fixed in **StillDeadcode/libr4d PR #1** (merged 2026-08-24). Not in a tag yet, which is why setup
builds libr4d at a pinned commit. The build is verified reproducible: it produces an `r4d.so`
byte-identical (sha256 `3026297b...`) to the one every number here was measured with.

The pin is deliberate. Nothing version-checks the library it loads, so a later commit that renames
an entry point or changes a compiled-in geometry constant makes `radiance_gdn.py` set
`ENABLED = False` and **fall back to the Triton path with one line on stderr**, costing performance
rather than raising.

**The clamp bounds the damage; it does not remove the cause**, and upstream sharpened this point
when merging. The original note here claimed the clamped product "evaluates to 0, which is the
correct answer". That is wrong: what leaves range is the distance from `cref`, not `g_i-g_j`, so on
a span-200 chunk the last token's own diagonal, and its `e^{gl-g_t} ~ 1` weight into the state which
the next chunk reads, are *attenuated* by `e^{80-(cref-g_t)}` rather than correctly vanishing. The
real fix is to stop splitting a weight that is provably <= 1 into a huge x tiny pair: stage `V'` in
fp32, or apply `e^{gl-g_t}` directly on the state path.

Fixing this also removed a second symptom: `RADIANCE_FAST_DRAFT` used to hang a worker at chunk 8192
because the draft head was being fed NaN like everything else downstream of the GDN core.

---

## CPU KV offload with dflash on the hybrid GDN model (2026-09-22) -- FIXED

The native `OffloadingConnector` CPU tier is meant to catch a long prefix evicted from the GPU KV
pool and promote it back instead of re-prefilling. With `SPEC_METHOD=dflash` it promoted nothing
(`vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} == 0`), so a 130k re-prefill cost
~66 s; with speculation off the same connector worked. Fixed by
**`patch_offload_eagle_fallback.py`** (applied by default by `serve-mxfp4.sh`).

**Root cause.** `SchedulerOffloadConfig.from_spec` (offloading connector) falls back to

```python
if use_eagle and not eagle_groups:
    eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
```

so on this model it marks **all nine** offload groups as EAGLE groups. Groups 0-5 are GDN/Mamba
(SSM), not draft attention: `is_store_reachable_swa_chunk` then stores `sliding_window+1 = 2` tail
chunks and the load demands `sliding_window+1 = 2` consecutive tail hits, but the GDN store only
ever captures a single state snapshot per block-aligned prefill step, so the tail run is length 1
and `_sliding_window_lookup` returns 0. Because the load loop is all-or-nothing
(`if num_hit_chunks == 0: return 0`), one failing SSM group abandons the restore for every group,
including the fully-hit attention ones. The core already exempts SSM from this rule
(`v1/core/kv_cache_coordinator.py:755` guards with `not isinstance(spec, MambaSpec)`;
`MambaManager.find_longest_cache_hit` ignores `drop_eagle_block` -- draft models have no Mamba
layers). The fix restricts the fallback to non-Mamba groups.

**Geometry that made the fallback fire** (TP=1, CHUNK=4096, `--mamba-cache-mode align`, MAXLEN
160000): 9 groups, `tokens_per_chunk=880` for all, `blocks_per_chunk=1`; 0-5 GDN `sw=1`, 6-7 full
attention, 8 draft SWA `sw=cdiv(2048,880)=3`; `alignment_chunk_count=None` everywhere (alignment
tokens == tokens per chunk). No group carried `is_eagle_group` (that annotation is DeepSeek-V4
only), which is why the coarse fallback triggered.

**Measured** (`RADIANCE_OFFLOAD_TRACE=1` instrumentation, first request cold, second 130k prompt
evicts it, third reuses it):

| CPU tier | before fix | after fix |
|---|---|---|
| 16 GiB, 130k | GDN `window=2` hits=1/miss=145 -> 0 -> re-prefill 67.2 s, CPU->GPU 0 | GDN `window=1` hits at tail, but 130k SSM keys evicted -> still aborts |
| 24 GiB, 130k | re-prefill 66.1 s | **restore 126720/130017 tok, TTFT 2.92 s, CPU->GPU 4.26 GB** |
| 16 GiB, 90k | re-prefill 40.9 s | **restore 88000/90002 tok, TTFT 1.6-4.7 s, CPU->GPU 3.0 GB** |

Correctness: 192 greedy tokens after an offload-restored 90k prefix are byte-identical to the
cold-prefill continuation.

**Capacity note.** GDN snapshots exist only at block-aligned prefill-step ends (indices
3,7,11,...,143,145 for a 147-chunk prompt), and the load's `window=1` caps the restore to the
nearest retained SSM boundary. Those SSM blocks are large and LRU-evicted first, so the CPU tier
must be sized for the context you want to restore: 8 GiB is too small even for one 130k prefix
(real KV is ~8.16 GiB for 228k GPU tokens), 16 GiB holds a 90k restore, 24 GiB a 130k restore.
Set `cpu_bytes_to_use` (offload region, `/dev/shm`) accordingly.

### Benchmarks (2026-09-23): the KV_MEM <-> CPU-tier trade

Workload: N distinct prompts of L tokens, round 1 fills (C1), round 2 reuses them in order
(C1), `ignore_eos`, 8 out. `bench_offload.py`, `bench_normal.py`. One R9700 (PCIe Gen5 x16),
TP=1, CHUNK=4096, MAXSEQS=8.

| config | reuse TTFT p50 | CPU->GPU promoted | normal C1..C8 |
|---|---|---|---|
| KV 6 GiB, offload off, 3x60k | 22.3 s (re-prefill) | 0 | — |
| KV 6 GiB, offload on 24 GiB, 3x60k | **2.30 s** | 5.53 GB / ~147k tok | — |
| KV 6 GiB, offload on 8 GiB, 3x60k | 22.2 s (re-prefill) | 0 | — |
| KV 6 GiB, offload on 8 GiB, 3x60k, newest-first | 1.03 s for the newest (GPU hit), rest re-prefill | 0 | — |
| KV 8.16 GiB, offload off, 5x80k | 35.9 s | 0 | 102/185/281/419 |
| KV 8.16 GiB, offload on 24 GiB | GPU-resident | — | 102/187/280/424 |
| multi-turn 60k, offload off | turns 2-6 = 0.75 s | — | — |
| multi-turn 60k, offload on | turns 2-6 = 0.75 s | — | — |

Findings:

- **The lever works.** Halving the GPU pool (8.16 -> 6 GiB) plus a 24 GiB CPU tier turns a 22.3 s
  re-prefill into a 2.30 s restore (9.7x) while freeing ~2 GiB of VRAM. Normal-usage throughput
  and multi-turn TTFT are unchanged (offload is inert until the GPU pool evicts).
- **Restore is a two-tier chain.** The GPU prefix cache serves the head; the CPU tier serves from
  the GPU hit boundary (`start_chunk_idx = num_computed_tokens // tokens_per_chunk`). Partial fit
  is real: measured GPU 47,520 + offload 29,920 = 77,440 of 79,996 tokens in one restore.
- **CPU capacity is ~2.0x less dense than the GPU pool.** An `OffloadKey` is a block hash plus a
  group index (`v1/kv_offload/base.py:29`), so one 880-token chunk spans up to nine keys (one per
  group), while each CPU block is sized from `worker_kv_bytes_per_block =
  total_gpu_kv_bytes // num_blocks` -- a full chunk across *all* groups
  (`.../offloading/config.py:111`). That charges each key a full all-groups block for a single
  group's payload, so the tier retains ~half the tokens the same bytes hold on the GPU: measured
  ~**14k tokens/GiB** vs ~**27.9k tokens/GiB** (≈75 KiB vs ≈37.5 KiB per token), i.e. 24 GiB
  ≈ 336k tokens and 28 GiB ≈ 392k. Size the tier at ~75 KB per restorable token.
- **Contiguity is required.** Full attention needs every block from chunk 0, and LRU evicts the
  oldest, so a prefix whose KV does not fit the tier loses its head and restores *nothing* -- not a
  partial head+prefill. The 8 GiB tier (~272 blocks, ~64k tokens) could not hold a 60k prefix once
  other prefixes were also storing, so offload never fired. Partial recovery only works when the
  GPU tier-1 already holds the head.
- LRU also means a cycling workload that exceeds the tier degrades badly: re-prefilling an early
  prefix evicts the newer ones before their turn (observed 0 restores for oldest-first reuse of
  3x60k at 8 GiB). Reusing newest-first restored the newest prefix. **ARC did not help** (see
  below); a per-request *head cap* does.

### LRU vs ARC, and the per-request head cap (2026-09-23)

8 distinct 60k prefixes, round 2 reuses them in order, KV 8.16 GiB, CPU tier 24 GiB,
`bench_offload.py`:

| policy | reuse | sum of TTFTs | promoted |
|---|---|---|---|
| `eviction_policy=lru` (default) | 0 restores, all 21.95 s | 175.6 s | 0 |
| `eviction_policy=arc` | 0 restores, all 21.96 s | 175.7 s | 0 |
| `lru` + `max_offload_tokens=40000` | all 8 restore 35,200 tok, 10.50 s | **84.0 s** | 10.13 GB |

The cycle exceeds the tier (8x60k = 480k tokens > the 24 GiB tier's ~336k), so plain LRU evicts
each prefix's head before its reuse turn and every reuse re-prefills; ARC tracks recency/frequency
but still cannot keep a contiguous head, so it is identical. **The fix is a per-request head cap**:
`max_offload_tokens` caps each prefix from token 0
(`_calc_num_offloadable_tokens = min(num_computed_tokens, max_offload_tokens)`), so each prefix
keeps a contiguous head that fits the tier and restores *partially* (head from the CPU tier,
remainder pre-filled) instead of nothing -- measured 2.1x on this workload. `patch_offload_head_cap.py`
sets the server default (`kv_connector_extra_config["max_offload_tokens"]` or
`RADIANCE_OFFLOAD_MAX_TOKENS`); `serve-mxfp4.sh` applies it unconditionally.

**Is offload worth it while staying at the production 8.16 GiB GPU pin?** Yes, but only when the
*sum of distinct live long prefixes* exceeds the GPU pool (228k tokens) -- i.e. multi-session /
multi-document traffic, not a single chat. A single long chat is always a GPU hit (~0.75 s) and
never touches the CPU tier. Measured envelope (KV 8.16 GiB, real + dummy runs):

| CPU tier | workload | reuse | promoted |
|---|---|---|---|
| 24 GiB | 5 x 60k = 300k tok | all 2.27 s | 9.79 GB |
| 24 GiB | 4 x 80k = 320k tok | all 1.96 s | 10.60 GB |
| 24 GiB | 3 x 130k = 390k tok | all 57.6 s (re-prefill) | 0 |
| 28 GiB | 3 x 130k = 390k tok | all 2.94 s | 12.79 GB |
| 24 GiB | 2 x 130k = 260k tok | all 2.92 s | 4.26 GB |

The tier must hold the **whole** set of prefixes you want to restore (each needs a contiguous head),
at roughly **14k tokens/GiB** (28 GiB ~= 390k tokens). So size `cpu_bytes_to_use` at about
`total_prefix_tokens / 14000` GiB: ~19 GiB for 2x130k, ~28 GiB for 3x130k, ~23 GiB for 4x80k.
Below that, LRU evicts a prefix's head and it re-prefills entirely -- no partial recovery. This is
RAM in `/dev/shm` plus PCIe traffic. **Measured 2026-09-23: the two cards are very different.**
`ROCR_VISIBLE_DEVICES=0` (0000:05, behind the 500-series chipset) does 0.80 GB/s host<->GPU;
`=1` (0000:09, on the CPU root complex) does 28 GB/s -- a 35x gap -- even though sysfs reports
`current_link_width=16`, `32.0 GT/s` for both, so sysfs is misleading here (`lspci -tv` shows one
card behind the chipset). Run offload-heavy instances on the CPU-attached card, or move the
chipset card to a CPU slot.

`./offload-size.py --context 130000 --sessions 3` computes this recommendation, checks it against
`/dev/shm`, and prints the ready-to-paste env (it reproduces every measured threshold above:
3x130k -> 27.9 GiB, 4x80k -> 22.9 GiB, 5x60k -> 21.5 GiB). `--capacity --cpu-gib 24` reports what
a fixed tier can retain; `--json` for tooling. When the working set exceeds `/dev/shm`, it warns and
emits a `max_offload_tokens` head cap (`shm_bytes / (bytes_per_token_cpu * safety) / sessions`,
chunk-aligned) so each prefix can still restore a contiguous head instead of nothing.


### Improving density, speed, and adding a disk tier (2026-09-23, research)

Prior art: `zzpanic/qwen3.6-vllm-gfx1201-launchers` (`kv-cache/`) is a house copy of this
`serve-mxfp4.sh`, re-configured for KV offload on the same image (radiance 0.9.3 / vLLM 0.27.1,
gfx1201, Qwen3.8-27B MXFP4 + DFlash2 FP8, TP=1). It runs GPU → RAM → disk with a 15-patch set and
publishes the analysis. Two things it states outright, both confirmed in our source at
`/tmp/kilo/vllm-src`:

- Our CPU-tier overhead is **temporal, not spatial**. A Mamba group is one recurrent state, and
  `get_sliding_window_size_in_chunks` already returns 1 for `MambaSpec`; the load path fetches
  exactly one Mamba chunk, yet `_build_store_jobs` writes a fresh snapshot of **all six** GDN
  groups at **every** chunk. Their measurement: 147,456 B/token written, ~34,264 read back — a
  **4.17x amplification**. Our measured ~2.0x is the same effect with a different chunk/grid.
- vLLM 0.27.1 **already ships a filesystem secondary tier** (`v1/kv_offload/tiering/{spec,fs}`).
  Disk offload is configuration, not a port — with one hard operational requirement (no eviction).

#### Density: `patch_mamba_stride.py` (the single biggest lever)

Store Mamba/GDN snapshots every Nth chunk, with `resolve_mamba_align_size` rounded to the same
grid so store and load agree. Both halves must match or a lookup probes a state that was never
written. Our model is 9 groups (0-5 GDN, 6-7 full attention, 8 draft) and a chunk of 880 tokens. Counting
one stored block per group per chunk, per N chunks: today `9N`; with the stride `6 + 3N` (the six
Mamba groups once each, plus 2 attention + 1 draft every chunk):

| stride N | today (9N) | with stride (6 + 3N) | density | truncation | dead zone |
|---|---|---|---|---|---|
| 1 | 9 | 9 | 1.00x | 0 | 0 |
| 4 | 36 | 18 | **2.0x** | 3,520 tok | <3,520 tok = 0 hit |
| 8 | 72 | 30 | **2.4x** (their measured 0.417x) | 7,040 tok | <7,040 tok = 0 hit |

Their measured result: CPU tier 115,360 → ~276,900 tokens (0.50x → 1.21x the GPU pool), which is
what moves a hit off the disk tier (a 64 s promotion) onto the CPU tier (1-2 s). Cost is real:
a hit is truncated to the N-chunk boundary, and prefixes shorter than N chunks get **zero**
external hit (MambaSpec window is 1, so `_sliding_window_lookup` finds nothing below the grid).

Ported, gated, and opt-in here: **`patch_mamba_stride.py`** (`RADIANCE_MAMBA_STORE_STRIDE`,
default 1 = inert). `offload-size.py --mamba-stride N` re-sizes the tier for it (stride 4 turns the
3×130k recommendation from 27.9 GiB to 14.0 GiB **modelled** — validate before trusting, the
stride>1 branch is arithmetic from group counts, not a measurement on this box).

#### Correctness: annotate the draft group properly (`patch_offload_eagle_groups.py`)

Our committed `patch_offload_eagle_fallback.py` restricts the scheduler's flag-them-all fallback
to non-Mamba groups, which fixes offload but leaves the **two full-attention groups (6,7)**
flagged as draft groups: `is_eagle_group` still drops their trailing chunk during decode and
shortens the servable prefix by a chunk. The correct fix is upstream's intent — annotate the
group holding the last-registered layer (the draft model's attention, g8). vLLM's annotator
(`_annotate_eagle_groups_deepseek_v4`) exists but is called only on the DeepSeek-V4 branch and
returns early unless a spec carries `model_version == "deepseek_v4"`. `patch_offload_eagle_groups.py`
drops that gate and calls it on the hybrid page-size path, so `eagle_groups == {8}` and the
fallback never fires (expect `EAGLE/MTP draft attention groups [8] detected`). Gated
`RADIANCE_OFFLOAD_EAGLE_GROUPS` (default 0 = today's behaviour). Upstream note in the function:
`# FIXME(yifan): avoid/generalize this hacky check.`; PR #52047 is the open attempt.

#### Correctness: cached ≠ cold in the low bits — and fp16 SSM

Two independent sources of low-bit divergence between a cached turn and a cold prefill, both
found and fixed by the sibling build with bit-identical serving as the standard:

- **Last-block alignment.** `Scheduler._mamba_block_aligned_split` exempts the prompt's final
  chunk from block alignment. If a prompt ends within ~400 tokens past a block boundary, the
  final full block is computed in a chunk whose GEMM tiling differs from a cold prefill's, so
  fp8 KV turns some low-bit differences into whole-ulp flips. Their `patch_sched_align_last_block.py`
  stops the final chunk at its last full block boundary (one extra ≤400-token step on ~25% of
  prompts) — an upstream candidate.
- **fp16 temporal state.** Our TP=1 single-GPU profile runs **fp16 SSM (conv bf16)**
  (`serve-mxfp4.sh:704`). The sibling build explicitly does **not** adopt that for a tiered
  deployment: an fp16 recurrent state computed along different chunk boundaries rounds
  differently, so a restored state and a cold one diverge; they use **fp32 ssm** because
  "a tier can't round-trip a 16-bit temporal state". This is a real trade: fp16 halves the page
  and doubles concurrency (3→6 at 880), but it weakens cached-vs-cold exactness.

Our offload validation so far is 192 greedy tokens byte-identical at 90k — necessary, not
sufficient (the sibling's `turnbench` gate is 3×7 turns token- *and* logprob-identical, and they
found 0/10 divergence alone vs 10/10 under one co-tenant, so it must run on a quiet card). If
bit-identical tier serving matters here, the levers are `sched_align_last_block` + fp32 ssm, and
the gate is a turnbench-style comparison, not a single greedy probe.

#### Speed

- **Host<->GPU DMA is NOT the bottleneck on the fast card.** Measured with the real op: a single
  64 MB `swap_blocks_batch` runs at **28 GB/s**, and even 8 x 8 MB *serialised* on one stream holds
  **27 GB/s** -- so neither the DMA engine, nor descriptor count (4096 x 16 KB = 26.8 GB/s), nor the
  worker's per-direction serialisation (`gpu_worker.py:376`) is the limit.

  **The "~1 s per-restore overhead" is not offload overhead.** Measured with the connector's own
  metrics on a 5x60k CPU restore: 10.22 GB moved in **0.367 s of load_time = 27.8 GB/s**, and the
  `load_size` histogram shows **one load job per request** (5 jobs, ~2.04 GB each, ~73 ms each).
  So the transfer is ~7% of the 5.0 s wall. On the *same server*, a 2x60k reuse that fits the GPU
  pool gave **1.02 s TTFT with `CPU->GPU delta = 0.000 GB`** -- zero bytes moved, a pure GPU-cache
  hit -- versus the offload restore's 0.98 s. The ~1 s is baseline long-prompt full-hit latency
  (block hashing + scheduling + the first decode step), shared by all prefix-cache hits; offload's
  marginal cost is the ~73 ms load and is not separable from it. There is no offload staging
  overhead to fix; reducing that ~1 s means optimising the baseline hit path, a different problem.
  The real costs are the disk leg and the chipset-attached card (see above).
- **ROCm batch-copy limit.** One `swap_blocks_batch` call with >=16384 descriptors faults in
  `__amd_rocclr_copyBufferBatch` ("Page not present", 16384 x 4 KB reproduced). Per-chunk offload
  jobs stay well under it, but any "merge into one giant transfer" optimisation must respect it.
- **Disk tier is device-bound** (~1.0 GB/s on their NVMe, 228 MB/s on SATA → ~29k tok/s vs 2k for
  recompute). The lever that matters is **fs read fan-out**: upstream submits one task per job and
  the tier serialises every 27 MB block file at queue depth 1 — their `patch_offload_fs_fanout.py`
  (a port of upstream PR #49225) splits a job across the tier's 8r/4w thread pool and is what
  turns a 64.26 s promotion into something usable. PR #54327 would retire their external reaper.
- **Reconcile re-ask** (`patch_reconcile_reask.py`): stock drops a GPU attention hit that has no
  Mamba state and recomputes the whole prompt (~270 s observed); the patch re-asks the tier from
  the lowered boundary.
- **Retention under ARC** (`patch_swa_align_touch.py`): `_touch` refreshes all attention chunks
  but only the newest sliding-window chunks, so a conversation's older Mamba snapshots are evicted
  while their attention survives (`evict_skew`: 4 of 5 big losses lost every g0 snapshot). Touching
  every group, ordered head-to-tail in one call, is what made ARC useful in their fill-then-cycling
  workload. Our ARC-vs-LRU test showed no difference because the *head* was lost, not the tail —
  the head cap is the fix for that failure mode; touch-all is the fix for the skew mode.

#### Disk tier: configuration, not code

vLLM 0.27.1 registers `TieringOffloadingSpec` with secondary tiers (`tiering/spec.py`). Recipe:

```json
{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
 "kv_load_failure_policy":"recompute",
 "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec",
   "offload_prompt_only":false,"eviction_policy":"lru",
   "secondary_tiers":[{"type":"fs","root_dir":"/kvcache","n_read_threads":8,"n_write_threads":4}]}}
```

- The CPU primary tier is `--kv-offloading-size <GiB>` (absolute GiB); `cpu_bytes_to_use` in
  extra_config is overwritten by it, so do not put it in the JSON.
- `spec_name` is required to reach the fs tier; the default spec raises on `store_threshold>=2`.
- `offload_prompt_only=false` stores generated tokens too (upstream default is prompt-only).
- **The fs tier has no capacity, quota, TTL, or eviction hook** — an external reaper is
  mandatory, or the filesystem fills. A corrupt/reaped block must be made non-fatal (their
  `patch_fs_failed_load.py`, an upstream candidate): stock `offloading/worker.py:361` is a bare
  `assert transfer_result.success`, and `kv_load_failure_policy` is **inert** for this connector.
- `PYTHONHASHSEED` must be pinned or the block filenames change between boots and orphan the cache.
- `blocks_per_chunk` coarsens the offload grid (and the dead zone); change it only with a fresh
  on-disk subdirectory, because the on-disk geometry is not rewritten.

#### Measured results on the box (2026-09-23)

`bench_offload.py`, clean boot per row, `DUMMY=1 FAST=1` (weights dummy, no cudagraph), KV pool
8.16 GiB, CPU tier 24 GiB. `cached=None` throughout, so "CPU-restore" means the prefix came back
from the CPU tier. TTFT is dominated by re-prefilling any uncached remainder (~0.365 s per 1k
tokens at enforce-eager), so a partial restore is not free.

| config | workload | result |
|---|---|---|
| annotation [8], stride 1, no cap | 5x60k | all 5 restore, 1.11 s each, 10.22 GB promoted |
| annotation [8], stride 1, no cap | 6x60k | all 6 restore, 1.13 s each, 12.27 GB |
| annotation [8], stride 4, no cap | 6x60k | 5/6 (oldest loses its head), 10.25 GB |
| annotation [8], stride 4, no cap | 7x60k, 8x60k | 0 restores (all re-prefill) |
| annotation [8], stride 1, no cap | 8x60k | 0 restores (matches the recorded baseline) |
| + touch-order | 8x60k | still 0 restores |
| + head cap 36,960 | 8x60k | all 8 restore, 10.49 s each, sum 83.9 s, 10.13 GB |
| + head cap 36,960 + touch-order | 8x60k | identical: 10.50 s, sum 84.0 s, 10.13 GB |
| auto head cap (`"max_offload_tokens":"auto"` -> 36,960) | 8x60k | all 8 restore, sum 84.0 s, 10.13 GB |
| **fs tier, 12 GiB CPU, NO cap** | 8x60k | **all 8 restore, 8.55 s each, sum 68.9 s, 16.36 GB (~435k tokens)** |

The fs row is the important one: it is the same workload that restores **nothing** on CPU-only
without a cap, and it restores *everything* once evicted blocks have somewhere to go. Two
conclusions follow:

- **The fs tier makes over-subscription non-fatal, without a per-request head cap.** 16.36 GB / ~435k tokens were
  promoted from a 12 GiB CPU tier -- 2.8x its capacity -- because the overflow was on disk and
  still loadable. Eviction is no longer loss: a block is recovered from fs, or (if reaped)
  recomputed via the failed-load fix, but never "restore nothing". That is the safety net the head
  cap was standing in for.
- **The `auto` head cap and the fs tier are mutually exclusive in effect.** `auto` sizes each prefix to
  `tier_tokens / max_num_seqs`, so 8 prefixes fill the CPU tier *exactly* (8 x cap ~= tier) and the
  fs tier is never consulted -- 12 GiB, cap 18,480, and the fs tier would sit idle. Pick one:
  `auto` cap for a CPU-only deployment that must survive over-subscription deterministically, or an
  fs tier (no cap) for a larger effective pool that survives eviction. Do not enable both and
  expect the disk to help.

Conclusions (these override the expectations in the plan below):

- **The head cap is the mechanism that makes partial reuse work.** 8x60k is 480k tokens against a
  ~360k-token tier (measured: 1787 blocks x 880 = 174k by the `num_blocks//groups` model, ~360k in
  practice). Without a cap the oldest prefix loses its head, `_lookup` returns 0 for the request,
  and *every* prefix restores nothing. With a cap each prefix keeps a contiguous head and restores
  it, prefilling only the remainder. This, not eviction ordering, is what "always works".
- **Auto cap is within ~7% of hand-tuning** (36,960 vs 39,600) and removes the manual step. Its
  model (`num_blocks // offload_groups * tpc`) under-reports the measured tier by ~1.8x on this
  model, carried as `RADIANCE_OFFLOAD_CAP_RATIO` (default 1.8) rather than assumed away.
- **Touch-order (hunk 2) measured NEUTRAL: identical to 0.1%.** A cap that fits means *no eviction
  happens at all*, so eviction order cannot matter. It was not the enabler. Kept gated off; it can
  only be judged under real eviction pressure (a disk tier, or a tier smaller than the cap sum).
- **Mamba stride 4 is a LOSS at this tier.** Every hit is truncated to the 4-chunk grid (dead zone
  3,520 tokens) and 6x60k fell from 6/6 to 5/6. It pays only when the tier is the binding
  constraint and the alternative is a disk promotion. Keep `RADIANCE_MAMBA_STORE_STRIDE=1` here.
- **Correct eagle annotation [8] is a real win**: same workload, more promoted (10.22 GB vs the
  recorded 9.79 GB at [6,7,8]) and 1.11 s vs 2.27 s per restore.
- **Cost of the cap**: a workload that FITS (5x60k = 300k tokens) restores fully in 1.11 s uncapped;
  capped at 36,960 each it would restore only heads and prefill the rest (~10 s). Size the tier so
  the working set fits (`offload-size.py`); use the cap only when it cannot.

Not re-validated with real weights: cached-vs-cold exactness of a capped PARTIAL restore (head from
CPU + prefilled remainder). The full-restore path was validated earlier (192 greedy tokens
byte-identical at 90k).

#### Base vs Mode B, end to end (2026-09-23)

True base (no `--kv-transfer-config` at all) vs Mode B (12 GiB CPU + shared fs disk, no head cap),
same shape (KV pool 8.16 GiB / 228k tokens, seqs 8, chunk 4096, real weights, greedy).

Reuse benchmark: 8 distinct 60k-token prefixes, round 2 reuses them in order. The working set is
480k tokens -- more than twice the GPU pool -- so base cannot hold it:

| | base | Mode B |
|---|---|---|
| round 1 fill | 203.6 s | 205.0 s (identical) |
| round 2, per prefix | 25.7 s **re-prefill** x8 | 8.57 s CPU-restore x8 |
| round 2 sum TTFT | 205.75 s | **68.81 s** |
| round 2 wall | 206.6 s | **69.7 s** |
| promoted | -- | 16.36 GB (~435k tokens) |
| fs blocks written | -- | 2760 files / 38 GB |

**~3.0x faster reuse.** Base re-prefills *every* prefix: at 480k tokens the GPU pool evicts each
one before its reuse turn, and reuse-in-order cascades. Mode B restores all eight from CPU + disk.

Normal usage is unchanged -- offload is inert until the GPU pool evicts (one conversation's 60k
prefix fits, so both hit the GPU cache):

| workload | base | Mode B |
|---|---|---|
| C1 / C2 / C4 / C8 (short prompt) | 83.6 / 150.4 / 237.6 / 357.9 tok/s | 85.0 / 152.8 / 238.9 / 362.0 |
| multi-turn 60k, turns 1/2/3 | 25.62 / 0.82 / 0.82 s | 25.67 / 0.82 / 0.82 s |

Caveat: the 8.57 s Mode B restore is mostly re-prefill of the uncached head plus disk promotion on
this **SATA SSD**; a whole prefix resident in RAM restores in ~1 s. RAM is the speed, the disk is
the net.

#### Prioritized plan

1. **Head cap (`"max_offload_tokens":"auto"`)** — validated, and the only lever that made the
   over-subscribed 8x60k case restore anything. Enable where the working set can exceed the tier;
   size the tier first (`offload-size.py`) and use the cap as the guarantee, not the default.
2. **Correct eagle annotation** (`RADIANCE_OFFLOAD_EAGLE_GROUPS=1`) — validated (more prefix
   restored, ~2x faster restores). Make it the default and retire the fallback patch.
3. **Re-validate exactness with real weights** under a capped partial restore (head from CPU +
   prefilled remainder), and decide the bit-identical question (`sched_align_last_block` + fp32 ssm)
   if it matters.
4. **Disk tier** — add the fs secondary tier + reaper + `patch_offload_fs_fanout.py` and
   `patch_fs_failed_load.py`; then re-test the Mamba stride and touch-order, which only pay once
   eviction (or a slow tier) is actually in play.
5. **Mamba stride / touch-order** — keep off here. Re-evaluate with a disk tier or a tier smaller
   than the cap sum, which is the only regime where they can matter.

## Productionizing the CPU/disk KV offload (2026-09-23)

### Where it stands

The offload was validated only in a bench harness. The deployment had **no offload at all** --
`coolify-compose-2gpu.yml` passed no `--kv-transfer-config`, and no `KV_OFFLOAD_*` env existed. It
is now wired into both instances, gated by `KV_OFFLOAD_GIB` (default `0` = exactly the old
behaviour), with `max_offload_tokens: "auto"`.

Facts that constrain the design:

- `--kv-offloading-size N` sets `cpu_bytes_to_use = N GiB` *and* selects `OffloadingConnector`
  (`vllm/config/vllm.py:910`); other `kv_connector_extra_config` keys still come from
  `--kv-transfer-config`, and are merged.
- `--kv-offloading-backend native` is the only usable backend: the image has no `lmcache`, `nixl`
  or `mooncake` (checked), so no LMCache/P2P/S3 tier without a rebuild.
- Each instance's CPU tier is **private**. Both containers run `ipc: host`, so they share the host
  `/dev/shm` (32 G) pinned-memory pool: the two tiers must **sum to <= ~28 G**. Host RAM is 62 G
  (53 G available), so raising `/dev/shm` is possible but competes with page cache.
- Measured CPU-tier density is ~15k tokens/GiB (24 GiB held ~360k tokens), against a 228k-token
  GPU pool. A 12 GiB per-instance tier therefore holds ~180k tokens -- less than one instance's
  worst-case 8x60k = 480k. Partial reuse is the normal case, not the exception.

### The three decisions that determine whether it helps

1. **Routing affinity.** A prefix cache is per instance. HAProxy was `balance leastconn` with no
   affinity, so a returning conversation lands on either replica and misses the other's tier about
   half the time -- halving the benefit before anything else matters. Cheapest fixes first:
   - HAProxy `stick on hdr(x-session-id)` (added): a client that sends a stable header pins to one
     instance; clients that send nothing keep plain leastconn. The header is a client contract.
   - vLLM Router (`consistent_hash` on the OpenAI `user` field, or `cache_aware` with a radix
     tree) -- a real KV/cache-aware router.
   - llm-d EPP `prefix-cache-affinity-filter`, or the vLLM production-stack load-aware router.
   - Or remove the need entirely with a **shared** tier (fs, below), at disk speed.
2. **Tier sizing** (`offload-size.py`). The upstream guide: for CPU-only, size the tier larger than
   the aggregate GPU KV cache, else it just mirrors what the GPU already holds. Here the shm split
   caps each instance below that, which is exactly why the cap matters.
3. **The head cap** (`max_offload_tokens`, wired as `"auto"`). Caps each request from token 0 to a
   fair share, so an over-subscribed cycle restores a contiguous partial prefix instead of losing
   the head and restoring nothing (measured: 8x60k 0 -> all 8 restore). This is the lever that
   makes it "always work" rather than all-or-nothing.

### Tier options beyond CPU

| tier | shared across the 2 instances | survives restart | needs | measured speed |
|---|---|---|---|---|
| CPU (`cpu_bytes_to_use`) | no (per instance) | no | nothing | ~28 GB/s measured (CPU-attached card); 0.8 GB/s if chipset-attached |
| fs (`secondary_tiers[].type="fs"`) | **yes -- shared `root_dir`, deterministic `NONE_HASH`** | yes | fast SSD/NVMe + `ops/kvcache-reap.sh` (the eviction policy; `KVCACHE_MAX_GIB` for a byte cap); failed-load fix is in the launcher | ~1 GB/s (~29k tok/s) |
| p2p (`type="p2p"`) | yes, over RDMA | no | `nixl` (absent) | RDMA |
| obj (S3) | yes | yes | `nixl` OBJ + bucket (absent) | network |
| LMCache | yes (standalone server) | yes | image rebuild (`lmcache` absent) | varies |

**If cross-instance hits or restart survival are wanted, the fs tier is the next step** -- point
both instances at one `root_dir` (`KV_OFFLOAD_DISK_DIR`) and a prefix cached by either is loadable
by both, which also neutralises the affinity problem (at ~1 GB/s instead of ~9). Its two historical
prerequisites are now solved in-repo: the **failed-load hang** by `patch_offload_fs_tier.py`
(propagated by the launcher when the disk dir is set) and **the missing eviction policy** by
`ops/kvcache-reap.sh`, which an operator must install and run (a systemd timer is provided). The
reaper is not optional: the tier never deletes on its own.

### Sizing the disk tier, and bounding it (GiB)

Unlike the CPU tier, the fs tier has no size parameter -- it never refuses a write and never
deletes, so its size is simply whatever the filesystem offers. Size it with a rule of thumb: it
holds ~70 KB per stored token at stride 1 (the same density as the CPU tier), so **1 GiB ~= 15k
tokens** and **1 TB ~= 14M tokens** (~230 x 60k prefixes). The CPU tier is a fixed, pinned budget;
the disk is effectively unbounded and slower (~1 GB/s vs ~9 GB/s).

Two complementary ways to bound it:

- **Soft cap on a shared disk:** `KVCACHE_MAX_GIB=N` (or `KVCACHE_MAX_MB`) in the reaper. It
  measures the KV *directory* (not the whole filesystem) and deletes oldest-first until the
  directory is back under N, so the offload can share a disk with other data and still be capped.
  Enforced each timer tick (5 min), so it can overshoot briefly.
- **Hard, reserved bound:** `ops/make-kvcache-volume.sh` creates a dedicated fixed-size ext4
  loopback volume (default 200 GiB), `fallocate`d so the host cannot give those blocks to anything
  else, and mounts it at the disk dir. The tier physically cannot exceed it and nothing else can
  take it -- this is the "reserve N GiB" case. Set `KVCACHE_MAX_GIB` ~10 GiB below the volume so
  normal operation never reaches ENOSPC.

A full volume is not fatal (a failed store drops that block -> miss -> recompute) but it silently
stops caching, so bound it deliberately.

### Robustness gaps to close before the fs tier

- **The fs tier is made safe in-repo** by `patch_offload_fs_tier.py` (a port of the sibling's
  `patch_offload_fs_fanout.py` / `patch_lookup_invalidate.py` / `patch_fs_failed_load.py`, applied
  in the launcher, inert unless its gates are set):
  - **I/O fan-out.** Both submit paths ended in `enqueue_*(job_id, 1, [task])`, so the 8 read / 4
    write threads were never used and a promotion read ~200 x 27 MB files at queue depth 1 -- that
    is the 64 s promotion. Jobs are now split across the pool (upstream PR #49225).
  - **Stale `absent` verdict.** A request that stored a block kept getting `absent` for it and
    recomputed. `invalidate()` drops the stale verdict when a store lands.
  - **Failed load.** Correction to an earlier note in this file: for `TieringOffloadingSpec` a
    failed fs read fails the fs->CPU *promotion* (`cpu/manager.py complete_store(success=False)`
    drops the half-written block), so `offloading/worker.py:361` is **not** reached and the engine
    does **not** crash. The real failure is a **hung request** -- nothing told the lookup cache, so
    it re-promoted the same missing file forever (measured: 294 failed reads in 240 s). `forget()`
    drops the verdict, turning the hang into one recompute. This is what makes reaping safe.
- **The reaper is still mandatory**: the fs tier has no capacity/quota/TTL and no eviction hook.
  `ops/kvcache-reap.sh` (+ `.service`/`.timer`) *is* the eviction policy -- oldest-mtime-first, with
  a hard `KVCACHE_MIN_AGE_MIN` floor (default 90 min) and an emergency stage that crosses the floor
  only when the volume is nearly full, which is safe now that a deleted young block costs a
  recompute rather than a hang.
- `kv_load_failure_policy="recompute"` is **inert** for this connector; do not rely on it.
- `PYTHONHASHSEED` **must** be pinned whenever a `root_dir` is shared (or reuse across a restart is
  wanted), for **any** prefix-hash algo -- not just `xxhash`. `init_none_hash` seeds the whole
  block-hash chain from `os.urandom(32)` when the variable is unset (`kv_cache_utils.py`), so
  without it each process computes different block hashes for the same tokens and the fs tier can
  neither dedup nor read another instance's blocks. The launcher and the compose both set
  `PYTHONHASHSEED=0`; it must be the **same value on both instances**.

### Correctness

The sibling's bit-identical claim holds **serially only** -- concurrent batches and spec decode
perturb numerics regardless of any cache, and this deployment runs `--max-num-seqs 8`. The offload
does not add exactness risk beyond the batch-shape variance already present; it inherits it.
**In operation this is academic here**: divergent turns do not occur at a rate worth acting on, so
the two cache-attributable contributors (last-block alignment, fp16 SSM) are *not* worth their
cost -- no `patch_sched_align_last_block`, no fp32 SSM. They stay documented (below) only for a
future deployment that actually needs token-identical output and would have to serialise to get it.

### Recommended production config (pick a mode)

Common to both: `KV_OFFLOAD_GIB=12` on **both** instances (24 G total <= 28 G shm, comfortable
margin; up to 14 each = 28 G is the aggressive max, leaving only 4 G of the 32 G shm). Both cards
are Gen5 x16 here, so the RAM tier is the fast path on either instance. Also common:
`offload_prompt_only: true` (a turn's decode tokens reappear as the next turn's prompt),
`eviction_policy: lru`, and a stable `x-session-id` from the client so a conversation stays on one
instance.

- **Mode A -- CPU only, deterministic partial reuse.** `max_offload_tokens: "auto"`. Fits the CPU
  tier exactly, so over-subscription restores a contiguous partial prefix (never nothing) and the
  fs tier is not used. Best when the disk is slow or unwanted.
- **Mode B -- CPU + shared fs (recommended here).** `KV_OFFLOAD_DISK_DIR` on one shared SSD/NVMe
  dir, and **no per-request head cap** (`max_offload_tokens` is left unset). Note the separate disk
  byte budget (`KVCACHE_MAX_GIB` / a reserved volume) is *still required* -- "no cap" never means an
  unbounded disk. The disk holds the overflow and evicted prefixes are still restored (measured:
  16.36 GB / ~435k tokens from a 12 GiB CPU tier, all 8 prefixes). Requires `ops/kvcache-reap.sh`
  installed and running, and a bound -- `KVCACHE_MAX_GIB` for a soft cap on a shared disk, or
  `ops/make-kvcache-volume.sh` for a reserved fixed-size volume. The launcher turns on the
  fanout/invalidate/failed-load gates for you. Slower per restore than CPU (disk ~1 GB/s) but never
  loses a prefix to eviction.
- Want more headroom without disk: raise `/dev/shm` (53 G available) or trade GPU pool for a larger
  CPU tier via `offload-size.py`.

### Runbook

- Enable: set `KV_OFFLOAD_GIB`, redeploy. Confirm in the log: `EAGLE/MTP draft attention groups
  [8] detected`, `[radiance] auto head cap: ... cap=N`, and `--kv-offloading-size` in the
  `non-default args` line.
- Watch: `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` (restore volume),
  `kv_offload_lookup_sync_delay_seconds`, `kv_offload_cpu_cache_usage_perc` (saturating = transfers
  dropped); panels already exist in the observability stack.
- Validate: a 2-turn run with a 60k+ shared prefix -- reuse must show a CPU-restore TTFT far below
  a cold prefill (~1-10 s vs ~22 s).
- Enable the shared fs tier (Mode B) with one host command:
  `sudo ./ops/setup-kv-offload.sh [--ram-gib 12] [--disk-gib 100] [--volume-gib N] [--env-file P]`.
  It prepares the disk-tier directory (or a reserved fixed-size volume), installs and starts the
  mandatory reaper with the GiB cap, writes the deployment env if asked, and prints the two Coolify
  variables to set on **both** services (`KV_OFFLOAD_GIB`, `KV_OFFLOAD_DISK_HOST_DIR`); the compose
  mounts that host dir at `/kvcache` and `KV_OFFLOAD_DISK_DIR` defaults to `/kvcache/blocks`, so no
  container-path editing is needed and no per-request head cap is required. Confirm the boot log
  shows `[radiance] fs tier applied -- ... invalidate=1 failed_load_forget=1` and `fs KV tier ON:
  root_dir=...`, the blocks appear under the host dir, and `journalctl -u kvcache-reap` reports each
  cycle. `--uninstall` reverses it; `--dry-run` previews.
- Roll back: `KV_OFFLOAD_GIB=0` (CPU offload off) or unset `KV_OFFLOAD_DISK_DIR` (disk tier off);
  no code change. **Never** run the fs tier without the reaper.

### Tuning knobs (all Coolify-env, no code change)

Both vllm services read these; a redeploy applies them. They are also published as labels on the
`radiance_serve_info` textfile metric (node-exporter `--collector.textfile`, shared `serve-shape`
volume) and shown in the Grafana row **"Offload tuning A/B (eviction / async / fan-out)"**:

- `KV_OFFLOAD_EVICTION_POLICY` (default `lru`): primary-tier policy; `arc` (Adaptive Replacement
  Cache) balances recency + frequency and can lift the RAM hit rate on repeat-prefix traffic. Judge
  it with **Offload hit rate %** (`vllm:external_prefix_cache_hits_total` /
  `..._queries_total` -- connector tokens served from offload / tokens queried) and the TTFT panels
  (RAM ~1 s vs disk ~8.6 s), not a cache-size gauge (RAM occupancy is not observable).
- `VLLM_ASYNC_SCHEDULING` (default `0` = keep `--no-async-scheduling`): `1` passes
  `--async-scheduling` to overlap CPU scheduling with GPU work. Test for regressions against the
  custom R4D/Mamba backends and spec decode before trusting it.
- `KV_OFFLOAD_READ_THREADS` / `KV_OFFLOAD_WRITE_THREADS` (default `32` / `16`) and
  `RADIANCE_FS_FANOUT_MAX` / `RADIANCE_FS_FANOUT_TARGET_MB` (default `32` / `512`): restore
  throughput. A single fs restore is split across up to READ threads by the fan-out patch, so
  fan-out, not the thread count, is what closes the gap to the device ceiling (measured SATA
  O_DIRECT: ~385 MB/s at QD1 vs ~477 MB/s at QD16, ceiling ~480 MB/s; NVMe will use the higher
  defaults).
- `PYTHONHASHSEED=0` (set on both, non-tunable): required for the shared `root_dir` to work at all
  (see Correctness above) and by the P2P tier.

### Async scheduling (why it is OFF, and how to test it without risk)

`VLLM_ASYNC_SCHEDULING=1` swaps vLLM's synchronous step for `AsyncScheduler`, which overlaps the
host's scheduling work for step N+1 with the GPU's execution of step N. It is **off here and stays
off**: the 2026-09-04 re-test (see `serve-mxfp4.sh`, with `patch_async_dynwidth.py` + a per-step
trace) found it *identical* to sync -- 22.35 vs 22.30 ms/step single, 38.0 vs 38.0 ms conc-8,
acceptance byte-identical -- because the GPU is already saturated (209 W cap, ~2.83 GHz) and the
worker CPU chain is only ~8.5 ms with <=1.5 ms/step idle. There is nothing to overlap, so async
buys nothing and only adds failure modes.

**Why it is risky on this stack** (all code-confirmed):

- **Hard gates** (`config/vllm.py:1081+`): with async forced on, vLLM *raises* unless the spec
  method is EAGLE-family (`mtp`/`dflash` qualify), the executor supports it (`uniproc`/`mp` do), and
  `disable_padded_drafter_batch` is **False**. Our unpadded drafter (`disable_padded_drafter_batch:
  true`, the ~+50% single-stream lever) is *rejected* -- so async and the padded drafter are the
  **same switch**.
- **Hybrid/Mamba state**: async spec-decode only syncs `num_accepted_tokens` to the CPU in
  `--mamba-cache-mode align` (`gpu_model_runner.py`: async non-align skips the D2H sync because it
  races). We always run align; a change would silently corrupt accepted-token counts.
- **Spec placeholder/rollback**: `AsyncScheduler` schedules the next step against placeholder draft
  ids and reconciles on the GPU (`num_output_placeholders`, `_update_request_with_output`). Anything
  that inspects drafts on the host in the same step (penalties, bad-words, structured output) rides
  that reconcile path.
- **KV offload**: the connector runs in the scheduler step, and our offload/eagle patches
  (`patch_offload_eagle_fallback/groups`, `patch_offload_fs_tier`, `patch_mamba_stride`,
  `patch_offload_swa_touch`) were validated under **sync** semantics -- a step-ahead scheduler
  changes when blocks are cached/rolled back vs stored. **Unvalidated.**
- **Custom kernels/shapes**: R4D attention, GDN kernels, cudagraph capture and the fusion patches
  assume stable batch shapes; the padded drafter helps, but the dynamic-width cap
  (`patch_async_dynwidth.py`) exists precisely because async changes how the verify width is sized.

**How the risk is removed (guards now in the compose):**

1. The two switches are coupled: `VLLM_ASYNC_SCHEDULING=1` derives
   `disable_padded_drafter_batch=False` (and `0` derives `True`), so the flag can never boot into
   vLLM's hard error.
2. A preflight refuses combinations this stack has not validated and **exits before serving**
   instead of degrading silently: async requires `mamba-cache-mode align`; async + KV offload is
   refused unless `RADIANCE_ASYNC_ALLOW_OFFLOAD=1` is set for a deliberate test.
3. Default is `0` (async disabled), so risk today is zero.

**To test it anyway:** set `KV_OFFLOAD_GIB=0` (or `RADIANCE_ASYNC_ALLOW_OFFLOAD=1` if you must keep
offload), set `VLLM_ASYNC_SCHEDULING=1` on **both** services, redeploy, run the same fixed load, and
compare the Grafana **Offload tuning A/B** row: TTFT/throughput should match sync, `Spec acceptance
%` should be unchanged, and `Preemptions rate` should stay ~0. Roll back with
`VLLM_ASYNC_SCHEDULING=0`.

### Faster iteration on connector changes

The offload scheduler is Python in the engine core, so code changes still need a process restart,
but most of the ~200 s startup is cudagraph capture (weights are only ~24 s, and `torch.compile` is
cached at ~5 s). Use `EXTRA="--enforce-eager"` to skip compile+capture, and for pure
connector/index diagnostics `EXTRA="--enforce-eager --load-format dummy"` to also skip the 20 GiB
weight read (startup ~85-165 s; dummy weights produce garbage output and skip the DFlash buffer,
so use it only for hit/size measurements, not throughput or correctness).


