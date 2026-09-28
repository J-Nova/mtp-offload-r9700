# MTP decode optimization plan (single-card, gfx1201)

Working plan for improving single-stream **MTP** decode on one R9700. Every item is
scoped so it can be run, scored, and checked off. Nothing here is landed; this is a
research/measurement backlog derived from the codebase and the 2026-09-26 full run.

## Target / baseline

- Target: `Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ`, `spec_method=mtp`, `SPEC=8`,
  `MAXSEQS=8`, `MAXLEN=205000`, `CHUNK=4096`, `GPU_UTIL=0.98`, `GPUS=1 TP=1`.
- Baseline (full BetterBench, `bench/overview.json`): **combined decode 80.4 t/s,
  update p50 55.9 ms / p99 62.4 ms, tok/update 4.64**, TTFT p50 88 ms,
  conc-8 313.9 t/s, prefill ~2.4-2.65k t/s.
- Model shape: 64 layers, hidden 5120, ffn 17408, 24 q / 4 kv heads, head_dim 256,
  1-in-4 full attention (hybrid GDN). Weights ~14.3 GB/step (27B x 4.25 bit).

## Diagnosis in one paragraph

The step is **not** launch-bound in the simple sense and **not** a model-size problem.
The decode GEMM alone runs at **95.5% of achievable streaming bandwidth** in isolation,
but the whole step achieves only **~257-280 GB/s of the 640 GB/s peak (~40%)**. The
weight-stream floor for one forward is ~22.3 ms; the measured step is 55.9 ms, so
~33 ms/step is **non-weight overhead**. MTP is autoregressive serial (1 first forward +
up to 7 loop forwards). The dynamic draft controller gates each slot on a **blocking
D2H readback** (`radiance_draft.py:327`) and adds per-slot capture/head launches, which
serializes the loop and idles the GPU between GEMMs. The repo's 35.7 ms/step / 134.8 t/s
TP=1 figure (`PERFORMANCE.md:27`) is the **dflash** path (one graphed block/step), not MTP.

## Cost model (batch 1, SPEC 8)

| Term | Value | Source |
|---|--:|---|
| Weight bytes/step | ~14.3 GB | 27B x 4.25 bit |
| DRAM floor @640 GB/s | ~22.3 ms | spec |
| Achievable streaming | ~574-580 GB/s | `autoround-tests/RESULTS.md:394-405` |
| Measured step | 55.9 ms | `bench/overview.json` |
| Effective step BW | ~257 GB/s (40%) | derived |
| dflash step (same card) | ~35.7 ms (62%) | `PERFORMANCE.md:27` |
| Decode GEMM at M=9 | issue floor ~70 us vs DRAM 41 us | `radiance_mxfp4_fp8.hip:762-767` |

Verify batch M = `MAXSEQS x (SPEC+1)`; at batch 1 / SPEC 8 it is **M=9**, padded to the
16-row cudagraph bucket. `RADIANCE_MXFP4_DECODE_MAX_M` is derived to 128 at this shape,
so M=9 takes the decode kernel (not the A-tiled prefill tile).

---

## Tier A - cheap A/Bs, no code changes (do these first)

Each is one server boot. Score `ms/step`, `tok/update`, acceptance, and combined t/s
together; never from a single number.

- [ ] **A1. int2 verify head under MTP.** `RADIANCE_VERIFY_HEAD=1`. The bf16 verify
  `lm_head` is one 2.02 ms GEMM/step (3.6% wall) at the DRAM roofline; dflash measured
  **+2.9%** combined decode (8/8 categories), conc +2.8/+2.5/+1.6%. Currently forced
  off for MTP (`serve-mxfp4.sh:505`). Risk: acceptance/output under MTP differs - check
  GSM8K or an output-equivalence probe. Effort: low. Ref: `PERFORMANCE.md:33`,
  `radiance_verifyhead.py:3-6`.
- [ ] **A2. libr4d decode kernel.** `RADIANCE_MXFP4_R4D_DECODE_MAX_M=1` (needs
  `WPERM=1`, already on). Routes the decode band to `r4d_gemm_mxfp4a8_nt_m64`. Dark and
  unmeasured. Effort: low. Ref: `serve-mxfp4.sh:1101`, `radiance_mxfp4.py:73-90`.
- [ ] **A3. Decode split-K / BK retune.** Sweep `RADIANCE_MXFP4_DECODE_KS` and
  `_DECODE_BK` over the actual M in {9, 18, 36, 72} (batch 1/2/4/8). The shipped policy
  is hand-fitted to six shapes and the M=9 issue floor is a split-K geometry problem.
  Effort: low. Ref: `radiance_mxfp4_fp8.hip:1088-1116`.
- [ ] **A4. Skinny GEMM `all`.** `RADIANCE_SKINNY_GEMM=all`. Measured **-3.9% step**
  with no acceptance cost when tested, but gated off because bf16-ULP shapes perturb
  drafter acceptance (GDN `in_proj_ba` 28.5 -> 3.6 us, 48x/step). Re-test against the
  MTP drafter; reject if acc/draft falls. Effort: low. Ref: `DOCKERHUB.md:107`.
- [ ] **A5. SPEC depth sweep.** SPEC in {5, 6, 7, 8} (restart per arm). Score
  tokens/update vs ms/step. The first forward is irreducible and the per-position
  acceptance tail is weak (positions 6-8 = 0.22/0.18/0.15), so the modelled optimum is
  6-7. Repo default is 8 (weighted mix favors deep on code/json). Effort: low.
  Ref: `NGRAM-MTP-PLAN.md:135-143`, `README.md:504`.
- [ ] **A6. Draft policy sweep (live).** Via `RADIANCE_DRAFT_KNOB_FILE`: `tau`
  {0.20, 0.28, 0.35}, `strong` {4, 6, 8, 12}, `window`, `cross_req`. Historical
  `tau=0.28` was **+5.3%** over 0.35. One boot for all arms. Effort: low.
  Ref: `DOCKERHUB.md:135`, `radiance_draft.py:130-170`.
- [ ] **A7. Power / clock policy.** The card is power-capped (210 W / ~2.83 GHz) and
  decode **degrades to 42 ms/update if the core boosts past 3.3 GHz** (`README.md:745`).
  Sample clock residency during decode and test a clock cap / undervolt. Two cards
  differ ~12%. Effort: low-med. Ref: `PERFORMANCE.md:826`.
- [ ] **A8. NUMA bind.** `RADIANCE_NUMA_BIND=auto` / `--numa-bind`. Documented no-op on
  single-socket; confirm on this host. Effort: trivial. Ref: `DOCKERHUB.md:138`.

## Tier B - code / structural (the real headroom)

- [ ] **B1. Device-side draft gate / single readback (top lever).** Remove the per-slot
  blocking `packed.cpu().numpy()` decision (`radiance_draft.py:327`) and fold the matcher
  and postprocess readbacks (`:504`, `:534`). Keep the confidence gate on-device, or do
  one readback at end-of-loop, so the GPU can run the draft loop ahead. This is the
  inferred root cause of the 40% effective bandwidth. Effort: high. Target: step <=45 ms.
  Ref: `radiance_draft.py:327`.
- [ ] **B2. Fuse the per-slot controller launches.** capture `_cap_s1`+`_cap_s2_local`
  (2 x N) into the int2-head epilogue; fuse the 3 matcher kernels
  (`_match_scan`/`_match_top2`/`_match_gather`); drop the per-slot full-248320
  `.argmax` (`radiance_draft.py:221`). ~9-15 extra launches/step today. Effort: med.
  Ref: `radiance_draft_gpu.py:214-302`.
- [ ] **B3. Overlap MTP loop with verify.** Explore issuing the next draft forward while
  the verify/sampler of the previous step runs (graph the MTP layer; reduce host-side
  batch prep on the critical path). Effort: high. Ref: `vLLM:llm_base_proposer.py:682`.
- [ ] **B4. Sampler fast path.** At temp 0.7 the sampler materializes FP32 target logits
  (copy + min_p/top_k/top_p/penalty kernels) every step. Not valid to force greedy for
  fidelity, but check whether the FP32 copy and unused param kernels can be elided.
  Effort: med. Ref: `vLLM:v1/worker/gpu/sample/sampler.py:158-159`.
- [ ] **B5. DECODE tile geometry for M<=16.** The decode kernel is issue-bound at tiny
  M. Explore DTM/split-K/`EFFAST` variants explicitly for M=9 (the current floor is
  ~70 us issue vs 41 us DRAM). Effort: high. Ref: `radiance_mxfp4_fp8.hip:762-796`.

## Tier C - validate the premise / alternatives

- [ ] **C1. dflash vs mtp, identical shape.** Confirms the ~1.6x is the drafter. If
  MTP is not a hard requirement, this is the single biggest win and already in the repo
  (measured 134.8 t/s TP=1). Effort: low. Ref: `PERFORMANCE.md:27`.
- [ ] **C2. blend-OCP vs native MXFP4 checkpoint.** Same bench, to rule out the
  `compressed-tensors mxfp4-pack-quantized` backend as a second-order cause. Same
  architecture and ~same 19.8 GB footprint, so any step delta is kernel/acceptance.
  Effort: low.
- [ ] **C3. n-gram tail-rate confirmation.** The 205k run answered the open checkbox:
  **~14% of draft tokens are n-gram tails, ~13% of rows matched**. Record it and confirm
  no prose regression. Effort: trivial. Ref: `NGRAM-MTP-PLAN.md:168,170`.

---

## Measurement protocol

- Harness: `bench-quick-ab.sh` / `bench-quick.py` (MTP probe), thinking **on and off**.
- Always report together: **ms/step, tokens/update, accepted/draft, combined t/s**.
  A single-stream t/s probe cannot resolve the policy gates
  (`NGRAM-MTP-PLAN.md:145-153`).
- One card only: `GPUS=1 TP=1`; keep `CHUNK=4096`, `MAXSEQS=8`, `MAXLEN=205000`.
- **Runner gate (2026-09-27):** `SPEC_METHOD=mtp` runs **V1** (the radiance controller,
  `radiance_draft._local_draft`, the int2 head and every `RADIANCE_DRAFT_*` knob are
  live); `SPEC_METHOD=dflash` forces **V2** (`vllm.py:631 _is_dflash2_draft`), where all
  of that is inert. Any MTP-path A/B must use `mtp`; a `dflash` run cannot exercise them.
- Live controller A/B: `RADIANCE_DRAFT_KNOB_FILE` (re-read each step, no restart);
  SPEC/boot-level knobs need a restart.
- Warmup 3 + 20 passes per category minimum; full BetterBench for a final number.
- Record GPU busy + clock residency during the run (power cap is a real variable).
- Reject any arm that loses acceptance on code/json/file_edit or output equivalence.

## Do NOT repeat (measured neutral or worse in this repo)

- Lazy GDN snapshots (`RADIANCE_GDN_LAZY`): decode parity, corrupts multi-turn chat.
- Rotation stream 3: 25.10 vs 24.19 ms/step, deadlocks.
- Async scheduling: identical (GPU saturated); coupled to rejecting unpadded drafter.
- Custom-op RMSNorm+quant: more launches than the plain-torch version.
- Prefill epilogue prefetch: 828 vs 726 us at M=8192.
- `COMPILE_SIZES`: accept/draft 1.837 vs 2.069.
- `RADIANCE_GDN_STRIDED_GATES`, `GDN_EMPTY_OUT`, `COOP_RED`: neutral.

## Pessimistic impact

Baseline **80.4 t/s / 55.9 ms**. Conservative (only low-risk A/Bs land, structural
work does not):

| Scope | Pessimistic delta | Result |
|---|--:|--:|
| A1-A8 all tried, most neutral, 2-3 land small | **+3 to +6%** | ~83-85 t/s |
| + B2/B4 partial fusion | **+8 to +12%** | ~87-90 t/s |
| + B1 device-side gate (unproven) | **+20 to +35%** | ~96-108 t/s |
| Switch to dflash (C1) | **~+60%** | ~130 t/s |

So the honest pessimistic floor for this plan is **~+5-10% (~84-88 t/s)** if only the
cheap items and modest fusion land; the large MTP-specific upside requires B1, and the
largest single win is C1 (dflash), which is a drafter choice, not a tuning item.

## Progress log

| ID | arm | knob/change | ms/step | tok/upd | combined t/s | acc/draft | verdict |
|---|---|---|---|---|---|--:|---|
| base | card-0 baseline | default, tau=0.20 | 50.4-56.0 / 53.3-58.0 | 1.73-6.21 / 2.35-4.22 | 85.9-93.0 / 74.0 | prose 1.73-1.87, json 5.73-6.31 | reference (2 boots; code-cat n-gram variance) |
| base1 | **card-1 baseline** | default, tau=0.20, x16 link | 49.4-54.7 / 52.7-58.4 | 1.84-7.02 / 1.83-4.05 | 93.8 / 73.2 | prose 1.835/1.826, json 5.756/3.861 | target baseline (card 1 freed 2026-09-27); ~4-5% faster/step than card 0 on prose |
| A1 | verify head | `RADIANCE_VERIFY_HEAD=1` | 50.4-56.4 / 53.3-59.1 | 1.76-7.03 / 1.97-4.27 | 88.5 / 73.4 | prose 1.758/1.968, json 5.417/4.163 | **reject (neutral)**; vhead=1 confirmed, JIT 12=base, prose ms/step 55.95 vs 55.99 |
| A2 | r4d decode | `RADIANCE_MXFP4_R4D_DECODE_MAX_M=1` | 49.5-56.3 / 53.8-59.5 | 1.56-7.16 / 2.13-4.67 | 86.6 / 77.3 | prose 1.558/2.126, json 5.490/4.671 | **reject (no-op)**; banner: `r4d has no mxfp4a8 gemm_nt; decode stays on ours` |
| A3 | split-K/BK | `_DECODE_KS=1` / `=4` (BK inert at tm=1) | ks1 52.3-58.3 / 54.5-60.7; ks4 50.1-56.6 / 53.9-59.4 | ks1 1.76-7.16 / 2.40-3.93; ks4 1.94-7.16 / 2.32-4.36 | ks1 86.9 / 75.1; ks4 91.9 / 77.5 | ks1 1.759/2.403; ks4 1.937/2.320 | **reject**; both inside 1-4% run spread, no cell beats shipped; `_DECODE_BK` is inert for M=9 (tm=1) |
| A4 | skinny all | `RADIANCE_SKINNY_GEMM=all` | 50.7-55.6 / 54.7-59.1 | 1.59-7.13 / 2.28-4.18 | 89.7 / 72.7 | prose 1.591/2.284, json 6.556/4.182 | **reject (neutral)**; mean ms/step 52.86 vs base2 52.94; prose gain offsets echo loss; DFlash's -3.9% not reproduced under MTP |
| A5 | SPEC depth | 5 / 6 / 7 / 8 | s5 53.9-69.2 / 68.6; s6 56.3-57.0 / 56.8-57.2; s7 59.1-60.6 / 60.1-71.3; s8 50.4-56.0 / 53.3-58.0 | s5 1.89-5.00 / 1.96-4.26; s6 1.53-6.00 / 2.18-4.72; s7 1.75-7.00 / 1.91-5.73; s8 1.73-6.21 / 2.35-4.22 | s5 70.8 / 60.3; s6 83.5 / 70.2; s7 88.1 / 65.2; s8 85.9-93.0 / 74.0 | s5 echo 5.000; s6 echo 6.000; s7 echo 7.000 | **reject**; keep SPEC=8 (best off & on); modelled 6-7 optimum does not hold on card 0 |
| A6 | draft policy | `RADIANCE_DRAFT_KNOB_FILE` tau/strong/cross/window | flat ~50-56 both modes across ALL arms | acceptance-luck only | confirm interleaved: off tau020 94.4/89.0 vs tau014 90.3/89.4; on tau020 70.8/75.5 vs tau014 76.6/76.4 | tau014 prose 1.86-2.46 | **reject**; tau014 +4.6% on-mode but -2% off-mode, tau028/strong/window/cross neutral; policy does NOT move ms/step (per-step cost is the serial gate, not draft depth) |
| A7 | power | clock residency / cap | decode 49.4-56.1 (same) | unchanged | 86.2 | prose 1.976 | **reject (no-op)**; SCLK mean 2658 max 2833 MHz (never past 3.3 GHz knee), power mean 203.8 / max 217 W; cap interfaces unsupported/locked (perfdeterminism invalid, setsrange needs out-of-spec consent, reset denied). Perf level restored to auto |
| A8 | numa | `--numa-bind` | n/a | n/a | n/a | n/a | **reject (no-op)**; host is 1 socket / 1 NUMA node, launcher wires no NUMA path |
| B1 | device gate | `RADIANCE_DRAFT_DEVICE_GATE=1` (torch policy + lagged async readback) | **60.1-61.2 / 59.8-61.4** vs base1 49.4-54.7 / 52.7-58.4 | 1.71-7.13 / 2.33-4.46 | **78.9 / 67.8** vs base1 93.8 / 73.2 | prose 1.708/2.330 | **reject (regression -16% off / -7% on)**; clean boot, device_gate=True, JIT 12=base; per-slot on-device elementwise kernels + `.all()` + pinned copy cost more than the small x16 D2H it removed. torch policy proven bit-equivalent to numpy (0 mismatches over 108k cases, /tmp/kilo/test_gate.py). Code reverted; a win needs B2-style kernel fusion, not host→device policy migration |
| B2 | fuse launches | controller/head | not pursued | | | | **deferred**: requires fused HIP/Triton (int2-head epilogue + matcher fusion + argmax reduction); superseded by C1 (dflash removes the MTP serial loop entirely) |
| B3 | overlap MTP/verify | graph MTP layer | not pursued | | | | **deferred**: MTP path; superseded by C1 |
| B4 | sampler fast path | vLLM sampler FP32 elision | not pursued | | | | **deferred**: lives in vLLM source (needs a new boot patch + rebuild); small vs C1 |
| B5 | decode tile M<=16 | `radiance_mxfp4_fp8.hip` geometry | not pursued | | | | **deferred**: decode GEMM already ~95% streaming in isolation; not the step bottleneck (see A3/A6: step is flat to decode knobs) |
| C1 | dflash | `SPEC_METHOD=dflash` + matching DFlash2-FP8 drafter, 65536 ctx | dflash **36.4-36.8 / 36.5-36.8** vs mtp 48.6-57.0 / 53.1-58.9 | dflash 1.31-7.66 / 1.68-5.84 | dflash **127.9 / 108.7** vs mtp 89.5 / 80.1 | prose 1.305/1.676 | **KEEP (biggest win)**: step -30%, t/s +43% off / +36% on, TTFT ~58 vs ~87 ms; step is flat across categories (one graphed block). Caveat: dflash needs 7.17 GiB KV at 205k (only 4.27 free at util 0.98) -> cannot serve the 205k shape; measured at 65536. SPEC=8 vs drafter's native 7|
| C2 | checkpoint | blend-OCP vs native MXFP4 | n/a | n/a | n/a | n/a | **blocked**: plain `...-blend-MXFP4` is quant `mxfp4_16` (vLLM: "Unknown quantization method"); the runnable native comparator `...-blend-MXFP4-mtpfp8` (quark) crashes at engine init (torch._dynamo Unsupported: `input_quant_fp8.py:190 assert (scale is not None) == self.static`). The compressed-tensors OCP-GPTQ is the only servable MXFP4 checkpoint here |
| C3 | n-gram tail | tail-rate + prose check | n/a | n/a | n/a | prose dup8 0.0% in every run | **confirmed**: 205k quick runs show 27.0% n-gram share / 22.8% rows matched (repetition-heavy probe; the plan's 14%/13% is the full BetterBench mix), prose acceptance stable (1.45-2.35 across arms) => no prose regression |
| df-base96 | x16 dflash base | dflash, 98304 ctx, defaults (vhead=1, skinny=1, SPEC=8) | 36.6-37.0 / 36.6-36.8 | 1.06-7.67 / 1.36-6.18 | 121.1 / 98.0 | prose 1.060/1.356 | reference for the x16 dflash re-test (C1 at 65536 was 127.9/108.7) |
| A1' | verify head (dflash x16) | `RADIANCE_VERIFY_HEAD=0` vs default 1 | head-off 39.5-39.9 / 39.8-40.3 vs head-on 36.6-37.0 | echo 7.730 vs 7.667 | head-off 115.0 / 87.8 vs 121.1 / 98.0 | prose 1.254/1.610 | **keep default (vhead=1)**: int2 head is -8% step under dflash; validates the shipped default |
| A4' | skinny all (dflash x16) | `RADIANCE_SKINNY_GEMM=all` | 45.9-46.6 / 46.0-46.9 | 1.24-7.80 / 1.59-6.00 | 98.4 / 83.4 | prose 1.236/1.585 | **reject**: +25% step vs dflash default; the DFlash-era -3.9% does not hold under dflash2 here |
| A5' | SPEC=7 (dflash x16) | `SPEC=7` | 45.6-46.4 / 45.6-46.4 | 1.23-6.65 / 1.54-5.21 | 98.2 / 78.8 | prose 1.226/1.544 | **reject**: keep SPEC=8 (36.7 ms vs 46 ms); dflash drafter's native depth is not the fast one on this build |
| TRACE-mtp | MTP step trace | `RADIANCE_STEP_TRACE=20`, mtp 205k | exec 15.70 / sample_tok 36.50 / rpc_wait 0.25 / gpu_span 52.06 | toks/step 5.0 | step ~50 | - | **diagnosis**: MTP is **host-CPU-bound** (host chain 52 ms = step; rpc_wait ~0) |
| TRACE-df | dflash step trace | dflash 98304 | exec 2.93 / sample_tok 3.33 / rpc_wait 30.91 / gpu_span 37.20 | toks/step 9.0 | step 37.2 | - | **target**: dflash host chain ~6 ms, **GPU-bound** (idles 31 ms/step) |
| TRACE-stock | stock MTP (controller off) | `RADIANCE_DYNAMIC_DRAFT=0` | exec 2.47 / sample_tok 40.53 / rpc_wait 18.42 / gpu_span 61.31 | toks/step 9.0 | step ~61.5 | - | controller is **net-positive** (50 vs 61.5 ms); keep it ON; 13 ms of exec is not the matcher |
| M1 | matcher stride | `RADIANCE_DRAFT_MATCH_EVERY=2` | 51.4-56.6 / 54.0-58.4; exec 15.7->14.8, sample_tok 36.5->33.8 | 1.66-7.23 / 1.68-4.88 | 96.0 / 68.2 | echo 5.341 | **reject (neutral)**: matcher+sync is only ~1-3 ms, not the 13 ms; flag kept inert (default 1) |
| B3 | deferred decide | `RADIANCE_DRAFT_DEFER_DECIDE=1` (fixed depth, one readback) | 58.7-61.0 / 59.1-61.4; sample_tok 36.5->43.9 | 1.66-7.27 / 2.15-4.32 | 80.5 / 69.1 | prose 1.658/2.145 | **reject (regression)**: full depth adds slots; the cost is the per-slot draft-head op chain (head GEMM + capture + 248320 argmax), not the host sync. Code reverted |
| B2 | head/top-1 fusion | `RADIANCE_DRAFT_HEAD_TOP1=1` (default 0) | off echo 49.48 / json 54.82; on echo 49.52 / json 53.79 | echo 4.118 both arms | off 91.1 / on 94.3 tok/s | off echo 4.118, json 6.416; on echo 4.118, json 6.449 | **neutral/small (measured 2026-09-27 on the live V1/mtp path)**: card 1, `SPEC_METHOD=mtp`, SPEC=8, MAXLEN=65536, MAXSEQS=4, REPS=3. At matched acceptance the deterministic echo arm is flat (49.48 vs 49.52 ms/step) and json is -1.0 ms/step (-1.9%); the +3.5% combined t/s is code-category acceptance variance (dup8 27.1% vs 43.9%), not B2. No acceptance regression; output-equivalent as designed. Keep default OFF |
| P0/P1/PD/WP | async verify / mtp cudagraph / parallel drafting / weight prefetch | `RADIANCE_ASYNC_VERIFY`, `RADIANCE_MTP_CUDAGRAPH`, `RADIANCE_WEIGHT_PREFETCH`, `RADIANCE_PARALLEL_DRAFTING`, `RADIANCE_NUM_DRAFTERS` | withdrawn 2026-09-27 | - | - | - | **WITHDRAWN -- never active; not implemented.** Verified 2026-09-27 against vLLM 0.27.1: (1) `serve-mxfp4.sh`, which `bench-quick.sh` drives, applies NONE of the four patches, so every A/B ran the unpatched base image; (2) all four A/B runs used `SPEC_METHOD=dflash` while these target the serial `mtp` path -- under dflash `parallel_drafting` drafts all slots in one parallel forward, so P1's serial-Python-loop premise does not apply to the configuration that was benchmarked; (3) in the baked chain all four pass one sentinel to every anchor in the same file, so `_patchlib.apply` NOOPs anchors #2..N -- only the P0/P1/PD `__init__` blocks ever inserted; (4) the P0 sample/draft/bookkeep anchors and the PD drafter-init anchor match 0x in 0.27.1 and would FATAL the build once sentinels are per-anchor; (5) P1 `_propose_cudagraph` and PD `propose` are stubs that call `self.propose()`; (6) WP's kernel is `(void)` and no Python caller invokes `prefetch_weights`. The 32.2-32.7 tok/s spread is run noise. Removed from both Dockerfiles and the launcher; the four patch files were deleted. |
| PHASE | per-phase timers | `RADIANCE_DRAFT_PHASE_TIMERS=1`, mtp 205k | per slot-call (ms): head 0.39, capture 0.07, argmax 0.02, **d2h 1.82**, decide 0.056, gate 0.014, pad 0.002. Per step: **sample (rejection) 1.0, `_bookkeeping_sync` 28.4**, prepare 0.39, propose_all 16.5 (slot_total 13.3 + loop/fwd slack 2.9), postprocess 0.07, copy_draft 0.0; sample_tok 40.0, exec_model 14.8 | toks/step 5.0 | 93.4 / 79.5 | - | **residual FOUND**: the ~20 ms is `_bookkeeping_sync` (28 ms/step) = `RejectionSampler.parse_output`, a D2H that waits for the **main verify forward** — GPU wait, not MTP work, not saveable. MTP's own cost is the drafter's 16.5 ms (d2h 1.82 ms/slot = the serial draft-forward+head sync). Controller host work trivial (prepare+postprocess 0.46). B2's target (capture+argmax 0.09 ms/slot) is ~0.5 ms/step. Instrumentation default-off |
| x16-recycle | re-run Tier A + M1 on card 1 | mtp 205k, both thinking modes, off/on combined t/s | base 93.2/80.9; M1 94.0/78.6; vhead0 91.1/79.8; skinnyall 92.2/81.2; SPEC6 84.5/67.7; SPEC7 86.6/65.3; tau0.14 91.7/78.7; tau0.28 94.5/81.2 | - | - | - | **no regressions, no new wins**: card-0 verdicts transfer; MTP defaults (SPEC=8, tau=0.20, vhead=1, skinny default, MATCH_EVERY=1) confirmed best on x16. B1/B3 already rejected on x16. dflash stays the keeper (121.1/98.0 at 98304) |

## B2 design — fused int2 draft-head top-1 (deep research, pre-edit)

Scope: make the MTP **draft head** emit the two numbers the controller needs (`drafted token id`,
top-1 confidence) directly, so the per-slot full-row logits `Y` is never materialised, read, or
reduced. No output can change (speculative decoding verifies every proposal); only draft depth and
acceptance can move, so the gate is acceptance, not equivalence.

### Exact per-slot path today (mtp, TP=1, FAST_DRAFT=1, radiance_drafthead.py:229 `_apply_head_int2`)
1. `_draft_head_int2` (Triton, one program per 64-wide column block; `nblk = ceil(N/64) ≈ 3880` at
   `N=248320`): accumulates coarse fp32 `acc[M,64]`, **stores `Y[M,N]` bf16**, and `_emit`s
   `KCAND=8` block maxima (`BM` values, `BI` token indices) per block -> `bm/bi [M, nblk*8]`.
2. `bm.topk(RERANK=32)` -> `idx`; `_rerank_exact` scores those 32 exactly off the bf16/fp8 head ->
   `ex [M,32]`; `y.scatter_(idx, ex)` writes the exact values back over `Y`.
3. returns `Y`.
4. `_local_draft` (radiance_draft.py:207) zeroes the padding tail, then `capture_local(Y)` =
   `_cap_s1` (64 programs/row over V) + `_cap_s2_local` -> `(lmax,lsum)`; `lidx = Y.argmax(-1)`;
   returns `(lidx+start, 1/lsum)`.

So `Y` is written once and read twice, and 4 extra kernels/slot (2 capture + argmax + scatter) run
to recover two numbers.

### The invariant that makes the fusion sound
The exact winner is always a block maximum; block maxima are what `_emit` stores, and the top-32 are
rescored *exactly*. The head's stated guarantee is matching the bf16 argmax on all 8192 captured
rows. Therefore `argmax(Y) == idx[argmax(ex)]` whenever the winner is among the rescored candidates,
so the drafted id comes from the candidate arrays with no `Y` read.

### Proposed kernel change (`RADIANCE_DRAFT_HEAD_TOP1=1`, default OFF)
- `_draft_head_int2`: also emit a per-block row sum-exp `SM[M, nblk]` = `tl.sum(tl.exp(acc - blockmax))`
  over its 64 columns (blockmax is already computed for `_emit`). Keep the `BM/BI` emit.
- Skip the `Y` store on this path (`Y` is not needed).
- Rerank unchanged -> `ex`, `idx`.
- `ids = idx.gather(1, argmax(ex)) + start`, dropping padding candidates (`idx >= N-npad`).
- `conf`: `Mrow = max_b blockmax_b`; `S = Σ_b SM_b · exp(blockmax_b − Mrow)`;
  `conf = exp(max(ex) − Mrow) / S`. Same top-1 softmax prob, from coarse block partials instead of
  the exact `Y` row.
- Removes 2 capture kernels + the 248320-wide argmax + the `Y` store; adds one 64-column reduction
  inside the existing head kernel.

### Risks (each one is acceptance-visible, not output-visible)
1. argmax source: candidate-argmax vs `Y.argmax` can differ when a non-candidate coarse value
   exceeds the best exact candidate (rare per the docstring, but the mechanism exists).
2. confidence source: coarse block sum-exp vs exact `Y` sum-exp differs slightly -> different draft
   depth.
3. padding / TP: must mask padding candidates+blocks, and keep the TP>1 `_local_draft` all-gather
   contract (`lmax/lsum/argmax`).
4. dflash shares this head via `candidate_logits_processor` (`_radiance_topk_only`); the fused path
   must be confined to the MTP argmax caller.

### Expected benefit, and the open question to close first
Head ≈ 473 us/slot incl. rerank; the removed passes are ~12 us DRAM each at M=16 plus ~3
launches/slot. So B2 returns ~1–3 ms/step — nowhere near the **33 ms** of MTP-specific `sample_tok`.
Two of the three host hypotheses are already dead: B1 (device policy) and B3-defer (no per-slot
D2H/numpy, fixed depth) both regressed, so the cost is inside `_local_draft`'s per-slot op chain or
its GPU waits, not the controller. **Close this first:** add per-phase host timers inside
`_local_draft` (`_apply_head` / `capture_local` / `argmax` / `packed.cpu` / `slot_decide`) behind a
flag, re-run the step trace, and let the split pick the kernel. If `_apply_head` dominates, B2 is
right; if it is the capture/argmax, B2's fusion is right but bigger; if it is GPU wait, the fix is
graphing that chain.

### Verification plan (before it can land)
1. Kernel unit test in the container: capture draft-head inputs from a live serve, run the fused
   path and the current path, require identical `id` and `conf` within fp tolerance on every row.
2. End-to-end: flag ON vs OFF at 205k on card 1, both thinking modes, step trace + ms/step + tok/upd
   + acc/draft + combined t/s; reject on any acceptance loss on code/json/file_edit.
3. Revert rule: one flag, default off, one-line revert.
