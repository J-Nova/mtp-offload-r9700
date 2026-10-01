# Open tasks index — radiance-vllm-mxfp4

Consolidated from `aijuus/WORKLOG.md` residuals and the plan docs' open checkboxes.
Generated 2026-09-30. Status: **READY** (actionable now) · **BLOCKED** (needs a
reload/A-B/GPU you own) · **PENDING** (needs your approval) · **PARKED** (deliberately
deferred) · **DORMANT** (implemented, env-gated off).

**A/B method (cont.32/37):** the compile cache is performance-critical. Clearing
`~/.radiance-cache-w4a8-093-gdnm-nqft-fp8s-gnq-tp1s` (vllm-0 only; vllm-1 is separate) makes the
*next* boot ~3-4x slower. Recovery is **multi-boot**: after a clear, restart until **c1 ≈ 68 t/s is
verified** before measuring (a single warm restart was not always enough; seeding the Triton cache
from the intact vllm-1 dir helped). Never measure an unverified boot.

## 0. Remaining tasks — refreshed cont.64

**Closed since the previous index:** M1 (n-gram HSA fix + enablement, cont.59–63) · **C1/C2** batched
matcher (cont.64) · T3 · V3 · H2 · **OS1/E1** (enabled in registry, cont.60) · A1–A5 · A10 · A11
(dormant, kept) · OK3–OK7 · PF1/PF5/PF6/PF7 · K1–K3 · T1/T2 · MT1/MT2 · X2 (N/A for V2) · V1 (N/A, V1-only).

**A. No-rebuild, actionable now (top):**
1. **rx10 libr4d — BUILT and LIVE (cont.66): +6% c1/c8.** `b9e42ab-rx10/r4d.so` now serves; its
   rx9 narrow-state kernels fixed the declined fp16 GDN fused path. Remaining: repoint the compose
   mount to `b9e42ab-rx10` (done) and redeploy; validate numerics (acceptance moved 1.93→2.17).
   **Lazy (OS3/E3)** now BOOTS (cont.67, `_Tables` drift fixed) with KV +11% (190,157), but multi-turn
   is still fail-open corrupt. Route 3c as pure Python is infeasible; choose **3a** (kernel `stale`
   flag → fail loud) or **K2** (backup stash page, 3→4 pages) — both need an `r4d.so` rebuild.
2. **H1** — commit this work (`patch_dynamic_depth.py`, `aijuus/batched_ngram_equiv.py`,
   WORKLOG/OPEN-TASKS-INDEX/ACTIONABLE-AB-PLAN).
3. **Instrumentation/validation:** **ST6** acceptance-by-position per category (serving measurement) ·
   **VC2** battery re-run (EXACTSET+FUSED on the frozen union-freq build) · **V2** keep watching
   acceptance post vocab-prune.
4. **Research (read-only):** **ST7** remaining HIP kernel packages · **TF4** TunableOp GEMM sweep ·
   **TF6/TF7** reduced-vocab handling / `CLAV_DRAFT_HEAD_REPLICATE`.
5. **Prefill:** **P6** o_proj quant fold (deprioritised by A5) · **P10** harness (BetterBench sweep +
   18k probe + GSM8K) · **P2** R4D h256 geometry (needs a selection-table entry; check runtime feasibility).
6. **Ops reliability:** the first boot after a reload can be ~3× slow (c1 **22**) until a warm restart
   — decide whether to pre-warm / stabilise the AOT key so a single cold boot is trustworthy.
7. **Docs/cleanup:** **H4** README overlay refresh · **K2** align `mtp-27B-MXFP4-Thinkingcap` kv pin
   (`8761733283`) only if/when it is piloted.

**B. Rebuild-class / kernel (needs a build you own):** **lazy-GDN integration** (rx10 `r4d.so` now
built; left: fix `radiance_gdn_lazy._Tables` + route 3c) · PF2 GDN chunk-scan parallelization · PF3 R4D
prefill scheduling · PF4 MXFP4 split-K @16k · S4 shrink each draft forward (C5 contract repair precedes
S5–S8) · B2 fused int2 draft-head top-1 · KB1 libr4d pin bump · TF2 `clav_attn`.

**C. Blocked / parked / dormant:** P3/P5 (on P4) · P4 AITER (needs `KV_OFFLOAD_GIB=0`; conflicts with
offload) · S3/S6–S8 · P8/P9 · X1 · OS2 (A/B negative) · D1/D2 (dormant; D2 available as hardening) ·
OS4/OS5 · TF5 (private fork) · VC1 tau gate (A/B negative).

**D. Deferred / inherent:** resume bit-identity Route A (benefit-destroying; not recommended) ·
MT1/MT2 (inherent resume-vs-cold numerics).

**Do NOT redo:** the "Do NOT repeat (measured neutral/worse)" list below.

---

## 1. Blockers (features that are off/unsafe)

| ID | Task | Status | Source | Blocker | Effort |
|---|---|---|---|---|---|
| M1 | N-gram tail HSA fault at bs≥2. | **FIXED (one line)** cont.59 | Cause = `_nblk` passed the window *base* not *size* → scan grid ~9x too big → `q` past `ML` → unmasked suffix load OOB (gfx1201 HSA). Fix: `gpu._nblk(n, window_size)` in `patch_dynamic_depth.py`. Not the UVA gather / not syncs. **NGRAM now ON**: adaptive per-request gate + variant B depth (cont.62/63); **C1/C2** batched matcher (cont.64) is bit-identical and perf-neutral (fewer launches/syncs, no regression). |
| M2 | **`RADIANCE_DRAFT_HEAD_TOP1`** — drop the arm (neutral + mutually exclusive with the vocab prune; crashes on reload). Registry stays off. | **CLOSED (dropped)** | WORKLOG cont.28 3b, cont.29 | no code change |

## 2. Calibration / config debt

| ID | Task | Status | Source | Notes |
|---|---|---|---|---|
| K1 | MTP KV calibrate. | **N/A for compose** cont.57 | `calibrate-kv.sh` writes `~/.cache/radiance-mxfp4/kv-profiles.local.tsv`, which only `serve-mxfp4.sh` reads; our compose uses explicit registry `kv_cache_memory` pins. Recalibrate only if we move to serve-mxfp4.sh. |
| K2 | ~~Dflash pilot pins 8.16 GiB~~ **CLOSED**: `dflash-27B-MXFP4`, `-blend`, `-Thinkingcap` now all `6442450944` (6.0 GiB). Remaining: `mtp-27B-MXFP4-Thinkingcap` still `8761733283` (not served; align if/when piloted). | CLOSED | WORKLOG cont.27/28 5 | registry shows 6.0 GiB |
| K3 | Capture-ladder trim: dense 26 sizes vs `[4,8,12,16,20,24,28,32]`. | **CLOSED (neutral)** cont.33 | `RADIANCE_CAPTURE_SIZES` knob added; coarse 11-bucket ladder c8 370.9 vs 369.1, KV 171,320 unchanged. |

## 3. MTP A/B + tuning battery

| ID | Task | Status | Source | Notes |
|---|---|---|---|---|
| T1 | **End A/B battery** on the frozen union-freq build: `EXACTSET`+`FUSED` on/off; AITER on/off; SPEC 4 vs 8 re-run. | **CLOSED by prior evidence** | EXACTSET must stay off (renormalizes the tau-gate confidence; R9700 decision); FUSED on; SPEC 8 wins (cont.3/28); AITER blocked (P4). |
| T2 | Untested MTP perf levers: `RADIANCE_DRAFT_TAU` 0.15/0.25; `RADIANCE_DRAFT_RERANK` 32 vs other. | **CLOSED by prior evidence** | tau 0.20 best (cont.11/28); rerank neutral (cont.28); live `RADIANCE_DRAFT_KNOB_FILE` path is V1-only and the served runner is V2. |
| T3 | N-gram tail-rate confirmation. | **DONE (cont.61/64)** | repetitive ~+9% c1 / +33% c8 vs NGRAM=0; extended rows ~3-8% (workload-dependent). |

## 4. MTP prefill / TTFT (MTP-PREFILL-PLAN)

| ID | Task | Status | Notes |
|---|---|---|---|
| P2 | R4D prefill geometry for h256/large-q (BLOCK_M/TILE/num_warps/waves); R4D selection table has no h256 prefill entry. | READY (reload/arm) | also audit redundant rope/quant launches; attention block 880 vs larger page + ssm dtype tradeoff |
| P3 | MXFP4 prefill GEMM TN4/split-K sweep per M (~+10% @M8192 documented). | BLOCKED (reload/arm) | diminishing |
| P4 | rDNA h256 backend: A/B R4D vs AITER attention. | **BLOCKED** (cont.32) | `ROCM_AITER_UNIFIED_ATTN` invalid with the OffloadingConnector (`['KV connector not supported']`); needs `KV_OFFLOAD_GIB=0`. Entrypoint now honours `R4D_ATTN` (default 1). |
| P5 | `long_prefill_token_threshold` policy; `#57951` scheduler overlay only if threshold≠0. | BLOCKED on P4/C | |
| P6 | Fold attention-output → o_proj quant (only if A1 confirmation A/B shows headroom). | **DEPRIORITISED (A5)** | A5 measured fusion on **2230** vs off 1949 tok/s (−12.6%); the fusion already captures this headroom |
| P7 | Confirmation A/B: `FUSE_RMS_QUANT=0`+`FP8_STREAM=0` vs on (A1). | **DONE (A5)** | fusion on **2230 tok/s** (TTFT 6979 ms) vs off 1949 (−12.6%); **keep fusion on** |
| P8 | Speculative/sparse prefill (#39060). | PARKED | biggest TTFT, high cost |
| P9 | GDN prefill; phase arena; cross-token prefetch. | PARKED | low / excluded |
| P10 | Measurement harness (BetterBench sweep + 18k probe + GSM8K). | READY | D |

## 5. MTP structural tracks (MTP-SPEEDUP-PLAN, longer horizon)

| ID | Task | Status | Notes |
|---|---|---|---|
| S2 | Move MTP to V2 vehicle | **DONE** | V2 live; controller hooks active; dynamic depth implemented |
| S3 | Graph the draft-loop body (kills the V1 host loop) | PARKED | V2 per-step FULL graphs already in place; residual = per-step metadata rebuild |
| S4 | Shrink each draft forward (fused head contract repair; metadata fast path; draft attn/KV) | OPEN | C5 contract repair precedes S5-S8 |
| S5 | Run fewer draft forwards (controller extension + cross-step depth) | **A/B NEGATIVE** | conf-exit port (cont.30) regressed |
| S6 | Raise acceptance per forward (rollout-aware native MTP retrain; parallel/single-pass head; probabilistic sampling) | PARKED | model work |
| S7 | Verify forward + scheduler | PARKED | |
| S8 | New-methodology candidates (T8a-f) | PARKED | research-grade |
| — | B2 fused int2 draft-head top-1 | DEFERRED | MTP-DECODE-OPT-PLAN; ~1-3 ms/step; needs fused HIP/Triton; close the per-phase-timer question first |
| — | MTP-DECODE-OPT Tier A (A1-A8) | **CLOSED/rejected** | see progress log; B1/B3 rejected; C1 dflash is the big keeper |

## 6. Attention / image-level

| ID | Task | Status | Source |
|---|---|---|---|
| X1 | flash_attn 2.8.3 | **PARKED (low value)** | Dockerfile pip-overlay possible, but CK cannot build on gfx1201 (Wave32/64) and the Triton path mainly helps ViT; our body uses R4D. |
| X2 | Reorder-threshold fix script (see below). | PENDING | REORDER-THRESHOLD-FIX-PLAN |

## 7. Reorder-threshold fix (REORDER-THRESHOLD-FIX-PLAN.md)

- [x] Script implementation — **N/A** (cont.33): `calculate_reorder_batch_threshold` is V1-only (`gpu_model_runner.py`); the served V2 `v1/worker/gpu/model_runner.py` has no reorder-threshold path, so the bug does not apply. No patch. · [ ] Upstream PR #55898 monitoring (maintain the backend invariant).
- Context: under current config it is a no-op (threshold already 5/8); only matters if a threshold-1 backend appears with a hybrid attention group. **Closed as N/A for the V2 runner.**

## 8. Validation gaps

| ID | Task | Status | Source |
|---|---|---|---|
| V1 | `radiance_draft.py` per-row n-gram windowed fallback (`base=n` empty-window). | **N/A (V1-only)** | cont.38 logic verified, but the served runner is V2; the path is inert there. Superseded by the V2 n-gram tail (cont.62/63). |
| V2 | Watch acceptance after vocab prune. | **MONITOR** | no regression across runs. |
| V3 | MTP combined depth + `[ngram]` tail validation. | **CLOSED (cont.63)** | variant B (`NGRAM_DEPTH=8` at bs≤2) + adaptive gate enabled; repetitive c1 +8.8%, generic c1 +9%, c8 neutral. |
| V4 | External KV tier: CPU tier structurally unreachable; forced external-tier hit method documented (WORKLOG cont.16/17). | **DOCUMENTED** | method in WORKLOG cont.17 §"How to force/verify" |

## 9. Housekeeping

| ID | Task | Status |
|---|---|---|
| H1 | Commit work. Latest (cont.64): `patch_dynamic_depth.py` (C1/C2 batched matcher), `aijuus/batched_ngram_equiv.py`, `aijuus/WORKLOG.md`, `aijuus/OPEN-TASKS-INDEX.md`, `aijuus/ACTIONABLE-AB-PLAN.md`. Earlier: `bench-quick.py`, `entrypoint.sh`, `model-registry.json`, `patch_mtp_conf_exit.py`, `patch_mamba_scratch_zero.py`, `bench-conc.py`, `MTP-RAGGED-VERIFY-AND-CONV-SCRATCH-PLAN.md`. | READY |
| H2 | Decide the n-gram code path (keep inert vs revert). | **CLOSED (enabled)** | see §14 |
| H3 | ~~Track untracked modules~~ **DONE** (`patch_aot_envkey.py`, `collect_tokens.py`, `draft_keep/*` now tracked-clean). | CLOSED |
| H4 | README/DOCKERHUB doc refresh. | **PARTIAL (overlay)** | root README/DOCKERHUB are upstream-owned; deployment-reality corrections added to `aijuus/README.md` (V2 inertness, NGRAM=0, HEAD_TOP1 dropped, R4D/cache method). |

## 10. Dormant ports from tcclaviger dev (cont.30/31) — implemented, off

| ID | Task | Status | Notes |
|---|---|---|---|
| D1 | `patch_mtp_conf_exit.py` (confidence early-exit + ragged verify) | DORMANT | A/B negative; revisit only with calibrated acceptance or higher concurrency |
| D2 | `patch_mamba_scratch_zero.py` (conv align-copy tail zeroing) | DORMANT, **perf-neutral** | cont.32: the earlier -55% was a stale-cache artifact; under clear+warm it is free (c1 identical, c8 387.5 vs 369.1). Viable correctness hardening; reachability of the stale scratch still unproven. |
| D3 | `aijuus/ab.env` hook + `bench-conc.py` | LIVE (baseline) | reusable A/B mechanism |

## Do NOT repeat (measured neutral/worse)

Lazy GDN snapshots; rotation stream 3; async scheduling for MTP; custom RMSNorm+quant op; prefill
epilogue prefetch; `COMPILE_SIZES`; `RADIANCE_GDN_STRIDED_GATES`/`GDN_EMPTY_OUT`/`COOP_RED`;
MTP Tier A A1-A8 (rejected in the progress log); B1 device gate; B3 defer-decide; **static** n-gram
tail `NGRAM=1` *without* the adaptive gate (net-negative on generic — the adaptive gate + variant B is
the shipped form); tcclaviger conf-exit/ragged-verify and conv-scratch ports here.

---

## 11. Chunk-size / prefill efficiency (new research)

Baseline fact: 16k chunk regresses tokens/s vs 4k; cause is **GDN scan occupancy collapse** (scan grid
`(2,48,N)` → 96·N WGs; 16k = one request, N=1; 4k packs ~4) plus transients and a growing all-reduce
(44→164 MiB), not GEMM arithmetic (MXFP4 GEMM is flat/power-bound at both M). Overlay/env knobs are
spent — none makes 16k faster. 8k is the pragmatic compromise.

| ID | Task | Status | Notes |
|---|---|---|---|
| PF1 | **Profile chunk 4096 vs 16384.** | **DONE (8k prompt)** | cont.37: prefill is chunk-independent at 7974 tok (452 vs 454 ms, ~17.6k tok/s); the 16k-chunk decode delta was a compile-key artifact. Needs **larger prompts/context** to exercise PF2/PF3. Harness `aijuus/bench-prefill-ttft.py`. |
| PF2 | GDN chunk-scan parallelization | **PARKED** | kernel lives in libr4d (upstream clone at build), not in this repo; no runtime overlay can change a compiled `.so`. Needs a libr4d source patch + rebuild to validate. |
| PF3 | R4D prefill scheduling (split-KV/stream-K, persistent triangular) | **PARKED** | libr4d kernel; no runtime overlay; needs source patch + rebuild (+ long-context validation). |
| PF4 | MXFP4 prefill GEMM at M=16384: revisit split-K only if PF1 shows a gap. | PARKED (low) | TN=8 accumulator wall; ~80-82% of WMMA, power-bound |
| PF5 | All-reduce AR-quant A/B. | **N/A (DP)** cont.57 | not bit-identical and only helps a TP>1 all-reduce we do not run (world_size 1). |
| PF6 | 8k chunk compromise. | **DECISION (keep 16384)** cont.57 | PF1 showed prefill chunk-independent at 8k; revisit only with an N>1 GDN occupancy profile at long context. |
| PF7 | Co-schedule more sequences per step. | DOCUMENTED | batching property, not a knob. |

## 12. KV offload / 4-bit KV research

Measured: tier blocks are near-incompressible losslessly (~1.18× zstd, dead); the connector is a raw
byte store with an O_DIRECT fixed-size round-trip contract, so a lossy tier cannot stay bit-identical.

| ID | Task | Status | Notes |
|---|---|---|---|
| OK1 | **Kill "Design B"** (4-bit tier via roundtrip-at-write). | **CLOSED (killed)** | either ~1.18× lossless (dead) or 4-bit quality model-wide + fp8 pool for tier-bytes-only benefit |
| OK2 | If 4-bit KV economics wanted → **TurboQuant on the GPU** (packed pages then store free; FileMapper folds dtype/canonical_format so old files age out). | PARKED (large) | the only real 4-bit path; big project |
| OK3 | Measure the tier. | **DONE** cont.57 | 90k: cold 52.2 s, GPU 1.6 s, FS/tier 17.7 s. Promotion latency, not bytes, is the cost. |
| OK4 | Verify promotion queue-depth fix. | **VERIFIED** cont.57 | fs promotion functional. |
| OK5 | Conditional tier compaction. | **NOT JUSTIFIED** cont.57 | OK3 shows the cost is promotion latency, not bytes; revisit only if tier capacity binds. |
| OK6 | Lossless tier compaction (~1.18×). | **CLOSED (dead)** | not worth the CPU cost |
| OK7 | Paper (UltraQuant/TurboQuant-class) decode/prefill uplift. | **CLOSED (none)** | decode win needs actual 4-bit KV; prefill not in scope; its primitives already used on MXFP4 GEMMs |

## 13. Offload streaming / scheduler (ADAPTIVE-KV-STREAMING, MTP-PREFILL E)

| ID | Task | Status | Notes |
|---|---|---|---|
| OS1 | **E1** suffix-only invalidation + eagle/MTP group inclusion. | **ENABLED (cont.60)** | `patch_offload_suffix_inv.py` + `patch_offload_eagle_include.py`; gates `RADIANCE_OFFLOAD_SUFFIX_INV` / `RADIANCE_OFFLOAD_EAGLE_INCLUDE` set to 1 on both MTP registry entries. Warm E1-on c1 **71.3**/c8 **371.6** vs off 67.8/369; tier 67.3→71.0%, recompute 25→24%, HEALTH PASS. Byte-identical outputs + unchanged acceptance. |
| OS2 | **E2** offload store decoupling (submit D2H at creation; stop the finished-req self-flush). | **IMPLEMENTED, A/B NEGATIVE, DORMANT** | `aijuus/kv-offload/patches/patch_offload_lazy_commit.py` (`RADIANCE_OFFLOAD_LAZY_COMMIT`, default off). Cold-18k TTFT **8031 vs 7560 ms (+5.6%)**: eager D2H contends with the 2nd prefill chunk. Corrected anchor: the real await is `pre_forward -> handle_preemptions -> worker.wait(jobs_to_flush)` (not `wait_for_save`, a no-op). |
| OS3 | **E3** bounded host-snapshot GDN rollback to re-enable lazy GDN snapshots. | **lazy BOOTS (cont.67); repair needs kernel work** | rx10 built + live (`gdn_lazy_update`/`_materialize`); its rx9 kernels gave **+6% c1/c8**. `radiance_gdn_lazy._Tables` drift fixed → lazy boots with `RADIANCE_GDN_LAZY=1`, **KV 190,157 (+11%)**, single-turn clean. Remaining: route 3c as pure Python is **infeasible** (fail-open decided on-device; the needed state is unrecoverable in Python). Next: **3a** kernel `stale` flag → runner raises (fail loud, safe to enable) or **K2** backup stash page (3→4 pages) for the real repair. Both need an `r4d.so` rebuild (~1 min). |
| OS4 | E4 phase arena / E5 cross-token prefetch+sparse feedback. | PARKED | excluded / decode-only single-digit |
| OS5 | E6 PDL fence-ordering principle. | N/A | CUDA-only |

## 14. MTP V2 controller gaps (TCCLA plan)

| ID | Task | Status | Notes |
|---|---|---|---|
| VC1 | Port the tau gate / n-gram tail from `radiance_draft.py` (V1) to V2. | **CLOSED** | tau gate via `patch_mtp_conf_exit.py` (A/B negative, dormant); **n-gram tail done** in `patch_dynamic_depth.py` (adaptive gate + variant B, cont.62/63; batched matcher cont.64) |
| VC2 | Battery re-run on the frozen union-freq build with/without `EXACTSET+FUSED`. | OPEN (deferred) | interactions may differ from the 49k-head measurements |
| VC3 | B3 MXFP4+GPTQ drafter (paroquant plugin + GPTQ calibration `.pt`). | NOT PURSUED | overlaps the dropped W4 head |
| H2 | Decide the n-gram code path (keep inert vs revert). | **CLOSED (enabled)** | cont.62/63: adaptive gate + variant B on by default in the registry; C1/C2 batched matcher (cont.64) validated bit-identical. Revert only if M1 is abandoned |

## 15. Open research questions (TCCLA plan §Open Questions)

1. tcclaviger's (private) vLLM fork location. 2. Reduced-vocab draft-head impact on our model.
3. fp8 KV quality impact. 4. AITER vs custom extensions. 5. ROCm 10 vs 7.14 impact.
6. "deadcodes int2 verifier/predictor". 7. "davethas pixie dust". 8. Reduced-vocab handling of tokens
not in the keep file. 9. Optimal `CLAV_DRAFT_HEAD_REPLICATE` for 2-GPU DP.

## 16. Structural / research open items (MTP-SPEEDUP §6, TCCLA research)

| ID | Task | Status | Notes |
|---|---|---|---|
| ST1 | T0 per-phase timers. | **DONE by prior data** | host-bound: ~28 ms `_bookkeeping_sync` + ~16.5 ms propose. |
| ST2 | T2a V2-MTP boot blockers. | **ANSWERED** | V2 runs MTP; FULL-graph per-step decode working. |
| ST3 | Matcher-context staleness (C4). | **N/A (V1-only)** | `radiance_draft.py` path is inert on the served V2 runner. |
| ST4 | TOP1 confidence contract (C5). | **CLOSED (subsumed by M2)** | arm dropped. |
| ST5 | Fused multi-step eligibility for this checkpoint's draft attention groups. | OPEN | needs the main-era tree or a backport |
| ST6 | Acceptance-by-position 1-8 per category (sizes S5/S6/S8b payoff). | READY | serving measurement |
| ST7 | Analyze remaining HIP kernel packages: rfhip, fp8hip, parohip, full r4dhip family. | READY (research) | |

## 17. tcclaviger fork feature evaluations (TCCLA research Next Steps)

Resolved already: vocab prune (A), EXACTSET+FUSED (B/C), TunableOp table (D), AOT env-key (F),
DRY, degen, ROCm 10 research, fp8 KV (we already run it), `RADIANCE_DYNAMIC_DRAFT` on/off (net-positive).

| ID | Task | Status | Notes |
|---|---|---|---|
| TF1 | Evaluate `gdn_verify_r` fused MTP spec-verify kernel (from `gdn_hip`) to replace our MTP verify path. | READY (research) | |
| TF2 | Evaluate `clav_attn` native unified attention vs R4D. | BLOCKED | overlaps P4 (KV connector) |
| TF3 | Evaluate `clav_ar`/`clav_ag` P2P-BAR collectives for the 2-GPU DP link. | READY | x1 PCIe card caveat |
| TF4 | Evaluate TunableOp GEMM sweep (`CLAV_TUNABLEOP_SWEEP=1`). | READY | |
| TF5 | Fetch tcclaviger's vLLM fork for the full patch set (deadcode int2 verifier/predictor, davetha "pixie dust"). | BLOCKED (private) | |
| TF6 | Reduced-vocab handling of tokens outside `draft_keep_file`. | OPEN | open question 8 |
| TF7 | Optimal `CLAV_DRAFT_HEAD_REPLICATE` for 2-GPU DP. | OPEN | open question 9 |

## 18. tcclaviger HIP kernel packages — analysis result (research, cont.36)

Hard blocker for all of them: tcclaviger is **py3.14 / torch 2.11+rocm10.0**; we are **py3.12 /
rocm7.14** — the pybind/ctypes `.so`s (`r4d.so`, `clav_ar_ext`, `clav_ag_ext`, `clav_attn_C`,
`gdn_hip_C`, `rfi_hip_C`, `fp8hip`) are ABI/ROCm-bound and source is not shipped for `fp8hip`/`parohip`.

| ID | Task | Status | Notes |
|---|---|---|---|
| TF1 | `gdn_verify_r` fused MTP verify kernel | **CLOSED (not an upgrade)** | tcclaviger's own libr4d chain is 1.2-1.5× faster than `gdn_verify_r`; we already run libr4d + our `fused_update`. Only the interface idea (single dispatch, raw int32 metadata, in-kernel per-token snapshots) is transferable as a Python-overhead port. |
| TF3 | `clav_ar`/`clav_ag` P2P-BAR collectives | **CLOSED (N/A)** | TP-group collectives only; we are DP (world_size 1, no collectives); the x1 card defeats BAR P2P anyway. |
| TF2 | `clav_attn` unified attention A/B vs R4D | **PARKED** | kernel source not shipped in the image (only the cp314 `.so`), so it genuinely cannot be overlaid/rebuilt by us. |
| TF4 | TunableOp GEMM sweep (`CLAV_TUNABLEOP_SWEEP=1`) | OPEN | not covered by the image analysis |
| TF5 | Fetch tcclaviger's vLLM fork | BLOCKED (private) | image-level analysis substitutes |
| KB1 | Bump libr4d pin (`r4d_pq_*`, dense w4a8) | **PARKED** | possible as a Dockerfile+extras build overlay, but needs a rebuild to validate and tcclaviger's r4d GDN NaNs on our model (`Dockerfile.ggz14:475-479`). Low value (we already run r4d). |
| KB2 | `gdn_verify_r` interface port | **PARKED** | kernel is cp314-only; superseded by libr4d (1.2-1.5x faster); only a Python-shape idea. |
| KB3 | RFI/RFA fused-prologue study | **PARKED (low)** | requires requantizing the model; same fusion class as `RADIANCE_FUSE_RMS_QUANT`. |
| KB4 | `fp8hip` block-scaled w8a8 | **PARKED (low)** | cp314-only `.so`; only relevant to an fp8 checkpoint we do not serve. |
| — | `q4hc`(HyperConnection), `plehip`(PLE), `r4d_qsa/ple/mhc/dsfp/dflash2` | **N/A** | Qwen4Exp/Flash-Next/DSV4-only |

## 19. NEW: pre-existing multi-turn cached-path divergence (correctness)

`turnbench --exact` (three-agent workload to ~110k tokens/session) diverges on CACHED-vs-COLD twins
**on the unmodified baseline**: GPU partial-hit path at turn 2 (first-div 15-111) and TIER path at
turn 4+. COLD-only turns are EXACT. This is independent of E1 (both arms diverge identically).

| ID | Task | Status | Notes |
|---|---|---|---|
| MT1 | GPU partial-hit divergence. | **CLOSED (inherent)** cont.53 | Bisected: persists with align patch off and with `SPEC_METHOD=none`; writer oracle is bit-identical. Cause = partial-hit resume prefills its suffix in a different launch/chunk geometry than a cold full prefill (+fp8 KV ulp flips). No KV fix. |
| MT2 | TIER divergence independent vs inherited. | **CLOSED (inherent)** cont.53 | Same mechanism: byte-identical KV does not imply bit-identical output when resume and cold use different prefill geometry. The harness's "TIER must be EXACT" premise is false under chunked prefill + fp8 KV. |

This is a real correctness item the E1 work surfaced; it gates trusting MTP group inclusion at high
hit rates, and may affect production multi-turn long-context quality today.

## 20. Resume bit-identity research verdict (cont.54)

Solvable only by (A) a pure-Python overlay that quantizes every prefix hit down to cold's real chunk
stride `C = B·floor((CHUNK−draft_slots)/B) = 15840` (`single_type_kv_cache_manager.py:77,783`,
`kv_cache_coordinator.py:666`; align the Mamba/offload store grid to C too) — cost: forfeits up to
~15.8k tokens of reuse per hit, and `prompt ≤ CHUNK` forces hit=0; or (B) kernel work (absolute-phase
GDN scan + fixed-grid fp16 state rounding; alters cold numerics; `r4d.so` rebuild). **Not recommended**:
fp8 KV sets a ~6 %/element noise floor, chunked prefill re-partitions reductions, and R4D opts out of
batch-invariance (`radiance_r4d_attn.py:260`, `backend.py:200,321`). Validate on acceptance/served
semantics. Route A left unimplemented (benefit-destroying) unless bit-exactness becomes a hard requirement.
