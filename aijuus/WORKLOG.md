# AIJUUS work log

A running, dated log of what was changed, why, and how it was verified. Newest entry first.
Complements (does not replace) `TCCLA-VLLM-MTP-RESEARCH.md` and
`TCCLA-VLLM-MTP-IMPLEMENTATION-PLAN.md`, which hold the analysis and the plan.

---

## 2026-09-29 (cont. 16) — controller reload endpoint (no redeploy for registry arms)

- Added an HTTP admin endpoint to `model-controller.py`: `POST /reload` restarts matching instance(s)
  so their entrypoint re-reads `/patches/aijuus/model-registry.json` with fresh settings (drains
  in-flight, docker-restarts, polls /health); `GET /status`, `GET /health`. Bearer-auth with
  `VLLM_API_KEY`; port `RELOAD_PORT` (default 8101); a module-level `_op_lock` serializes swaps and
  reloads (one restart op at a time). Targets: `?instance=` (or "all") > `?model=` > all ready.
- Verified: py_compile + isolated HTTP smoke test (401 without key, 200 /status, 404 unknown).
- The controller script is bind-mounted ro → to activate it, **restart just the controller**
  (`docker restart <model-controller>`), fast, no model reload.
- This is the fast loop for vocab/knob A/Bs (each was previously a full Coolify redeploy).

---

## 2026-09-29 (cont. 17) — External tier: CPU is structurally unreachable; FS serves; mamba-from-tier proven bit-identical

### Structural finding (why the tier looked inert)
- Live redeploy: GPU KV cache **215,094 tokens**; CPU primary tier **104,720 tokens**
  (mmap 12.87 GB, geometry 4 groups x 30,638,080 B per 880-token block).
- Both tiers are LRU over the same stream, and the CPU tier is smaller than the GPU cache, so
  **a prefix evicted from the GPU has already left the CPU tier** — a CPU-tier hit is impossible
  by construction, in tierbench's own words "a staging buffer for the fs tier, not a cache."
- `kv_offload_lookup_skip_short_window_total` firing on every production lookup is the correct
  signature of "the GPU already holds >= what the tier offers", **not** the upstream #54360
  zero-hit defect. There is nothing to patch here.
- To make the CPU tier serve, its logical token capacity must exceed 215k — about **37 GiB** of
  /dev/shm, impossible with two instances sharing 32 G. The reachable external tier is **fs**.

### Proof: mamba-from-fs-tier restore is exact (equivbench --yes on vllm-1, isolated to that instance)
- `coldA` 51.60s `coldB` 52.85s (RECOMPUTE), `gpu` 2.80s (GPU), `fs` **11.99s (OFFLOAD)**.
- Served tier confirmed from the engine's per-tier series:
  `kv_offload_tiering_chunk_hits_total{tier="1:fs"} = 106`, `external_prefix_cache_hits_total
  = 84,480`, `kv_offload_load_bytes_total = 3.03 GB`.
- Correctness: **`coldB vs fs` and `coldA vs fs` are both `tokens_identical=True` and
  logprobs bit-identical (`max|dlogprob| = 0.000e+00`)**. The fs-tier restore of the hybrid
  attention + Mamba/GDN + draft state reproduces the cold continuation exactly. Stride 4 was
  live for this run.

### How to force/verify external-tier hits (documented method)
- `equivbench.py --yes` (phases `coldA,coldB,gpu,fs`): bit-exact correctness, drains then evicts
  CPU+GPU then probes the fs tier. This is the "mamba from tier is correct" gate.
- `tierbench.py --yes --phases cold,gpu,fs`: timing/attribution and per-tier validation.
- Both **evict the whole KV cache of the instance they hit**, so run against one instance
  (`:8001`) at idle; the other keeps serving. Pass `TIERBENCH_GPU_TOKENS`/`TIERBENCH_CPU_BLOCKS`
  when running inside the container (it cannot read `docker logs` there).
- Tool fixes: `tierbench` now preserves the `tier=` metric label and `classify()` names
  `OFFLOAD/FS` from `kv_offload_tiering_chunk_hits_total|tier=1:fs` (this build emits
  TieringMetricsTracker series, not the older `kv_offload_fs_load_bytes_total`). Verified:
  `classify -> OFFLOAD/FS`.

### Verdict
- The external-tier path does **not** need fixing: CPU non-service is design, not defect, and the
  fs tier restores the hybrid/Mamba state bit-identically. Keep CPU as a staging/fan-out buffer;
  treat fs as the external hit tier. If more external hit rate is wanted, the lever is fs
  residency/eviction policy, not the CPU tier size.

---

## 2026-09-29 (cont. 16) — Deep research: mamba-state correctness, KV-offload reality, ROCm 10; two config fixes

### Live verification of the hybrid MTP + offload path (vllm-0, boot 10:18)
- Boot log: **every functional KV-offload patch applied cleanly** (mixed-hit, instrumentation,
  eagle-fix, eagle-groups, mamba-stride knob, swa-align, head-cap, reconcile re-ask, deferral
  metrics) — no "did NOT apply" lines.
- Live shape confirmed: `mtp/8`, `mamba_cache_mode=align`, `mamba_cache_dtype=bfloat16`,
  `mamba_ssm_cache_dtype=float16`, `async_scheduling=False`, `OffloadingConnector` +
  fs secondary tier, `kv_offloading_size=12 GiB`.
- **The offload tier is inert on current traffic**: `vllm:prefix_cache_hits_total = 0` against
  `vllm:prefix_cache_queries_total = 3228`; every one of the 40 `kv_offload_lookup_calls_total`
  landed in `kv_offload_lookup_skip_short_window_total`; `kv_offload_cpu_cache_usage_perc = 0`.
  This is the signature of **vLLM #54360** (spec decode on hybrid GDN silently zeroes
  prefix-cache hits) and sits in the same connector-conditional path as **#53505** (hybrid
  Mamba align + ANY KV connector corrupts under spec decode, even at zero transferred tokens)
  and **#50454** (OffloadingConnector assert with kv offloading + mamba-hybrid + prefix caching
  + MTP). The 12 GiB CPU tier + fs tier are provisioned but delivering ~0.
- Correctness implication: the mamba recurrent state is the risk area the dev flagged. Our
  `turnbench.py` exact gate is the right test, but the live server enforces `VLLM_API_KEY`
  (401 without Bearer) and the harness (`turnbench`/`equivbench`/`tierbench`) sends no
  `Authorization` header; its `--dry-run` otherwise works end to end against `:8000`.
  ACTION: add optional Bearer auth to the KV tools before running the gate.

### Bounded exact gate (session A, 2 turns; auth added)
- Added `tierbench.auth_headers()` (reads `VLLM_API_KEY`/`TIERBENCH_API_KEY`) and wired it into
  `tierbench.http`/its streaming request, `equivbench.ask`, `turnbench.ask`. `py_compile` OK;
  header confirmed present in-container.
- Result: A1 25,184 tok COLD (recomputed all); A2 38,782 tok **GPU hit 23,760, ext 0,
  recomputed 15,022**. So the **GPU prefix cache works**, and the **external CPU/fs tier is
  never asked beyond it** (`ext=0`) — the offload tier really is dead weight on this path, as
  the metrics said.
- Exactness: A1 **EXACT** (max|dlp| 0). A2 GPU-hit path **DIFF** at token 30, max|dlp| 0.133,
  accepted 136/121. Harness verdicts: `HEALTH PASS: cache-caused none`, `TIER paths
  INCONCLUSIVE: no turn was served from the tier`. Read: the cached-vs-cold delta is the
  documented prefill-schedule numerics (`REVIEW-LOG.md:149-158`), not the #53505 corruption
  signature (that is looping/garbage/runaway, none present). The tier-served mamba-restore path
  remains unvalidated because no turn reached the tier; a full multi-session run plus a way to
  force external hits is the remaining gate.

### Mamba-state findings (upstream, not in our pinned v0.29.0 tag)
- #53505 (closed on main; fix not confirmed in our tag): attaching a connector switches the
  scheduler from `get_computed_blocks` to `get_computed_blocks_for_connector`; under spec decode
  the boundary diverges from the mamba spec-verify rollback point → decode resumes from a
  mismatched recurrent state.
- #55450 align-mode mamba states pinned across null gaps → upstream backport `patch_mamba_retire.py`.
  The R9700 review found it does NOT occur with async scheduling off + explicit previous-step
  free (`aijuus/refs/r9700-tp1/REVIEW-LOG.md:293`), which is our config; not ported.
- The GDN `torch.empty` nan/inf crash tcclaviger fixed in 29.04.1: **already avoided** — we run
  `RADIANCE_GDN_EMPTY_OUT=0`, so `patch_gdn_glue.py` allocates `torch.zeros`.

### Changes made
- `aijuus/coolify-compose-2gpu.yml` (both services): **`RADIANCE_MAMBA_STORE_STRIDE=4`** — the
  Coolify deploy never set it, so the mamba store-cadence patch ran inert (default 1) while
  `serve-mxfp4.sh:919` ships 4. Prereq (eagle-groups) is applied. Trades a truncated-prefix
  dead zone for ~2.4x more CPU-tier tokens at N=4. Validate with turnbench once auth is added.
- `aijuus/kv-offload/ops/entrypoint.sh`: cudagraph ladder is now the **union** of the dense
  small-step ladder and the exact `(SPEC+1)` multiples up to `CAP=SEQS*(SPEC+1)`. The exact
  train covers fixed-depth steps (vLLM pads to the next captured size); the dense train is kept
  because `RADIANCE_DYNAMIC_DRAFT` makes the per-step draft count variable, so a strict
  `(SPEC+1)` grid would under-cover dynamic decode. Verified: SPEC8/SEQS8 →
  `[1,2,4,8,9,12,16,18,20,24,27,28,32,36,40,44,45,48,52,54,56,60,63,64,68,72]`; `bash -n`, YAML parse OK.

### ROCm 10 verdict (deepened)
- Official AMD validation for ROCm 10.0.0 is **vLLM 0.27.0 only**. The one relevant gfx1201
  change is that ROCm 10 **deprecates `ROCBLAS_USE_HIPBLASLT_BATCHED`** (batched hipBLASLt no
  longer needs disabling) — it closes our logged batched-GEMM caution, but changes nothing about
  our AOT kernels. Counter-evidence that a version bump hurts: ollama#16624 bisected a ~14-20%
  gfx1201 regression to a Tensile-library selection change; ROCm#8242 is a gfx1201 hipBLASLt FP8
  heuristic SIGSEGV in a 7.14 nightly.
- tcclaviger's own image is a 0.29 tree on ROCm 10.0 / torch 2.11 / Python 3.14, so
  0.29-on-ROCm-10 is demonstrably workable — but with their kernels. Ours (libr4d,
  `radiance_mxfp4_fp8.so`, ~107 fork anchors, the TunableOp table) is AOT-built on 7.14 and
  needs a full rebuild + revalidation. **Do not move for performance; revisit only if a fix we
  need lands only on ROCm 10.**

### Sources
- vLLM #53505, #54360, #50454, #55450, #51599; ROCm 10.0.0 release notes;
  `blog.robai.net/vllmdocs` (capture-ladder rule, per-op/state-copy kernels, mamba dtype notes);
  `aijuus/refs/r9700-tp1/REVIEW-LOG.md:141-158,285-301`.

---

## 2026-09-29 (cont. 15) — DRY A/B: marginal quality, sampled t/s regression -> NOT adopted

- Arm `RADIANCE_DRY_MULTIPLIER=0.8` + `RADIANCE_DRY_RANGE=2048` (boot: `[run] DRY enabled:
  --dry-multiplier 0.8 --dry-range 2048`).
- Quality vs baseline (rep = 1 - unique/total words): loop-phrase 0.523 → 0.451 (better);
  loop-count 0.047 → 0.105; loop-sentence 0.941 → 0.941 (degen fires either way); normal-code
  0.431 → 0.511; normal-prose 0.341 → 0.333. Only the forced phrase loop clearly improved.
- t/s: greedy 87.6 → 91.4-91.6; **sampled 90.3 → 76.3-83.4** (consistent regression across runs;
  DRY adds per-step work in the rejection-sampler path). Acceptance healthy (~54%).
- **Verdict: not adopted** — no clear quality win, degen already covers pathological loops, and
  sampled throughput regressed. Revert: clear `RADIANCE_DRY_MULTIPLIER`/`RADIANCE_DRY_RANGE`.
- The DRY code/port stays in the tree (default-off, selectable) — only the env is cleared.

---

## 2026-09-29 (cont. 14) — DRY (4.1) ported and validated

- Added `patch_dry.py` (repo root): runtime overlay porting tcclaviger's `patches/dry_sampler` onto
  vLLM 0.29.0 across 10 files (new `dry.py` + penalties/metadata/sampler/rejection_sampler/
  sampling_params/input_processor/gpu_input_batch/config/arg_utils). All-or-nothing, idempotent,
  `RADIANCE_LOCAL_DRY=0` gate; anchors on the degen-inserted lines so it runs AFTER `patch_degen.py`.
- Entrypoint: copies `dry.py` from the refs, then runs `python3 patch_dry.py` (warn-and-continue).
- **Validated in-container**: patch applies; all touched modules import; `SamplingMetadata.dry_params`
  present; `SchedulerConfig`/`EngineArgs` dry defaults 0.0/0.75/2/-1/None; `apply_dry` runs;
  `DryParams.from_any(None) is None`; `--dry-*` flags registered.
- **Default-off** (`dry_multiplier` 0.0) → no behaviour change until a request or `--dry-multiplier`
  enables it. Optional later A/B.

---

## 2026-09-29 (cont. 13) — DRY port fully mapped (ready to execute); not applied

Mapped the entire `patches/dry_sampler` change set from the extracted reference (10 files) and wrote
it up as an exact port spec in the plan. DRY is **default-off** (`dry_multiplier` 0.0), so landing it
is behaviour-neutral until enabled. I did **not** apply it in this session: it rewrites the sampler /
rejection-sampler hot path plus the input-processor / GPU-input-batch plumbing, and an unvalidated
single pass risks generation correctness. It is the one remaining implementation item and is ready
to implement as `patch_dry.py` + an entrypoint `dry.py` copy, then validate in-container.

---

## 2026-09-29 (cont. 12) — SKINNY=all regressed (reverted); both tcclaviger tuning levers were worse

- **`RADIANCE_SKINNY_GEMM=all`**: greedy 92.6 (flat) but sampled **84.5** vs baseline 88.1;
  acceptance fell to 27-40% (mean accept ~3.2-4.2 vs ~5-6). **Reverted to `1`.** The bf16-ULP
  shapes hurt MTP drafting acceptance on our workload.
- Both tcclaviger tuning suggestions (TAU 0.28, SKINNY=all) are **worse** for our MTP config →
  our existing tuning (TAU 0.20, SKINNY=1, captures ladder) stands.

---

## 2026-09-29 (cont. 11) — Degen live; TAU 0.28 worse (kept 0.20); SKINNY=all armed

- Redeploy confirmed `[patch_degen] applied` and the banner shows `RADIANCE_DRAFT_TAU = 0.28`.
- **TAU arm result: 0.28 is WORSE** — greedy 86.6 / sampled 83.5 vs 0.20's 92.4 / 88.1 (−6.3% /
  −5.2%). **Reverted to 0.20** (our deep-dive value was already optimal; tcclaviger's 0.28 advice is
  relative to *their* 0.35 baseline).
- **Armed `RADIANCE_SKINNY_GEMM=all`** (compose, both services) — next redeploy measures it vs the
  0.20 baseline. Watch acceptance (a bf16-ULP change can move drafting acceptance).
- **Degen e2e validated**: live scheduler runs `check_stop(request, max_len, self.degen_params)` per
  appended token, and a spy confirmed `check_degeneration` is invoked. A text-periodic prompt
  ("x" x360) correctly does **not** fire because its *tokens* are varied (not a token-level loop);
  the detector is validated directly on synthetic period-1/period-4 token tails. Note for testing:
  degen keys on TOKEN periodicity, not character repetition.
- `docker rmi tcclaviger/vllm` was blocked by a pre-existing `tcclaviger_extract` container (left in
  place); refs are already extracted so the image can be dropped later.

---

## 2026-09-29 (cont. 10) — Deep dive vs tcclaviger: no missing MTP/kernel features; two tuning levers

- Radiance repo: **ours is a strict superset** (no `def`/`class` in theirs missing from ours).
  vLLM `spec_decode` (eagle/medusa/ngram) effectively identical. MTP patches `loopbreak`/`mm_mask`/
  `skinny_gemm` identical; `radiance_fusion`/`gdn_metadata` differ by our fork edits only.
  wvSplitK already upstream; R4D attention already used; shard-local draft confidence is TP>1 only (N/A).
- **Missed tuning 1 — `RADIANCE_DRAFT_TAU`**: tcclaviger bakes 0.35 (bf16 head) / **0.28 with
  `RADIANCE_FAST_DRAFT=1`** (+5.3% over 0.35); ours was 0.20. **Armed 0.20 → 0.28** (measure vs 3.3).
- **Missed tuning 2 — `RADIANCE_SKINNY_GEMM=all`**: ULP-level shapes incl. GDN `in_proj_ba`
  (48x/step, 28.5→3.6us) → **+3.5% tokens/s** in their case, no acceptance cost, but bf16-ULP can
  move acceptance. Next arm.

---

## 2026-09-29 (cont. 9) — Degen (4.2) ported, tested, wired

- Added `patch_degen.py` (repo root): runtime overlay porting tcclaviger's `patches/degen_detect`
  onto our vLLM 0.29.0 — `DegenParams` + `check_degeneration` in `v1/core/sched/utils.py`,
  `check_stop(..., degen_params)`, scheduler hook, `SchedulerConfig` fields, `EngineArgs`/CLI
  `--degen-*`, `Request.degen_counter`. All-or-nothing buffered edits, idempotent markers,
  `RADIANCE_LOCAL_DEGEN=0` disables.
- Wired into the entrypoint (`python3 patch_degen.py`, warn-and-continue).
- **Tested in-container**: patch applies, all modules import, `check_stop` signature updated,
  `SchedulerConfig`/`EngineArgs` defaults 100/6/128; detector fires on a period-1 run at ~129
  tokens, period-4 at ~132, never on a non-periodic sequence. Enabled by default (tcclaviger
  defaults); `--degen-max-period 0` disables.
- **4.1 DRY** remains staged (refs complete); not rushed into the production sampler.

---

## 2026-09-29 (cont. 8) — B4 neutral; wvSplitK already upstream; DRY/degen refs complete; ROCm 10 dug in

- **B4 (AITER off)**: greedy 92.1 / sampled 87.4 vs the 3.3 baseline 92.4 / 88.1 → **neutral** (within
  noise; our config's AITER role is minimal with R4D attention + sub-flags 0). **Reverted to
  `VLLM_ROCM_USE_AITER=1`.**
- **wvSplitK PR #34709**: NOT a backport — it is **already in vLLM 0.29.0 and enabled**
  (`use_skinny` accepts `on_gfx1x()`, `VLLM_ROCM_USE_SKINNY_GEMM` default True; `wvSplitK_hf` wave32
  kernels present in `_rocm_C.abi3.so`). Applies only to unquantized bf16 linears (our linears use
  radiance kernels), so no action.
- **DRY/degen port**: reference set completed (15 files) in
  `aijuus/refs/tcclaviger-vllm-29.05.12/`. Raw diff vs ours is ~1400 lines and interleaves
  tcclaviger's *other* fork-local changes (e.g. scheduler.py), so a blind port is unsafe; the
  DRY/degen hunks are tagged `FORK-LOCAL` and can be isolated. Port staged as the next focused task
  (runtime `patch_dry.py` / `patch_degen.py`).
- **ROCm 10 (5.2)**: deeper research in the plan. HIP 10 / LLVM 24 / rocBLAS 5.6 / hipBLASLt 1.4.1;
  validated vLLM is 0.27.0; "3.3x" is ROCm.AI/Hyperloom on Instinct. gfx1201 items are SystemDB
  refresh, hipBLASLt optimizer, HIP-graph replay gap reduction. Deferred: our libr4d +
  `radiance_mxfp4_fp8.so` + vLLM patched stack is AOT-built on ROCm 7.14 and would need a full
  rebuild/re-validation on 10.

---

## 2026-09-29 (cont. 7) — 3.3 adopted; B4 armed; DRY/degen refs extracted; ROCm 10 researched

- **3.3 ADOPTED.** Clean re-measure (pull finished): greedy 92.4 (was 88.1), sampled 88.1 (was 80.2)
  → +4.9% / +9.9%. Denser capture ladder kept.
- **B4 armed**: `VLLM_ROCM_USE_AITER=0` (both services) → next redeploy tests it vs the 3.3 baseline.
- **4.1 DRY / 4.2 degen**: pulled `tcclaviger/vllm:29.05.12` (vLLM `0.29.0.dev0+g2bdbbc8080`) and
  extracted the reference into `aijuus/refs/tcclaviger-vllm-29.05.12/` (dry.py 391 lines + sampler/
  scheduler/params plumbing). Port scoped in the plan; deferred as a multi-file vLLM patch.
- **3.2 flash_attn**: on gfx1201 the CK backend cannot build (Wave32); only the Triton backend works
  and it mainly helps ViT. Low value for our R4D text path.
- **5.2 ROCm 10**: researched (see plan). Validated vLLM is only 0.27.0; gfx1201 gains are a
  refreshed SystemDB + hipBLASLt optimizer; our 0.29 + libr4d/AITER patches would all need
  re-validation. Bigger near-term lever: upstream vLLM PR #34709 `wvSplitK` RDNA4 skinny GEMM
  (~15% decode on R9700) — backport candidate, independent of ROCm.

---

## 2026-09-29 (cont. 6) — Remaining items: 3.3 armed; 5.1 done; the rest blocked/low-value

- **3.3** cudagraph capture ladder densified in the entrypoint (added 12,20,28,36,44,52,60,68 to
  close the 8->24 gap, mirroring the branch's `[4,8,12,16,20,24,28,32]`). ARMED for the next redeploy.
- **5.1 DONE**: cloned the public `codeberg.org/tcclaviger/vllm-radiance` (936K, no `patches/`
  dir; radiance build context like ours). It confirms the current "davetha" path is the **DFlash2
  drafter's int4 decoder projections** under `RADIANCE_FAST_DRAFT` (codes derived at load, no
  calibration) — i.e. the DAVETHA axis we removed was MTP-only and deprecated. tcclaviger's *vLLM*
  source (the `tcclaviger/vllm` image: DRY `--dry-*`, degen `--degen-*`) is **private**
  (`git ls-remote` needs auth), so 4.1/4.2 are not portable without extracting from that image.
- **4.1 DRY**: blocked (not in vLLM 0.29; private source). **4.2 degen**: vLLM 0.29 ships upstream
  per-request `SamplingParams.repetition_detection` (scheduler-enforced), but no server-wide default.
- **3.2 flash_attn**: not installed; optional attention-backend A/B (R4D likely wins). **B3**: not
  pursued (needs the branch's paroquant plugin + GPTQ calibration `.pt`; low value). **5.2**: deferred.

---

## 2026-09-29 (cont. 5) — Removed the DAVETHA / W4 drafter-quant axis (superseded)

tcclaviger removed `DAVETHA_DRAFTER_QUANT` and the reduced-vocab W4 draft head in his current code
(its MTP path is different). Since we are not using it, removed it from the plan and codebase:
- deleted `aijuus/qwen3_5_mtp_w4.py`, `aijuus/draft_w4_lmhead.py`, `aijuus/r4d_lib.py`,
  `aijuus/draft_keep/merge.py`, `aijuus/draft_keep/keep-union.json`,
  `aijuus/TCCLA-VLLM-MTP-PLAN-A-FALLBACK.md`.
- removed `_install_draft_w4` + its call from `radiance_kernels.py` (patch 092 regenerated).
- removed `DAVETHA_DRAFTER_QUANT` and `RADIANCE_DRAFT_KEEP_FILE` from compose; dropped their overlay
  copies from the entrypoint.
- `build_vocab.py`: dropped the W4 `--json-out`.
- plan/research marked the Option A/B axis superseded; battery B2 removed; `draft_keep/.gitignore`
  no longer mentions keep.json.
Kept: the torch-level `RADIANCE_DRAFT_VOCAB` prune on the int2 head (Phase 1.0) — the adopted win.
`R4D_SO` still points at `.../v0.5.0-w4a16`, a valid libr4d with all serve kernels (name is now
just historical).

---

## 2026-09-29 (cont. 4) — A/B B2 result: W4 head NOT adopted

`DAVETHA_DRAFTER_QUANT=1` (reduced-vocab W4 head) redeployed and measured. Boot confirms it armed
(`[radiance.w4] 4-bit drafter hook installed`, kernel `gemm_w4a16_nt_m64` for M=64 K=5120 N=5120).

| Arm (SPEC 8) | greedy mean | sampled mean | acceptance len |
|---|---|---|---|
| int2 + vocab (baseline) | ~88 | ~80 | ~5-6 |
| W4 reduced head | ~85-87 | ~83-87 | ~3-4 |

Runs are noisy (sampled-json swings ~95-120 t/s). W4 **lowers** acceptance length and is flat-to-
slightly-worse on greedy, flat-to-slightly-better on sampled — no clear win. The R9700 branch also
kept int2+vocab for MTP in production. **Reverted `DAVETHA_DRAFTER_QUANT` to 0** (int2). The W4 path
stays implemented and validated, selectable via the env, for a sampled-heavy workload.

---

## 2026-09-29 (cont. 3) — A/B battery: B1 result, B2 armed

### B1 (SPEC depth) — RESULT: SPEC 8 wins
SPEC 4 (`spec_tokens=4`, redeployed): greedy mean 86.5, sampled mean 77.6 t/s vs the SPEC 8
baseline greedy 88.1 / sampled 80.2. Acceptance length dropped to ~2.8-3.4. **Reverted to
spec_tokens=8.** The branch's "SPEC 4 best under sampling" does not transfer to our workload/hw.

### B2 (head) — ARMED
`DAVETHA_DRAFTER_QUANT=1` set for both services (compose default). This runs the reduced-vocab W4
draft head (`Qwen3_5MTPW4`) instead of the int2+vocab head; will be measured against the recorded
SPEC-8 int2 baseline (greedy ~88, sampled ~80). Revert to 0 to switch back.

---

## 2026-09-29 (cont. 2) — Phase 1.0 FUSED ported+validated; Phase 3.1 verified

### Deploy verification (redeploy)
- `[radiance] using patched r4d.so from /home/juup/.cache/radiance-libr4d/v0.5.0-w4a16`;
  `import r4d` reports **0.5.0** and exposes `gemm_w4a16_nt_m64` → **Phase 1.1 fully unblocked**.
- `[aot-envkey] applied` (env key changed with the new env).
- `[radiance] DRAFT_VOCAB: 49159 of 248320 rows … EXACTSET=False`.
- TunableOp: **`reading tuning results from /cache/tunableop/skinny0.csv`** (validator accepted).

### Phase 1.0 COMPLETE — `RADIANCE_DRAFT_FUSED` ported
- Ported `_draft_head_int2_cand` + `_rerank_scatter` + `_apply_vocab_fused` from the R9700 branch,
  env-gated `RADIANCE_DRAFT_FUSED` (default 0).
- GPU-validated with the live module: with the exact set on, fused output is **byte-identical** to
  the unfused path (same finite mask, same bf16 values, 32 finite/row == RERANK); with the exact
  set off it is correctly **inert** (identical, all 512 sub rows finite).
- Precondition: `_radiance_topk_only` (EXACTSET or a candidate processor) and no embedding bias.
  Our MTP keeps EXACTSET **off** (finding #3, tau-gate), so FUSED is inert here until the tau-gate
  confidence is decoupled from the masked row — the follow-up that would unlock EXACTSET+FUSED.

### Phase 3.1 verified (TunableOp table)
- Table loads. Smoke bench on one prompt: greedy 123.3 t/s (flat), **sampled 64.5 → 72.3 t/s
  (+12%)**, acceptance ~60% unchanged. Dropped the temporary `PYTORCH_TUNABLEOP_VERBOSE=1`.

### A/B harness + baseline
- Added `aijuus/tools/mtp-bench.py` (dependency-free; greedy + sampled, multi-prompt mean, optional
  acceptance via `/metrics`). Run inside a container: `python3 /patches/aijuus/tools/mtp-bench.py`.
- Baseline (int2+vocab, TunableOp, SPEC 8): greedy mean ~88, sampled mean ~80 t/s (2 reps x 400 tok).
  Single-prompt numbers swing a lot; use the mean.
- Battery knobs recorded in the plan's "How to run an arm". Phase 1.1 verified ready to arm
  (`DAVETHA_DRAFTER_QUANT=1`: registry remap + `draft_w4_lmhead.available()=True`).

---

## 2026-09-29 (cont.) — Phase 1.1 blocked then unblocked; Phase 3.1 wired

### Verified live (post-redeploy)
- `[aot-envkey] applied` (env key `1f2e5a7357ca`); new AOT dir seeded from the base inductor cache.
- `[radiance] DRAFT_VOCAB: 49160 of 248320 rows … 0.07 GiB/rank, EXACTSET=False`.
- Greedy 123.8 t/s, sampled 64.5 t/s; mean acceptance length 5–6, draft acceptance 49–66% — no
  regression from the vocab prune.

### Phase 1.1 blocker (important)
The reduced-vocab W4 draft head (`DAVETHA_DRAFTER_QUANT=1` -> `aijuus/qwen3_5_mtp_w4.py` +
`draft_w4_lmhead`) needs libr4d's `r4d_gemm_w4a16_nt_m64`. **Our prebuilt libr4d builds
(`b9e42ab-rx6`, `b9e42ab-rx9`) do NOT export it** — `readelf -sW` shows only
`r4d_gemm_bf16_nt_m16`; the `.hip` source is not on the host. So the W4 kernel path cannot run
and Phase 1.1 is blocked until we obtain/build libr4d with that kernel (tcclaviger codeberg
libr4d, or an r4dhip build). (An earlier note that r4d.so carried the symbol was wrong.)
Note: the boot's `[radiance.w4] no w4a16 gemm_nt kernel … disabled` is the *full-head* path
(`radiance_w4.py`, gated `RADIANCE_DRAFT_W4_FULL`, default off), not this one.

**Resolved (same day):** the blocker was the compose override, not libr4d. The Dockerfile already
pins `R4D_VERSION=v0.5.0`, and the image's *baked* `r4d.so` **has**
`r4d_gemm_w4a16_nt_m64` (also `w4a8`/`mxfp4a8`); `b9e42ab` is an ancestor of `v0.5.0`, so our GDN
guards are included. Compose was overriding it with the older `b9e42ab-rx9` build via
`R4D_SO` + `/r4d`. Fix: extract the baked `r4d.so` to
`/home/juup/.cache/radiance-libr4d/v0.5.0-w4a16/r4d.so` and repoint both services' `R4D_SO`/`/r4d`
at it. No image rebuild needed. Phase 1.1 stays env-gated (`DAVETHA_DRAFTER_QUANT=0`); activation
is the B2 A/B at the end.

Also aligned the reduced-vocab W4 keep file to the same canonical set: `build_vocab.py` now emits
`keep-union.json` (the W4 head reads JSON), and `RADIANCE_DRAFT_KEEP_FILE` defaults to it.

### Phase 3.1 started (unblocked, +5.1% measured on the branch)
- `aijuus/kv-offload/ops/entrypoint.sh` copies the prebuilt table to
  `/cache/tunableop/skinny0.csv`.
- `_defaults.server_env` now sets `PYTORCH_TUNABLEOP_ENABLED=1`,
  `PYTORCH_TUNABLEOP_TUNING=0`, `PYTORCH_TUNABLEOP_FILENAME=/cache/tunableop/skinny%d.csv`,
  and `PYTORCH_TUNABLEOP_VERBOSE=1` (temporary, for confirming the table loads; remove after).
- Validators match torch 2.11.0 / HIP 714 / gfx1201; hipBLASLt version could not be read
  statically — the boot log will say if the table is accepted.
- NEXT: redeploy -> confirm the table loads (or "ignored") -> benchmark; then remove VERBOSE.

---

## 2026-09-29 — R9700 branch integration, Phase 0.5/1.0, review fixes

### Context
Folded the findings of `mtstanfield/vllm-mxfp4@r9700-tp1` (see
`aijuus/refs/r9700-tp1/REVIEW-LOG.md`, extracted this session) into our MTP plan, then ran a
`/review uncommitted` pass and fixed every finding.

### Decisions
- **Integrate directly (proven lossless/byte-exact), do not defer to A/B:** Phase 0.5
  (0.29 correctness gates) and Phase 1.0 (draft-head surface: vocab prune + exactset). A/B-only
  items stay in the plan's "End A/B Battery" (SPEC depth, int2-vs-W4 head, MXFP4+GPTQ drafter,
  AITER, heavy branch wins).
- **`RADIANCE_DRAFT_EXACTSET` must NOT be enabled on the MTP entry.** It masks the draft row to
  the RERANK candidate set, and `radiance_draft.py:_local_draft` derives its tau-gate confidence
  from that row (`1/lsum`), so the confidence is renormalized over 32 candidates -> inflated
  `cum` -> deeper drafting than `RADIANCE_DRAFT_TAU` was tuned for. Kept the feature env-gated
  but OFF. (`RADIANCE_DRAFT_VOCAB` masks to ~49k rows, a much smaller renormalization, and is on.)
- **Keep-list caveat:** the branch's `qwen38-draft-vocab-49152.txt` is a SEED only. The shipped
  `keep-union.txt` is the union of the seed and our collector output, built by
  `aijuus/draft_keep/build_vocab.py`. Grow it via `--extra <tokenized-corpus>` (robust) rather
  than the draft-pass hook.
- **Redeploy semantics:** all of this is runtime overlay + registry, read from `/patches` at
  container start, so it needs a Coolify **restart/redeploy, not an image rebuild**.

### Changes
- `radiance_drafthead.py` — added `RADIANCE_DRAFT_VOCAB` / `RADIANCE_DRAFT_EXACTSET`
  (`_SubHead`, `_apply_head_vocab`, guards for top-1/bounds/too-small lists); fixed
  `_apply_head_int2_top1` so a repeated coarse-only call cannot raise `UnboundLocalError`.
- `patch_aot_envkey.py` (repo root; Phase 0.5) — ports the branch's 0.29 AOT + piecewise
  compile-cache env-keying, now with a GC that keeps the newest 3 env-keyed dirs per base hash.
  Wired into the entrypoint and enabled via `RADIANCE_LOCAL_AOT_ENVKEY=1` in `_defaults`.
- `aijuus/kv-offload/ops/entrypoint.sh` — runs `patch_aot_envkey.py`; overlay copies and that
  patch now warn-and-continue instead of aborting the boot under `set -e`.
- `aijuus/model-registry.json` — `RADIANCE_DRAFT_VOCAB` (union file) in the MTP entry; interim
  `kv_cache_memory=8761733283` on the MTP entry to unblock 160000 (see below); AOT env-key on.
- `aijuus/collect_tokens.py` — single `ensure_installed()` entry point installing both hooks
  (`compute_logits` + `get_top_tokens`) with the same class list (incl. `Qwen3_5MoeMTP`) and
  `_capturing()` guard. `radiance_kernels._install_token_collector` now delegates to it.
- `aijuus/draft_keep/idset.py` (new) — one id-list parser; `build_vocab.py` and `merge.py` use it.
- `radiance_draft.py` — windowed n-gram fallback rescans only the miss rows and copies back only
  their rows (was a full-batch rescan + a second whole-pack host sync).
- `fp8_mtp.py` — import falls back to `aijuus.quant_defaults`.
- `gpu-detect.sh`, `serve-mxfp4.sh` — pass the spec depth to `rad_kv_lookup` so `mtp:<depth>` KV
  pins are reachable (previously the reader queried `mtp:` and never matched).
- `aijuus/patches/` — regenerated `010/030/050/060/062/092` from `origin/main`; added
  `033-serve-mxfp4-kv-depth.patch`.

### Deploy incident (fixed)
MTP boot failed: `kv=none(profiled)` -> vLLM profiled 4.59 GiB but 160000 needs 6.05 GiB.
Cause: the MTP registry entry had no `kv_cache_memory`. Interim fix: pin `8761733283`, borrowed
from the identical-weights `-blend-MXFP4-OCP-GPTQ` (dflash) entry (MTP loads fewer weights).
Proper fix: run `./calibrate-kv.sh SPEC_METHOD=mtp SPEC=8` and replace the interim pin.

### Verification
- `py_compile` on all edited Python; `bash -n` on entrypoint/serve/gpu-detect.
- `build_vocab.py` -> `keep-union.txt` = 49,160 ids; `merge.py` -> 56 ids (smoke).
- All 20 overlay patches: reverse-check on the worktree; forward-apply on a pristine
  `origin/main` worktree reproduces the worktree files byte-for-byte (20/20, 0 mismatches).
- `collect_tokens`: both hooks install (`cl=True gt=True`).
- `patch_aot_envkey`: generated `decorators.py`/`backends.py` parse; GC block present.
- `fp8_mtp` import resolves; registry JSON valid.

### Residuals / next
- The 49k vocab still mildly renormalizes the tau-gate confidence (much smaller than the removed
  mask-to-32). Watch acceptance after redeploy.
- `radiance_draft.py` per-row fallback is logic-verified; needs a GPU run to confirm the
  `base=n` empty-window path.
- `patch_aot_envkey.py`, `aijuus/collect_tokens.py`, `aijuus/draft_keep/*` and the new modules
  are still UNTRACKED. Track them (or accept warn-and-continue) before deploying from a clean
  checkout.
- Continue: redeploy -> confirm `[aot-envkey] applied` and `DRAFT_VOCAB: 49160 of 248320 rows` ->
  smoke acceptance -> Phase 1.1 (W4 reduced head) or the `FUSED` draft-head kernel.
