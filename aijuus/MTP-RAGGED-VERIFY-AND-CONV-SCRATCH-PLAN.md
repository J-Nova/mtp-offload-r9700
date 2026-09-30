# Port plan — tcclaviger dev MTP ragged-verify + mamba/conv spec-scratch zeroing

Status: **planning only — no code changed.** Written 2026-09-30.

Sources of truth for the port:
- Reference implementation: `tcclaviger/vllm:dev` = tag **29.06.18** (digest
  `sha256:a7e7f1bccab4…`, built 2026-09-30), extracted at `/tmp/kilo/tccla-dev-root`.
  Prior reference (baseline for the delta): `tcclaviger/vllm:29.05.12` at
  `/tmp/kilo/tccla-2905-root`.
- Our baked runtime: `juupp/vllm-radiance:0.9.3-collect-tokens` = **vLLM 0.29.0**,
  extracted at `/tmp/kilo/ours-vllm/vllm` (baked) and `/tmp/kilo/ours-patched/vllm`
  (running, runtime-patched).
- Our overlay mechanism: root `patch_*.py` are executed at container boot from the
  `/patches` mount (`aijuus/kv-offload/ops/entrypoint.sh:235-270`); personal runtime
  patches live in `aijuus/kv-offload/patches/`; build-time overlays in
  `aijuus/patches/*.patch`.

This plan ports exactly the two items identified as transferable:

- **Track A (performance / concurrency):** MTP per-request confidence early-exit +
  *ragged verify* — verify only the drafts each request actually kept, instead of the
  full scheduled width K.
- **Track B (correctness):** mamba/GDN conv-state **spec-scratch zeroing** — the
  conv-state columns added by `num_speculative_tokens` are read at the accepted offset
  but never initialised, so stale values from the page's previous owner can reach the
  convolution.

The PLE fusion and QSA ring-hazard guard in the same image are **out of scope**: both
belong to the Qwen4Exp / Qwen3.8-Flash-Next architecture, not to our Qwen3.8 blend.

---

## 1. Evidence — what tcclaviger dev actually changed

### 1.1 Track A (`patches/mtp_confidence_exit`)

The baseline (29.05.12) already had `draft_confidence_threshold`, `draft_confidence`,
`_update_early_exit`, `_mask_stopped_tail`, `num_draft_steps` and the metrics. Dev
reworks it into a concurrency feature:

| Concern | 29.05.12 | dev (29.06.18) |
|---|---|---|
| Exit signal | raw top-1 draft prob | calibrated P(accept) from an online acceptance estimator |
| Exit state | torch tensors on device | numpy on host; per-request `draft_lens` |
| Shortened drafts | masked to `-1`, **still verified at full K** | physically cut: verify runs at `kept = min(K, draft_lens)`, ragged batch |
| K per batch | fixed | optional `draft_len_halving` (`draft_len_cap`: K at bs 1-2, halved per power of two, floor 1) |
| Threshold per batch | one server value | optional `draft_confidence_schedule` `"<batch>:<thr>,…"` + per-request `SamplingParams.draft_confidence_threshold` (also via xargs) |
| CUDA graphs | varlen only for adaptive verification | varlen also for early-exit batches: `min_decode_query_len=2` + uniform 1-token capture descs |
| Scheduler | stats/EMA use scheduled K | consume `ModelRunnerOutput.num_verified_draft_tokens`; **rollback stays on scheduled K** |
| Log label | "Draft confidence per position" | "Pred. accept per position" |

Exact reference anchors (dev root, py3.14):
- `config/speculative.py:95-127` (`parse_draft_confidence_schedule`, `draft_len_cap`),
  `:468-476` (fields), `:1645-1664` (validation).
- `sampling_params.py:393-395,567-572`; `v1/engine/input_processor.py:301-321,495`.
- `v1/worker/gpu/sample/states.py:40-47` (NaN-default per-request threshold table).
- `v1/worker/gpu/spec_decode/speculator.py:172-206,277-289,479-491,493-544,546-573`.
- `v1/worker/gpu/spec_decode/autoregressive/speculator.py:51-73,139-160,175-201,
  348-358,387-409,450-494,496-530,650-708,710-764`.
- `v1/worker/gpu/model_runner.py:735-760,1285-1333,1372-1391,2057-2062,2171-2173,
  2272-2280,2431-2447`.
- `v1/worker/gpu/cudagraph_utils.py:148-160,303-325,584-596`;
  `.../spec_decode/autoregressive/cudagraph_utils.py:51-61`.
- `v1/core/sched/scheduler.py:2024-2068`; `v1/outputs.py:375-381`;
  `v1/worker/gpu/spec_decode/utils.py:36-83`; `v1/spec_decode/metrics.py:125`.

Ragged-verify core (dev), in words:
1. The speculator keeps one graph replay per draft step (`use_step_graph_decode`),
   computes the running product of per-position acceptance on the host, and stops a
   request at the first step whose product drops below its (per-request over
   per-batch) threshold. `draft_lens[req] = stop_col + 1`, capped by `max_steps`.
2. `gather_batch_req_state` computes `kept = min(scheduled, draft_lens[idx])`; when any
   row was cut it rewrites `num_scheduled_tokens`, `num_tokens`, `max_query_len` and
   sets `num_draft_tokens_per_req` + `verified_draft_tokens`.
3. `prepare_inputs` sizes logits/query rows from `num_draft_tokens_per_req`.
4. The runner reports `num_verified_draft_tokens` in `ModelRunnerOutput`; the scheduler
   uses it for `_dynw_observe` and `make_spec_decoding_stats` only. **Rollback
   (`num_computed_tokens -= num_rejected`, placeholders) is unchanged** — the cut tail
   is treated as rejected, which is correct.

### 1.2 Track B (conv "spec scratch")

Spec decode allocates conv-state width `kernel_width - 1 + num_spec`; the kernel's
logical `state_len` is `kernel_width - 1`, so columns `[state_len, conv_state_len)` are
scratch used during verify. They are only ever written by a full verify, and:
- prefill writes only `[0, state_len)`;
- the align state-copy shifts by `token_bias` and leaves the tail holding the destination
  page's previous owner.

When the verify reads at `conv_state_token_offset = num_accepted_tokens - 1 > 0`, the
window reaches into uninitialised scratch → foreign values convolved into q/k/v.

Dev fix, exact:
- `model_executor/layers/mamba/ops/causal_conv1d.py`: new `conv_state_len` arg and
  `ZERO_SPARE`/`NP2_SPARE` consts; zeroing added at 5 write sites (fwd `:263-277,
  323-335,363-375,453-465`; update `:1114-1123`); launcher flags at `:653-654,882,
  905-906,1352,1463,1488`.
- `v1/worker/mamba_utils.py`: `_copy_mamba_state_block` zeroes destination columns past
  the shifted window in both conv branches (DS `:350-371`; SD removed early `return`,
  `:391-426`).
- HIP twins rebuilt because the fork shadows the Triton paths by default:
  `clav_conv1d`, `clav_state_copy`, `gdn_hip` (sizes 742,952→771,624 / 259,016→283,592 /
  1,708,776→1,712,872). Triton-only is insufficient on their fork.

---

## 2. Where our stack stands

Our fork is vLLM **0.29.0** and already carries a large part of the support machinery.
Existence audit (baked and patched trees identical unless noted):

| Item | Ours | Notes |
|---|---|---|
| V2 `GPUModelRunner` + `AutoRegressiveSpeculator`/`MTPSpeculator` | present | the served MTP vehicle |
| `num_speculative_tokens_per_batch_size` + `dynamic_sd_lookup` | present | scheduler path `scheduler.py:1458-1472` |
| `adaptive_verification.py` + `enable_adaptive_verification` | present | dspark-only today |
| `varlen_decode` capture plumbing | present | `cudagraph_utils.py:119,128,233-252,478-486` |
| `num_draft_tokens_per_req` derivation | present | `model_runner.py:~1186-1208` |
| `patch_dynamic_depth.py` (runtime) | applied | per-batch K from `spec_schedule`, forces non-fused loop, edits the same two files |
| `patch_dynamic_sd_cudagraph.py` (runtime) | applied when `SPEC_SCHEDULE` set | `_init_candidates` fix |
| `radiance_draft.py` controller (`RADIANCE_DRAFT_TAU`) | present | V1 hooks inert; V2 hooks liveness-only |
| `OnlineAcceptanceEstimator` / `acceptance_estimator.py` | **absent** | dev has a 19.4 KB module |
| `draft_lens` producer + `DraftTokensHandler` length input | **absent** | handler records one batch width (`spec_decode/utils.py:~22-52`) |
| `num_verified_draft_tokens` / `num_draft_tokens_per_req` in output | **absent** | `outputs.py` `ModelRunnerOutput` |
| `min_decode_query_len` + uniform 1-token descs | **absent** | `cudagraph_utils.py` |
| `draft_confidence_threshold` config / per-request param | **absent** | ours is env `RADIANCE_DRAFT_TAU` only |
| `ZERO_SPARE` / `NP2_SPARE` / conv tail zeroing | **absent** | both `causal_conv1d.py` and `v1/worker/mamba_utils.py` |
| GDN conv width > `state_len` under MTP SPEC | **present** | `conv_kernel-1+num_spec = 3+8 = 11`, `state_len=3` |
| R4D GDN fused conv | **used** | `RADIANCE_USE_R4D=1`; tail zeroing absent in `r4d_radiance_extras*.patch` |

Served config for `mtp-27B-MXFP4-blend`: `spec_schedule=[[1,2,5],[3,8,4]]`,
`RADIANCE_DYNAMIC_DEPTH=1`, `RADIANCE_DRAFT_NGRAM=0`, async scheduling off
(`VLLM_ASYNC_SCHEDULING=0`). So today the draft width is batch-scheduled (5 at bs 1-2,
4 at bs 3-8) and **the verify still runs at the full width** — ragged verify is the
missing lever.

Relevant existing plan material: `aijuus/MTP-SPEEDUP-PLAN.md` §T2c/T2d ("length bridge
is missing, not already plumbed"; hybrid rollback across variable lengths), and
`aijuus/REORDER-THRESHOLD-FIX-PLAN.md` (the reorder-threshold invariant). This port is
the concrete implementation of T2c/T2d, with tcclaviger dev as the reference.

---

## 3. Track A — MTP ragged-verify overlay

### A.0 Design decision: drive the exit with our policy, not the estimator (first cut)

tcclaviger couples the exit to the calibrated estimator, but the ragged mechanism itself
is independent of the confidence *source* (`_update_early_exit` falls back to raw top-1
when no estimator is present). We already have a per-slot top-1 confidence policy
(`RADIANCE_DRAFT_TAU`, product gate). Therefore:

- **A1 (primary):** port the ragged mechanism; gate the exit on the running product of
  the **raw top-1** draft confidence at `RADIANCE_DRAFT_TAU`, computed on the V2
  autoregressive speculator. No new module.
- **A3 (optional, later):** port `acceptance_estimator.py` + calibrated P(accept) +
  `draft_confidence_schedule` / `draft_len_halving` / per-request `SamplingParams`
  threshold, only if A1 gates show raw top-1 underperforms.

`draft_len_halving` is redundant with our existing `spec_schedule`; we keep the
schedule-driven K and add per-request early exit under it.

### A.1 Per-file change set (anchors are text-stable, line numbers drift)

1. **New file** `vllm/v1/worker/gpu/spec_decode/utils.py` edit:
   extend `DraftTokensHandler.set_draft_tokens(input_batch, draft_tokens, draft_lens=None)`
   and truncate rows in `get_draft_tokens()` to `lens` (placeholder rows for
   non-structured requests). Mirror dev `utils.py:36-83`.
2. **`spec_decode/speculator.py`** (`DraftModelSpeculator`):
   add `sample_draft_with_confidence` (a variant of `_greedy_sample_draft` at
   `:358` that also returns the picked token's softmax probability), a device buffer
   `draft_token_confidence_probs`, and the per-step capture hook. Keep `sample_draft`
   unchanged for all non-exit paths.
3. **`spec_decode/autoregressive/speculator.py`** (`AutoRegressiveSpeculator`):
   - host state: `draft_lens`, per-batch threshold, `req_thresholds` (bound by runner),
     survival product, stop column (dev `:51-73`).
   - in `propose` (`:212`): reset survival, compute `max_steps` (our `_radiance_eff_k`),
     per-step `_update_early_exit`, `_finish_early_exit` → `draft_lens`
     (`input_batch.idx_mapping_np`).
   - `_multi_step_decode` (`:529`, already bounded by `_radiance_eff_k` after our patch)
     and the per-step `decode_cudagraph_manager.run_fullgraph(batch_desc)` path (`:567`)
     must read the confidence tensor back to host after each replayed step and stop.
     This is the hardest integration point (see A.4).
4. **`v1/worker/gpu/model_runner.py`**:
   - bind `speculator.req_thresholds = self.sampler.sampling_states.<table>` (dev
     `:756-760`) — or skip in A1 (single server threshold), add in A3.
   - in the batch-state gather used by `prepare_inputs`: compute `kept`, trim
     `num_scheduled_tokens`/`num_tokens`/`max_query_len`, populate
     `num_draft_tokens_per_req` and `verified_draft_tokens` (dev `:1285-1333`).
   - logits/query sizing from the kept count (dev `:1372-1391`).
   - report `num_verified_draft_tokens` on `ModelRunnerOutput`.
   - pass `draft_lens` into `draft_tokens_handler.set_draft_tokens` (dev `:2272-2280`).
5. **`v1/outputs.py`** `ModelRunnerOutput`: add `num_verified_draft_tokens` field.
6. **`v1/core/sched/scheduler.py`** `update_from_output`: use `num_verified_draft_tokens`
   for `_radiance_dynw_observe`/`make_spec_decoding_stats`; **do not touch rollback**
   (dev `:2024-2068`).
7. **`v1/worker/gpu/cudagraph_utils.py`**: add `min_decode_query_len` to
   `CudaGraphManager`/`ModelCudaGraphManager`; in candidate generation use
   `num_reqs = min(num_tokens // min_q, max_num_reqs)` and add uniform 1-token descs when
   `min_q > 1` (dev `:148-160,303-325,584-596`).
8. **`.../spec_decode/autoregressive/cudagraph_utils.py`** and
   `AutoRegressiveSpeculator.init_cudagraph_manager` (`:135`): build the speculator
   prefill/decode managers with `varlen_decode=True`, `min_decode_query_len=2` when the
   exit is on (dev `:175-201`), and pass the runtime `max_query_len` into the dispatcher
   (dev `:348-358`). Reconcile with `patch_dynamic_sd_cudagraph.py`.

### A.2 Overlay packaging

- **One runtime patch script:** `aijuus/kv-offload/patches/patch_mtp_ragged_verify.py`,
  modeled on `patch_dynamic_depth.py` (marker-idempotent, `ast.parse` every edited file,
  hard-fail on anchor mismatch). It edits items 1-7 above and (optionally) rewrites the
  speculator to add the step-graph path.
- Wire into `aijuus/kv-offload/ops/entrypoint.sh` **after** `patch_dynamic_depth.py`,
  gated:
  ```
  if [ "${RADIANCE_MTP_RAGGED_VERIFY:-0}" = "1" ]; then
    python3 aijuus/kv-offload/patches/patch_mtp_ragged_verify.py \
      || { echo "[run] FATAL: patch_mtp_ragged_verify failed"; exit 1; }
  fi
  ```
- No rebuild needed for A1. If A3 lands, add
  `patch_mtp_acceptance_estimator.py` which writes the new
  `spec_decode/acceptance_estimator.py` + the config/sampling plumbing.

### A.3 Env / config surface (A1)

- `RADIANCE_MTP_RAGGED_VERIFY` (default 0; master switch).
- `RADIANCE_MTP_EXIT_TAU` (default inherits `RADIANCE_DRAFT_TAU`, i.e. 0.20).
- `RADIANCE_MTP_EXIT_MAX_K` (default = scheduled `_radiance_eff_k`; a safety cap).
- Keep `RADIANCE_DRAFT_NGRAM=0` for the first port (the matcher has an unresolved
  `HSA_STATUS_ERROR_EXCEPTION` at bs≥2 on this host — WORKLOG 2026-09-29).

### A.4 Risks / hard parts

1. **Per-step confidence readback under FULL cudagraphs.** Our draft decode loop runs
   each step through `decode_cudagraph_manager.run_fullgraph` (speculator `:567`). To
   stop between steps we must (a) have the confidence written by the captured graph and
   (b) D2H it after each replay. tcclaviger's `_step_graph_multi_step_decode` does
   exactly this (in-graph metadata update + host read). Porting that is the critical
   piece; if it cannot be made stable, fall back to the **eager per-step** loop
   (`use_step_graph_decode=False`, dev `:459-474`) with `_radiance_eff_k` bounding K —
   cheaper to port, some launch overhead.
2. **Hybrid rollback across variable lengths (T2d).** Shortened verify changes the
   query lengths the GDN conv/recurrent state sees; GDN state selection and the conv
   window (`causal_conv1d.py`) must stay correct for 0-draft-after-N-accepted and across
   block boundaries/preemption/cache modes. This is where **Track B interlocks** — the
   spec-scratch columns exist precisely because of this window.
3. **Reorder-threshold invariant.** Varlen spec rows must classify as decode. tcclaviger
   pins `min_decode_query_len=2` so every captured row stays on the spec-verify kernels;
   our `REORDER-THRESHOLD-FIX-PLAN.md` requires the global threshold to stay
   `1+SPEC` (GDN) and the drafter backend must not report threshold 1. Re-check
   `calculate_reorder_batch_threshold` after enabling varlen.
4. **Cudagraph capture-shape growth.** Varlen + uniform 1-token descs add captures;
   our KV pin is tight (OOM history at `max_num_batched_tokens=16384`). Count capture
   sizes and peak memory before/after.
5. **Anchor fragility.** `patch_dynamic_depth.py` edits the same two files first; the
   new script must run after it and anchor on its output. All anchors are text-based and
   fail-closed.
6. **Async scheduling** excluded from the first port (`VLLM_ASYNC_SCHEDULING=0` is our
   default; T2c says async installs batch-wide placeholder lengths).

### A.5 Gates (A1)

- **Correctness (must pass before any throughput claim):** GSM8K 500q paired vs current,
  greedy snippet byte-compare on the standard probe set; output must be byte-identical
  at a fixed seed with the exit on vs off (the mechanism is lossless: fewer *proposed*
  tokens, same rejection sampler). Acceptance/draft per position unchanged within noise.
- **Throughput:** BetterBench (`bench/`), mtp8, conc 1/2/4/8; report ms/step,
  tok/update, combined t/s, and per-position acceptance together. Never one sample per
  arm (acceptance is bimodal on this host).
- **Stability:** no HSA fault at bs=1→2; no cudagraph capture failure; boot clean.
- **Rollback:** flip `RADIANCE_MTP_RAGGED_VERIFY=0` + restart (no rebuild).

---

## 4. Track B — conv spec-scratch zeroing overlay

### B.0 Reachability gate first

The hazard is structurally present (conv width 11 > `state_len` 3 under SPEC=8), but we
must confirm it actually reaches our output path before spending effort:

- Deterministic offline repro: build a spec-decode step where a state block is reused
  from another request (align copy across a page boundary) and the verify reads at
  `naccept > 1`; compare GDN conv output against a reference with the scratch forced
  zero. Run with `RADIANCE_USE_R4D=0` (FLA/Triton) and `=1` (R4D) separately.
- If it does not reproduce on our path, Track B drops to a hardening item and only the
  Triton align-copy fix is worth landing.

### B.1 Triton overlay (runtime, no rebuild)

- **`aijuus/kv-offload/patches/patch_mamba_scratch_zero.py`**:
  - `v1/worker/mamba_utils.py` `_copy_mamba_state_block`: zero destination columns
    `[num_dst_tokens, conv_width)` in both the DS and SD conv branches (and drop the SD
    early return), mirroring dev `:350-371,391-426`. This is the half that runs on our
    stack regardless of R4D (the align state copy stays vLLM Triton).
  - `model_executor/layers/mamba/ops/causal_conv1d.py`: add `conv_state_len`,
    `ZERO_SPARE`, `NP2_SPARE` and the zeroing sites (dev fwd `:263-277,323-335,363-375,
    453-465`; update `:1114-1123`; launcher `:653-654,882,905-906,1352,1463,1488`).
    Covers the FLA/Triton fallback (`RADIANCE_USE_R4D=0`) and any step the R4D hook
    declines.
- Gate `RADIANCE_MAMBA_ZERO_SCRATCH` (default 0), wired into the entrypoint like the
  other aijuus runtime patches.

### B.2 R4D overlay (requires a libr4d rebuild — build-time)

Our GDN layer runs libr4d for the fused conv/recurrent step (`RADIANCE_USE_R4D=1`). The
R4D conv kernels in `r4d_radiance_extras.patch` write only `slen_eff` columns and read
`hist[FU_ST]` from `off = naccept-1` — the same read-past-the-written-column pattern.

- Patch the relevant `r4d_radiance_extras*.patch` (the one matching our `R4D_VERSION`,
  `v0.5.0`, and the `rx9`/`rx10` variants) to zero columns
  `[slen_eff, state_len_max)` after the conv write.
- Build/roll as `serve-mxfp4.sh` already supports (`R4D_PATCH` rebuild at container
  start, or bake the patched `r4d.so` via `Dockerfile:310-317`). This is the only
  rebuild in the whole plan and is **deferred until B.0 proves the R4D path reachable**.
- Gate `RADIANCE_R4D_CONV_ZERO_SCRATCH` if the kernel can take a runtime flag; otherwise
  it ships in the rebuilt `.so` and is compared as an A/B image.

### B.3 Gates (B)

- The offline repro from B.0 flips from mismatch to match.
- Full-model output equivalence: greedy snippet byte-compare vs pre-fix on a fixed
  corpus; GSM8K 500q paired.
- No throughput regression (Triton zeroing adds a store per step; measure step time).

---

## 5. Sequencing

1. **P0 — confirm current state & baselines (no code).** Boot with
   `RADIANCE_DRAFT_PHASE_TIMERS=1`, `RADIANCE_STEP_TRACE`, log scheduled K and verify
   width; record BetterBench mtp8 conc 1/2/4/8 + GSM8K baseline. Run the B.0 repro.
2. **P1 — Track B Triton (B.1).** Small, correctness-first; verify with B.3. Land
   behind `RADIANCE_MAMBA_ZERO_SCRATCH`.
3. **P2 — Track A1.**
   - P2a: length bridge (`DraftTokensHandler`, `outputs.py`) + scheduler consumption —
     inert until a producer exists, fully unit-testable.
   - P2b: speculator per-step confidence + `draft_lens` (eager loop first).
   - P2c: model-runner ragged verify + cudagraph varlen `min_decode_query_len=2`.
   - P2d: end-to-end A/B against P0, behind `RADIANCE_MTP_RAGGED_VERIFY`.
4. **P3 — Track B R4D (B.2)** only if B.0 reproduced and A1 landed (the ragged verify
   increases the variety of conv windows, so the fix should precede or accompany it).
5. **P4 — optional A3** acceptance estimator + config/sampling, only if A1 gate is
   insufficient.

Stop rule: if A1 does not move conc-8 combined t/s at matched acceptance by a clear
margin, the verify width is not the binding constraint on our workload; keep B and stop
A, and revisit `MTP-SPEEDUP-PLAN.md` T3/T5.

---

## 6. Testing conventions (this repo)

- Paired gates, never single-arm: GSM8K 500q + greedy snippet byte-compare +
  `RADIANCE_MXFP4_CHECKALL` where applicable (project convention: no perplexity
  byte-compares).
- BetterBench is the benchmark (`bench/`), full BetterBench before landing.
- Warmup and ≥20 passes/category; both thinking modes where applicable; card 1.
- Every runtime patch is idempotent, marker-guarded, `ast.parse`-checked, and fails
  closed on anchor drift.
- User owns restarts/redeploys — this plan never runs them.

---

## 7. Open decisions for the user

1. **Track A confidence source:** raw top-1 (A1, small, matches `RADIANCE_DRAFT_TAU`) or
   tcclaviger-parity calibrated estimator (A3, +19 KB module + config/sampling)? Default:
   A1 first, A3 only if needed.
2. **Track A step loop:** port the graphed `_step_graph_multi_step_decode` (faithful,
   harder) or start with the eager per-step loop (simpler, some launch overhead)?
3. **Track B R4D rebuild:** acceptable to ship a libr4d `.so` rebuild for the R4D half,
   or Triton-only until B.0 forces it?
4. **Overlay placement:** runtime `aijuus/kv-offload/patches/*.py` (recommended, no
   rebuild) vs build-time `aijuus/patches/*.patch`.

---

## 8. A/B results (2026-09-30, vllm-0) — Track A does not pay here

Both halves of Track A were implemented and measured on `mtp-27B-MXFP4-blend` (MAXSEQS=8),
conc 8 (`aijuus/bench-conc.py`) and single-stream (`aijuus/bench-quick.py`).

| Arm | conc-8 agg tok/s | single-stream combined | note |
|---|--:|--:|---|
| baseline (no override) | 361.9 / 364.0 | 103.9 | two baseline runs |
| fork ragged verify only (`RADIANCE_DYNAMIC_WIDTH=1`) | 368.8 (+1.9%) | n/a (gated <3 running) | within spread |
| ported confidence early-exit (`patch_mtp_conf_exit.py`, tau 0.5) | 351.6 (-2.8%) | 95.1 (-8.5%) | dormant after revert |

Mechanism worked as designed (per-row survival; bs8 drops width 4→2, bs2 keeps 5), but cutting
drafts below the batch-size schedule only loses tokens: the target forward is weight-stream-bound
at this concurrency, `ms/step` stays flat/up, and proposer depth has a hard geometry cliff below
k=3 (WORKLOG cont.26). The batch schedule (`spec_schedule` k=5/4) already sits at the
per-concurrency optimum, so a content-adaptive rule has no headroom.

**Status:** Track A closed as not transferable. `patch_mtp_conf_exit.py` remains in the overlay,
env-gated (`RADIANCE_MTP_CONF_EXIT=1`, default off) for a higher-concurrency/calibrated-confidence
revisit. Track B (conv spec-scratch zeroing) is still open and is a correctness item, not a
throughput one.

## 9. Track B A/B result (2026-09-30) — align-copy zeroing is a -55% regression

`patch_mamba_scratch_zero.py` (SD+DS conv tails in `_copy_mamba_state_block`), gated
`RADIANCE_MAMBA_ZERO_SCRATCH=1`, booted clean with acceptance/output unchanged, but conc-8
aggregate fell **361.9 -> 161.8 tok/s (-55%)`. Layout is SD only (DS branch dead), so the cost is a
Triton occupancy/recompile side effect of the added dynamic-range loop, not store volume. Reverted;
baseline re-confirmed 363.9.

Track B must therefore be either (a) a cheap vectorized tail store, or (b) the conv-write side
(`causal_conv1d` `ZERO_SPARE`, HIP `gdn_hip`/`clav_conv1d`) — and only after reachability of the
stale scratch is proven on our R4D GDN path (an offline repro was not run). Left dormant.
