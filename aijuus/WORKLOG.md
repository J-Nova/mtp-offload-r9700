# AIJUUS work log

A running, dated log of what was changed, why, and how it was verified. Newest entry first.
Complements (does not replace) `TCCLA-VLLM-MTP-RESEARCH.md` and
`TCCLA-VLLM-MTP-IMPLEMENTATION-PLAN.md`, which hold the analysis and the plan.

## 2026-09-30 (cont. 33) — Reorder batch threshold corruption analysis + overlay fix plan (vLLM #55894 / PR #55898)

### Problem
`GPUModelRunner.calculate_reorder_batch_threshold` takes `min` across all attention groups'
`reorder_batch_threshold`. FlashInfer reports 1; GDN/Mamba report `1+k` (k = num_spec_tokens).
With MTP spec-decode + a threshold-1 backend (FlashInfer), the global threshold drops to 1,
causing draft-decode rows to be misclassified as prefills → state slot corruption → garbage output.
Upstream issue: [vLLM #55894](https://github.com/vllm-project/vllm/issues/55894); fix PR:
[vLLM #55898](https://github.com/vllm-project/vllm/pull/55898) (open since Sep 8, 2026, 22 days,
awaiting review from 10 code owners, no approvals).

### Current deployment: safe by configuration
- `serve-mxfp4.sh:784` pins target to `ATTN=R4D`
- `serve-mxfp4.sh:438` pins dflash drafter to `TRITON_ATTN`
- `serve-mxfp4.sh:831` pins MTP drafter to `ATTN=R4D`
- R4D (`radiance_r4d_attn.py:146`) subclasses `TritonAttentionMetadataBuilder`; neither sets
  `reorder_batch_threshold` → inherits `None` from `backend.py:592`
- GDN reports `1+SPEC` (5 or 8)
- So global threshold stays at `1+SPEC` — no corruption under current config
- Production entrypoint (`aijuus/kv-offload/ops/entrypoint.sh:177,179`) also hardcodes R4D/TRITON_ATTN

### PR #55898 fix (verified via diff)
Adds `requires_decode_ordering: bool = False` to `AttentionMetadataBuilder` base class; sets `True`
on GDN/Linear/Mamba builders; modifies `calculate_reorder_batch_threshold` to use
`max(min(all_thresholds), max(required_thresholds))` instead of plain `min`.

### Recommended approach: runtime patch script (Option 2)
Create `aijuus/kv-offload/patches/patch_reorder_threshold.py` (not a build-time overlay patch in
`aijuus/patches/`):
- **Why runtime**: no Docker rebuild needed; consistent with existing pattern
  (`patch_gdn_metadata.py`, `patch_dynamic_depth.py`, etc.); easier to remove when upstream merges
- **Defensive value**: doesn't change current behavior (threshold already 5/8) but protects against
  configuration drift (drafter backend change to FlashInfer, new threshold-1 backend, etc.)
- **Risk**: patch may conflict if upstream changes same files before PR merges; standard aijuus
  revert workflow handles removal when PR lands

### Plan (6 steps, code not yet written — user said "do not edit code yet")
1. Create `aijuus/kv-offload/patches/patch_reorder_threshold.py`: idempotent Python script that
   (a) adds `requires_decode_ordering: bool = False` to `AttentionMetadataBuilder` in `backend.py`,
   (b) sets `True` on GDN/Linear/Mamba builders, (c) modifies `calculate_reorder_batch_threshold`
   in `gpu_model_runner.py` to use `max(min(all), max(required))` logic, (d) includes logger.info
   when threshold is raised
2. Verify clean apply against current vLLM source
3. Test CPU regression
4. Deploy (user restarts containers)
5. Monitor upstream PR #55898; remove patch when it merges
6. Add documentation to WORKLOG and plan doc

### Relevant files
- `vllm/v1/worker/gpu_model_runner.py:7308-7326`: `calculate_reorder_batch_threshold`
- `vllm/v1/attention/backend.py:592`: `AttentionMetadataBuilder` base class
- `vllm/v1/attention/backends/gdn_attn.py`: `GDNAttentionMetadataBuilder`
- `vllm/v1/attention/backends/mamba_attn.py`: `BaseMambaAttentionMetadataBuilder`
- `vllm/v1/attention/backends/linear_attn.py`: `LinearAttentionMetadataBuilder`
- `aijuus/kv-offload/patches/patch_gdn_metadata.py`: existing runtime patch pattern to follow

---

## 2026-09-30 (cont. 32) — B4: R4D vs AITER attention A/B prep

**Task B4** from `MTP-PREFILL-PLAN.md`: A/B `R4D_ATTN=0` (AITER unified attention) vs
`R4D_ATTN=1` (R4D paged attention, default) at h256. Baseline (R4D_ATTN=1, vllm-0,
BetterBench prefill sweep): **1813 / 1995 / 2117 / 2050 / 1909 PP t/s** at 2k/8k/16k/32k/64k.

### Current state
- vllm-0: ThinkingCap (ready), vllm-1: MTP (ready). User requested MTP on both cards via
  `POST /load` (controller needs restart first to pick up the new endpoint).
- `R4D_ATTN` not explicitly set in compose or registry; defaults to `1` (R4D).

### Plan
1. ~~Add `R4D_ATTN=0` to compose env for both vllm services (after `R4D_ATTN_FP8=3`).~~ **DONE** (both vllm-0 and vllm-1 blocks).
2. User **redeploys** the compose (Coolify) to activate `R4D_ATTN=0` (AITER attention). A `/reload` won't pick up compose env changes.
3. Run BetterBench prefill sweep on the MTP card with `R4D_ATTN=0`.
4. Revert to `R4D_ATTN=1` (remove the line or set to 1).
5. User redeploys again.
6. Run BetterBench prefill sweep with `R4D_ATTN=1` (confirm baseline).
7. Compare and log results.

### Current state (2026-09-30 10:05)
- vllm-0: ThinkingCap (ready), vllm-1: MTP (ready).
- Compose now has `R4D_ATTN=0` for both vllm services (AITER attention arm).
- Next: user redeploy + run prefill sweep.

### Notes
- `R4D_ATTN` is read by the vLLM/radiance stack to select the attention backend:
  `1` = R4D paged attention (default, +37.8% prefill at 260k vs AITER per README),
  `0` = AITER unified attention (`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` already set).
- This is a config-only change; no code changes needed. One reload per arm.

## 2026-09-30 (cont. 31) — model-controller: new `POST /load` (load a model onto cards that don't have it)

The controller previously had no way to *change* which model an instance serves over
HTTP: `POST /reload` keeps `state["model"]` and only re-reads registry settings, and the
only swap path was the router-written `trigger.json` (`handle_trigger`, no HTTP surface,
and it no-ops when the model is already ready anywhere). Added a direct admin route:

- `POST /load  {"model": <key>, "instance": <name|"auto"|"all">, "force": bool}`
  (query-string equivalents also accepted). Loads `model` onto instances that are not
  already serving it.
  - `instance` omitted / `"auto"` -> one instance NOT already serving the model, idle-first
    with the same rotation/MRU preference as `pick_target`.
  - `instance="all"` -> both cards (each skipped if already serving it).
  - `instance=<service>` -> just that card.
  - `force=true` -> restart even an instance already serving the model (settings refresh,
    same effect as `/reload` for that card).
  - Response: `{"ok", "model", "targets":[{"instance","ok"[, "already_loaded"|"error"]}]}`;
    `{"ok":true,"already_loaded":true}` when nothing needed a restart. Unknown model ->
    `registry` list; unknown instance -> `instances` list.

Implementation reuses the swap machinery: `_resolve_load_targets()` (target choice with
`_actual_key()` reading the live `/v1/models` via `served_to_key`), then per target
`state={model,ready:false}` -> drain (`wait_until_idle`) -> `docker restart` -> poll
`/health` -> `ready:true`. Serialized under the existing `_op_lock` (same as `/reload`).
`do_reload` unchanged; `do_POST` now routes `/load` and `/reload` (trailing slash tolerated).

Verified offline: `py_compile`; unit-tested `_resolve_load_targets` (explicit / unknown /
all / auto-with-both-free / auto-one-serving / auto-all-serving) and `do_load`
(unknown model/instance errors; `all` with both already serving -> 0 restarts; `auto`/`all`/
`force` restart counts and ready flips correct). Live use needs a controller restart (user-owned):
`curl -sS -X POST -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
 -d '{"model":"<key>"}' http://172.18.0.10:8101/load`.

## 2026-09-29 (cont. 30) — model-router load balancing: RR tie-break, metrics-aware load, prefix-affinity (flagged), shared-tier finding

Plan: ROUTER-LB. Implemented in `aijuus/model-router.py` (bind-mounted to the router
`/code/model-router.py`, so a **router restart** activates it; no redeploy).

### W1 round-robin tie-break (ON by default, `RADIANCE_LB_RR=1`)
- New `order_targets()` replaces `ordered()` on every routing path. It least-load-sorts,
  then rotates the equally-loaded head group with `_rr` under `_lock`.
- Fixes the old lexicographic tie-break (`ordered()` sorted `(inflight, endpoint)`), which
  sent **all** non-overlapping/sequential traffic to vllm-0 while vllm-1 idled.

### W2 metrics-aware load (ON by default, `RADIANCE_LB_METRICS=1`)
- New `metrics_loop` scrapes each endpoint `/metrics` every `RADIANCE_LB_METRICS_INTERVAL`
  (1 s) and stores `{running, waiting, kv, ts}`. `load_key()` orders by
  `(running, waiting, kv_cache_usage_perc, inflight)` when fresh
  (`RADIANCE_LB_METRICS_STALE` 3 s), else falls back to the connection count `_inflight`.
- Metrics are unauthenticated; scrape failure degrades gracefully to connection counts.
- `/metrics` also now exposes `radiance_router_load{endpoint}` and
  `radiance_router_dispatched_total{endpoint}` (actual commits per card) for validation.

### W3 prefix-hash affinity (OFF by default, `RADIANCE_LB_AFFINITY=0`)
- Implemented (`_affinity_key` = sha1 of system + first user turn; `_affinity_target` =
  rendezvous/highest-random-weight over ready endpoints; `apply_affinity` promotes the
  affinity endpoint only when within `RADIANCE_LB_AFFINITY_SLACK` of the least-loaded).
- Left **off**: see W4 — the shared fs tier already gives cross-card reuse, so affinity
  only saves a disk read, not a recompute. Enable if local-HBM TTFT matters more than balance.

### W4 shared fs KV tier gives cross-instance prefix reuse (verified)
- Same 3619-token prefix: sent to vllm-0 (cold), then to vllm-1 (which had never seen it).
  vllm-1 delta: `external_prefix_cache_hits_total +2640`, `prefix_cache_hits_total +2640`
  on the **first** card-1 request → the fs tier (`root_dir=/kvcache/blocks`) served the
  prefix from disk. So prefix caching **is live under MTP spec-decode here** (contradicts the
  earlier "inert under spec-decode #54360" note for this build).
- Implications: LB affinity is optional; a cross-card miss is a disk load, not a recompute.

### W5 harness
- `aijuus/tools/mtp-conc-bench.py`: added `--no-metrics` (LB endpoints don't proxy
  `/metrics`; accept% prints `n/a`) and `--instances url0,url1` to print the per-card split
  (from each instance's `vllm:prompt_tokens_total` delta).

### Status / validation (2026-09-29)
- Activated: router restarted twice (the first restart exposed a bug — `_rr += 1` without `global _rr`
  in `order_targets` → `UnboundLocalError` on every POST; fixed, offline unit-tested).
- **W1 verified**: LB `conc=1 × 20` sequential → `radiance_router_dispatched_total` = vllm-0 11 / vllm-1 10
  (was 20/0 under the old lexicographic tie-break). Bench `--instances` split 50%/50%.
- **W2 active**: `radiance_router_load` present; under a cold `conc=16` burst the router put **7–8**
  concurrent requests on **each** card (sampled `num_requests_running`), i.e. correct balancing.
- **Aggregate**: warm-cache burst reached **835 tok/s ≈ 1.95× one card** (single card ~428);
  cold (`--unique`, no cache reuse) ~450–540, and single-card unique ~401–417. So the LB distributes
  correctly and the aggregate ceiling is **host-side** (CPU/power/disk + prefill work), not the router.
  Throughput is dominated by prefix-cache warmth, so warm vs cold runs must not be compared.
- Router logs: all 200s, no tracebacks after the fix. Bench now tolerates per-request failures
  (prints `[warn] n/m failed`) and has `--unique`.
- Minor hardening added but **not yet live** (needs one more router restart): `LISTEN_BACKLOG=128`
  (`ThreadingHTTPServer.request_queue_size` default 5 caused a couple of client resets at conc≥20).

---



## 2026-09-29 (cont. 29) — TOP1 draft-head arm is BROKEN on vLLM 0.29 (MTP boot-fail); swap trigger traced; MTP re-homed to vllm-0

### `RADIANCE_DRAFT_HEAD_TOP1=1` crash-loops MTP at engine init
- With the arm staged (`RADIANCE_DRAFT_HEAD_TOP1=1`, `RADIANCE_DRAFT_VOCAB` removed) MTP fails **during
  `profile_run`** -> `speculator.propose` -> `_prefill` -> `_greedy_sample_draft` -> `compute_logits`:
  `vllm/model_executor/layers/logits_processor.py:198  logits = logits[..., : self.org_vocab_size]`
  raises `TypeError: tuple indices must be integers or slices, not tuple`.
- The fused top-1 path makes the head return a tuple, but vLLM's `LogitsProcessor` still expects a tensor
  from `lm_head`. Boot log confirms the **full** head is used (`draft head (248320, 5120)`), because the
  fused top-1 path declines the vocab prune (`radiance_drafthead.py:672-674`).
- Consequence: container crash-loops; MTP cannot boot until `HEAD_TOP1` is back to `0`.
- **Action taken:** MTP registry entry reverted to the validated arm — `RADIANCE_DRAFT_HEAD_TOP1=0`,
  `RADIANCE_DRAFT_VOCAB=/patches/aijuus/draft_keep/keep-union-freq.txt`, `RADIANCE_DRAFT_NGRAM=0`.
  vllm-0 then booted clean and benched on baseline (88.4/155.8/253.2/427.0). See backlog item for the
  proper fix.

### Swap-trigger traced: who moved vllm-0 to ThinkingCap (21:07:54Z)
- An **authenticated** `POST /v1/chat/completions` (Bearer key) with `model=ThinkingCap-…` reached
  model-router :8100 via the public host **`ai.ragfodder.com`** (Coolify Traefik 172.18.0.8, fronted by the
  `*.ragfodder.com` cloudflared tunnel). Router (by design) wrote `/model-state/trigger.json`, served that
  request with the loaded MTP model (fallback), and model-controller swapped **vllm-0 -> ThinkingCap**
  (`swap start` 21:07:54, `done` 21:17:11). `usage.json` records ThinkingCap at 1790716074.39.
- **Caller identity is not recoverable from logs**: Traefik access logs are disabled and the router logs
  only its direct peer (the proxy), discarding `X-Forwarded-For`/`CF-Connecting-IP`. Not open-webui
  (its `webui.db` has no users/chats); not the manual reloads (those came from 172.18.0.1).

### MTP re-homed to vllm-0
- vllm-0 stopped; state cleared; controller re-seeded; vllm-0 booted MTP. vllm-1 also still serves MTP.
- `model-controller.reconcile()` reverts a manual `state.json` edit within one TICK (5 s) while the
  instance is up, so a direct state edit + reload races; deleting state (root-owned, `sudo`) and letting
  the controller re-seed is the reliable path.

---

## 2026-09-29 (cont. 28) — MTP validation: n-gram tail HSA fault at bs>=2 (2 fixes failed), schedule re-confirmed, vocab A/B a wash

Ran on vllm-0 (MTP blend), clean caches, `RADIANCE_COLLECT_TOKENS=0`.

### Baseline (NGRAM=0, keep-union-freq, schedule [[1,2,5],[3,8,4]])
conc 1/2/4/8 tok/s **88.3 / 155.8 / 264.4 / 421.9** (accept 56.4/55.4/58.0/55.3).
Matches/beats cont.26 (88.3/155.7/252.6/423.8) -> **dynamic depth intact**.

### N-gram tail = deterministic bs>=2 crash (blocker)
- `RADIANCE_DRAFT_NGRAM=1` faults at the **bs=1->bs=2 transition** with
  `HSA_STATUS_ERROR_EXCEPTION` (GPU hardware exception; `Queue error` + coredump attempt), right after
  `[sd-trace] bs=2 nspec_sched=5`. Reproduced on **both cards**, with a **fully cleared cache** and the
  collector off, and ThinkingCap uninvolved. bs=1 is fine (~23 proposals). The engine wedges
  (`num_requests_running=2`, 0 tok/s, `/health` still 200) -> a restart is required after each attempt.
- The tail (`_radiance_ngram_extend`) is **not in HEAD** — it is part of the uncommitted 154-line
  `patch_dynamic_depth.py` diff — and was absent from the cont.26 dynamic-depth validation.
- **Fix attempt 1 — fixed width**: take the n-gram only when `clen >= K` and REPLACE the K MTP drafts,
  never append past K (the old `di += c[kk:cl]`, up to `num_speculative_steps`). **FAILED** — still HSA
  at bs=2 with the row width already == K.
- **Fix attempt 2 — per-row matcher**: run `match_gpu` once per row at **B=1** (the proven-safe path)
  instead of one B=R launch. **FAILED** — still HSA at bs=2.
- **Conclusion**: the fault is neither the draft-row width nor the matcher launch shape — merely
  *executing the matcher during a bs>=2 draft step* faults, independent of B. Needs a dedicated
  **standalone `match_gpu` repro** (B=1/B=2 off-server, serving stopped) before it can be re-enabled.
  `NGRAM` kept **0**. Both safety changes (fixed-width + per-row) are left in the overlay (inert while off).

### Schedule refinement: boundary stays at 3
- bs=3 at **k=4** (current [[1,2,5],[3,8,4]]): **198.2 / 55.3%**.
- bs=3 at **k=5** ([[1,3,5],[4,8,4]]): 193.0 / 50.1%.
- k=4 wins -> keep `[[1,2,5],[3,8,4]]` (reconfirms cont.26).

### Vocab A/B (head-only, one reload per arm)
keep-union-freq (65327 rows) vs keep-union-v2 (65133 rows), conc 1/2/4/8 tok/s (accept):
- **freq**: 88.3 / 155.8 / 264.4 / 421.9 (56.4 / 55.4 / 58.0 / 55.3)
- **v2**:   87.9 / 152.7 / 264.6 / 430.3 (56.4 / 55.0 / 57.9 / 55.4)
- **No clear winner**: freq wins bs1-2 (latency-critical), tie at bs4, v2 +~2% at bs8 (within noise —
  freq measured 421.9/423.8 across runs). **Kept `keep-union-freq.txt`**.

### Other
- `RADIANCE_COLLECT_TOKENS=0` forced on the MTP entry: the B2 token-collector wraps the MTP draft
  head `compute_logits` and does `argmax` + a host sync **every eager draft step**, and was rewriting
  `/patches/aijuus/draft_keep/rank1.json` during serving. Off for all measurements.
- Cleared all stale host state before this run: `/var/lib/radiance-model-state/*` (state/usage/trigger),
  the 60 GiB fs-KV tier, the 20 GiB compile cache, and the collector artifacts.

### Residuals / remaining backlog (do not forget)
1. **N-gram tail offline repro** — run `match_gpu` standalone at B=1/B=2 with a serving instance stopped,
   compare against the in-engine bs=2 context, find the faulty kernel, then re-enable `NGRAM=1` + validate.
2. **End A/B battery** on the frozen union-freq build (one reload per arm): `EXACTSET+FUSED` on/off;
   AITER on/off (`VLLM_ROCM_USE_AITER` 1 vs 0); SPEC 4 vs 8 re-run.
3. **Untested MTP perf levers** (one reload per arm): `RADIANCE_DRAFT_TAU` 0.15/0.25;
   `RADIANCE_DRAFT_RERANK` (32 vs other); capture-ladder trim (dense 26 sizes vs `[4,8,12,16,20,24,28,32]`).
   (`RADIANCE_DRAFT_HEAD_TOP1` is **BROKEN** — see item 3b.)
3b. **FIX `RADIANCE_DRAFT_HEAD_TOP1`** (cont.29): the fused int2 top-1 draft head returns a **tuple** from
   `compute_logits`, but vLLM 0.29 `LogitsProcessor._get_logits` (`logits_processor.py:198`) still slices
   the head output as a tensor -> `TypeError: tuple indices must be integers or slices, not tuple` at
   engine init. Either return a tensor / adjust the gadget to vLLM's LogitsProcessor contract, or drop the
   arm. NOTE it is mutually exclusive with the vocab prune (fused top-1 declines `RADIANCE_DRAFT_VOCAB`,
   keeps the full 248320-row head, `radiance_drafthead.py:672-674`). It must NOT be left enabled in the
   registry (a plain reload of MTP would crash-loop).
4. **KV calibration** — `./calibrate-kv.sh SPEC_METHOD=mtp SPEC=8` to replace the MTP entry's interim
   borrowed pin `8761733283` (needs a GPU run / serving stopped).
5. **Dflash pilot pins** — `Qwen3.8-27B-MXFP4-mtpfp8` and both blend-dflash entries still carry the
   8.16 GiB pin; port 6.0 GiB or recalibrate, else they OOM if served.
6. **flash_attn 2.8.3** install + A/B against R4D/AITER attention (image-level, plan §3.2).
7. **Housekeeping** — commit/track the uncommitted work (`patch_dynamic_depth.py`, `radiance_w4.py`,
   `model-registry.json`, `WORKLOG.md`); decide the n-gram code (keep inert vs revert).
Out of scope: DRY (adapted, not adopted), degen (live), ROCm 10 (deferred), KV prefix-cache inert under
spec-decode (upstream #54360).

---



## 2026-09-29 (cont. 27) — DFlash boot incident (W4 over-band crash, AOT envkey, KV pin, expandable_segments); MTP impact nil

The controller moved **both** vllm-0/vllm-1 to `ThinkingCap-Qwen3.8-27B-MXFP4-OCP-GPTQ` (dflash/16),
which then crash-looped through four distinct failures. All fixed; it boots. (dflash path only.)

### Failures and fixes
1. **W4 over-band compile crash.** `ConstraintViolationError` at `radiance_w4.py:362`, the
   `torch.cat([... for i in range(0, m, _MAX_M)])` chunk loop: `range()` over a SymInt specialized
   the token dim. `kernel_projection` (1280x5120) and `candidate_selector.hidden_projection`
   (256x5120) — the drafter's unquantised bf16 linears, listed in `_CFG_UNQUANT`/`_CFG_UNQUANT_A8`
   — see the **whole draft block** `num_reqs*(1+nspec) = 8*17 = 136` rows, past `_MAX_M=64`.
   Two W4-preserving fixes were **rejected**: a `radiance::w4_linear_chunked` custom op and a static
   `torch.tensor_split(x2, 8)` both hit `AssertionError: Expected tensors only, but got: <class
   'int'>` in inductor `copy_misaligned_inputs` (`torch/_inductor/utils.py:3442`) — any W4 chunking
   of an over-band drafter linear corrupts the piecewise graph input list in 0.29. **Fix:** emptied
   `_CFG_UNQUANT`/`_CFG_UNQUANT_A8` (comment in-file); those layers stay bf16. Boot then logs
   `[radiance.w4] declined N=… (no measured config)` and the drafter AOT compiles. Re-enabling needs
   an r4d kernel band above the draft block (`GEMM_W4_MAX_M >= ~136`), not an overlay change.
2. **Stale AOT artifact.** `patch_aot_envkey.py` keys the AOT dir on
   `sha256(sorted RADIANCE_* env)[:12]`; it excludes overlay source, so the old broken artifact
   was reused. **Fix:** `RADIANCE_LOCAL_AOT_EPOCH=1` in `_defaults.server_env` →
   env key `2184c1ceca86` → `640b6393d1be`, fresh compile (one-time recompile for every model).
3. **KV alloc OOM** at the shipped 8.16 GiB pin (model ~20.1 GiB + drafter + ~2.5 GiB reserved).
   Lowered to 7.5 GiB → KV allocated but then the **FULL cudagraph capture OOM'd** (20 MiB
   `mxfp4_linear_pq` alloc, 2.58 GiB reserved-but-unallocated).
4. **`expandable_segments` is structurally banned here.** `PYTORCH_CUDA_ALLOC_CONF=
   expandable_segments:True` → vLLM 0.29 `VllmConfig` value_error: incompatible with the
   **OffloadingConnector** unless `enable_cumem_allocator` is also on (the VMM allocator remaps the
   registered/pinned KV). Removed; the real fix is headroom, not a different allocator.
5. **KV sizing.** Lowered `max_model_len` 160000 → 120000 and set the pin from a naive ~44.2k
   B/tok estimate to 5308440000 (~4.94 GiB) — **too small**: vLLM refused at
   `_check_enough_kv_cache_memory` (`120000 needs 5.36 GiB, available 4.93 GiB; estimated max
   length 105600`). Real cost is ~48–50k B/tok (radiance KV-group pick size 8 → 366 blocks/request,
   + 3 padding layers "may waste up to 60%"). **Final pin 6442450944 (6.0 GiB).**

### MTP impact: none functional
- The emptied `_CFG_UNQUANT` shapes belong to the **DFlash drafter** (`DFlashGroupedConv.
  kernel_projection` + `candidate_selector.hidden_projection`). The MTP blend uses the MTP head and
  loads **no drafter**, so it never instantiates them → the MTP W4 path is unchanged.
- The only change reaching MTP is `RADIANCE_LOCAL_AOT_EPOCH` (a pure hash-salt): it forces one fresh
  AOT recompile on the next MTP boot (slower startup), with **no behaviour or throughput change**.
  The env key is stable once set, so it is a one-time invalidation.
- `max_model_len`/`kv_cache_memory` edits are per-entry (`ThinkingCap…` only); the MTP entry is untouched.

### Residuals
- Other dflash entries (`Qwen3.8-27B-MXFP4-mtpfp8`, both blend-dflash) still carry the 8.16 GiB pin
  and will OOM if served; port the 6.0 GiB pin or recalibrate.
- MTP combined **depth + `[ngram]` tail validation still pending** (was interrupted); see cont. 28.

---

## 2026-09-29 (cont. 26) — Dynamic SD made real on V2: headroom measured, depth overlay implemented

- **Headroom (static depth sweep, `RADIANCE_DYNAMIC_WIDTH=1` active throughout)**, conc 1/2/4/8 tok/s:
  k=8 80.1/140.4/227.4/361.0 · k=6 85.9/150.3/236.7/383.9 · **k=5 88.2/155.5/251.9/406.8** ·
  **k=4 86.2/153.1/266.7/418.8** · k=3 24.0/47.5/85.9/157.1 (a reproducible ~3.5x cliff, avoid).
  Optimum is concurrency-dependent: **bs 1-2 → k=5, bs 3-8 → k=4**. Acceptance rises as depth drops
  (36.9%→56.4% at conc1), i.e. the deep positions were mostly wasted. These numbers were *with*
  `RADIANCE_DYNAMIC_WIDTH=1` already on, so the dominant lever is proposer depth (MTP forward steps),
  not verify width.
- **Why the vLLM feature can't do it**: `num_spec_tokens_to_schedule` is read only by the V1 runner
  (`gpu_model_runner.py`) and async scheduler; the live V2 runner ignores it (cont.25). The scheduler
  computes it correctly regardless.
- **Enabler**: probe shows MTP runs the V2 runner on the NON-FUSED speculator path
  (`V2 propose ENTERED … fused=False steps=8 adv=True`), so draft depth is a plain Python
  `for step in range(1, num_speculative_steps)` over per-step FULL-graph replays. `DraftTokensHandler`
  already sizes on `draft_tokens.shape[1]` and the scheduler schedules `request.spec_token_ids` as
  returned, so a K-wide draft list is a supported shape (this is how `patch_dynwidth.py` already
  varies verify width).
- **New overlay `patch_dynamic_depth.py`** (gated `RADIANCE_DYNAMIC_DEPTH=1`, idempotent, `ast.parse`
  guarded): (1) V2 runner remembers `SchedulerOutput.num_spec_tokens_to_schedule`, hands it to the
  speculator before `propose`, and passes only the first K draft columns to `DraftTokensHandler`;
  (2) speculator bounds `_multi_step_decode` (and the fused loop) to K via `_radiance_eff_k`, with a
  `[dyn-depth] propose eff_k=…` once-per-distinct-value diagnostic. Fused graphs bake depth, so when
  `use_fused_multi_step_decode` is True the overlay is a no-op at full depth (never a graph mismatch).
- **Wiring**: entrypoint re-adds `SPEC_SCHEDULE`/`SPEC_SCHED_ARG` and applies
  `patch_dynamic_sd_cudagraph.py` + `patch_sd_sched_trace.py` when a schedule is set, and
  `patch_dynamic_depth.py` when `RADIANCE_DYNAMIC_DEPTH=1`. Registry (MTP): `spec_tokens=8` (ceiling),
  `spec_schedule=[[1,2,5],[3,8,4]]`, `RADIANCE_DYNAMIC_DEPTH=1`, FUSED/EXACTSET=1.
- **VALIDATED (cont.26, live)**: dynamic depth works end to end. `[dyn-depth] propose# … k=5 num_reqs=1
  / k=4 num_reqs=8 fused=False sched=True`. Sweep (conc 1/2/3/4/8, reps 2–3, mt=400):
  **dynamic 88.3 / 155.7 / 184.7 / 252.6 / 423.8 tok/s** vs static k=8 **80.1 / 140.4 / — / 227.4 /
  361.0** → **+10.2% / +10.9% / — / +11.1% / +17.4%**, and it matches the per-conc static optima at
  bs1-2 (k=5) and bs8 (k=4). Acceptance steady ~55-56%. Measured draft width at bs8 = 3.88 tokens/draft
  (K=4 applied). Correctness sanity: greedy `17*23` → `391` (correct), coherent reasoning. Stable across
  re-runs (conc4 252.3-253.0, conc8 423.3-424.3).
- **Implementation note (v1 → v2)**: v1 forwarded `SchedulerOutput.num_spec_tokens_to_schedule` from
  the runner into `speculator._radiance_dyn_k`; probes showed the runner received K=4/5 but `propose`
  still saw 0/8 on the same object id, so the hand-off was dropped (cause not fully isolated; the two
  speculator instances, one `fused=True` one `fused=False`, made it fragile). v2 is **self-contained**:
  the speculator builds the batch→K lookup from its own `vllm_config.speculative_config` and derives K
  from `input_batch.num_reqs`; the runner only slices the returned drafts to K. v2 also **forces
  `use_fused_multi_step_decode=False`** whenever a schedule is set, so capture uses per-step graphs and
  depth is Python-controlled on every instance (fused graphs bake depth). Debug probe files removed.
- **Rollback**: set `RADIANCE_DYNAMIC_DEPTH=0` and `spec_schedule=""`. Note: the overlays edit the
  container filesystem; a Coolify redeploy re-applies from the image cleanly (during dev we reset the
  two files to pristine from the image before re-applying).

---

## 2026-09-29 (cont. 25) — Deep review of dynamic SD: it is inert by construction on the V2 runner

Root cause established by code trace (not just measurement). The plumbing is not miswired — vLLM 0.29
implements dynamic SD for the *V1* GPU runner and async scheduler only; the live V2 runner drops it.

- **Path**: registry `spec_schedule` → entrypoint `SPEC_SCHEDULE`/`SPEC_CFG`
  (`num_speculative_tokens_per_batch_size`) → `SpeculativeConfig` (field only; `_verify_args` does NOT
  validate it) → `Scheduler.__init__` builds `self.dynamic_sd_lookup =
  build_dynamic_sd_schedule_lookup(sched, max_num_seqs, num_spec_tokens)` → per step,
  `num_spec_tokens_to_schedule = dynamic_sd_lookup[len(num_scheduled_tokens)]` → placed in
  `SchedulerOutput`.
- **Dead end**: `SchedulerOutput.num_spec_tokens_to_schedule` is consumed ONLY by
  `v1/core/sched/async_scheduler.py:25` and `v1/worker/gpu_model_runner.py` (the **V1** runner:
  lines 5096/5105/5139/5161/5188/5206/5354, which pass it as `num_speculative_tokens=`).
  `v1/worker/gpu_worker.py:457-476` picks `vllm.v1.worker.gpu.model_runner.GPUModelRunner` when
  `use_v2_model_runner` (our case, log "Using V2 Model Runner"), and **that file has ZERO references to
  `num_spec_tokens_to_schedule`**. `async_scheduling=False` here, so the async consumer is off too.
- **Draft depth** is fixed at `self.num_speculative_steps = vllm_config.num_speculative_tokens` (=8) in
  the V2 speculator; the draft count fed to the target comes from `scheduled_spec_decode_tokens`
  (previous step's actual drafts), never from the schedule. So neither draft depth nor verification
  width changes on the live path.
- **Only real effects of arming it**: (a) `scheduler.py:1103` disables decode-request padding
  (`pad_spec_decode`) whenever `dynamic_sd_lookup is not None` — a *negative* for full-cudagraph
  uniformity; (b) `cudagraph_utils._init_candidates` takes the dynamic branch and expands candidate
  query lengths, which is what crashed the speculator decode manager (our overlay patch makes it not
  crash, but the expanded values are unused on V2).
- **Unit tests** (`aijuus/tools/test_dynamic_sd.py`, run in a throwaway container): all pass —
  schedule validation (accepts ours; rejects None/empty/short/start-0/overlap/starts-at-2/negative-K),
  lookup `[0,8,7,7,6,6,6,6,5]` for max_num_seqs=8 (+ gap carry-forward, K clamp), the cudagraph
  formula (unpatched speculator manager `{-2,-1,0,1}` → `round_up(...,0)` raises; patched `{1}`; main
  runner unchanged `{6,7,8,9}`), and the installed-guard presence checks.
- **Conclusion**: keep `spec_schedule` off; the re-test's remaining value is only the integration
  confirmation that the schedule reaches the scheduler (`[sd-trace]`) while throughput stays unchanged.
  A working dynamic depth would require the V2 runner/speculator to consume `num_spec_tokens_to_schedule`
  (a vLLM-side change), not a config/plumbing change on our side.
- **Integration evidence (rigorous re-test, cont.25)**: Arm A boot (schedule armed, FUSED/EXACTSET=1) →
  `[sd-trace]` logs `bs=1 nspec_sched=8`, `bs=2→7`, `bs=3→7`, `bs=4..7→6`, `bs=8→5` (schedule IS
  applied), no `ZeroDivisionError`, healthy. Sweep (conc 1/2/4/8, reps 2, mt=400):
  **Arm A 80.1 / 140.4 / 227.4 / 361.0 tok/s, accept 36.9 / 35.6 / 49.2 / 47.5 %**;
  **Arm B (schedule off) 79.9 / 140.1 / 227.2 / 358.5, accept 36.9 / 35.6 / 49.2 / 47.4 %** —
  within noise. So the schedule is genuinely consulted by the scheduler but has no effect on the live
  V2 path, confirming the code trace above. Verdict: the plumbing is a dud; drop it (keep
  `patch_dynamic_sd_cudagraph.py` + `patch_sd_sched_trace.py` + `aijuus/tools/test_dynamic_sd.py` as
  reference/regression only).

---

## 2026-09-29 (cont. 24) — Review fixes: dynamic-SD plumbing dropped; V2 conf capture gated; registry restored

Code review of the uncommitted set flagged four issues; all addressed:

- **Registry restored to the decided config** (`aijuus/model-registry.json`): MTP `server_env` back to
  `RADIANCE_DRAFT_EXACTSET=1` + `RADIANCE_DRAFT_FUSED=1` (arm-C's `0`/`0` was a transient A/B setting
  and contradicted cont.22); `RADIANCE_DRAFT_VOCAB=keep-union-freq.txt` unchanged.
- **Dynamic-SD plumbing dropped** (`aijuus/kv-offload/ops/entrypoint.sh`): removed the `SPEC_SCHEDULE`
  print and the `SPEC_SCHED_ARG` / `num_speculative_tokens_per_batch_size` block (mtp `SPEC_CFG` is
  back to the static form), and removed the unconditional `patch_dynamic_sd_cudagraph.py` invocation.
  Also removed the `spec_schedule` key from the MTP registry entry. Rationale: inert on this stack
  (cont.23) and a maintenance/crash surface for no gain. `patch_dynamic_sd_cudagraph.py` is kept
  untracked as the reference fix, with a STATUS header saying it is NOT applied and MUST be applied if
  the schedule is ever re-armed.
- **V2 confidence capture gated** (`radiance_draft.py`): new opt-in `RADIANCE_DRAFT_V2_CONF` (default
  0). `_install_v2_hooks` now only wraps `_greedy_sample_draft` (and resets `_radiance_conf_hist`) when
  the knob is on; `propose_v2` keeps the liveness log. Documented that the served decode loop is a
  replayed FULL CUDA graph (`Capturing decode CUDA graphs (FULL)`), so the Python sampling body only
  runs on eager PIECEWISE draft steps and cannot yield a per-step confidence — the old "stage-1
  capture" was a silent no-op on the served path.
- **Fused confidence contract documented** (`radiance_drafthead.py`): corrected the comment that
  claimed the fused conf equals the unfused-EXACTSET capture. The fused conf is a COARSE kept-vocab
  softmax with an exact-reranked numerator; unfused-EXACTSET (`_apply_head_int2` `y.fill_(-inf)` +
  32-candidate scatter) is a RERANK-only softmax. `RADIANCE_DRAFT_TAU` must be tuned per arm.
- Verified: `py_compile` clean on both edited modules; entrypoint `bash -n` clean; no
  `spec_schedule`/`SPEC_SCHEDULE`/`patch_dynamic_sd_cudagraph` references remain in the entrypoint or
  registry. Live vllm-0 still runs the previous boot's config until its next restart.

---

## 2026-09-29 (cont. 23) — Dynamic-SD A/B: no measurable gain on the MTP path

- New harness `aijuus/tools/mtp-conc-bench.py`: concurrency sweep (C in 1/2/4/8) with aggregate decode
  tok/s + acceptance from `/metrics` deltas. (`mtp-bench.py` is single-stream and at bs=1 the schedule
  resolves to the static depth, so it can't see a dynamic-SD effect.) Metric names on this build are
  `vllm:spec_decode_num_draft_tokens_total` / `_accepted_tokens_total` (not `num_drafted_tokens`).
- Method: same boot, warm, `--conc 1,2,4,8 --reps 2 --max-tokens 400` (greedy) per arm.
- **Arm A** (dynSD=ON, FUSED/EXACTSET=ON): 80.0 / 140.2 / 227.4 / 359.8 tok/s, accept 36.9 / 35.6 / 49.2 / 47.6 %.
- **Arm B** (dynSD=OFF/static nspec=8, FUSED/EXACTSET=ON): 75.0 / 140.0 / 225.0 / 358.6 tok/s, accept 36.9 / 35.6 / 48.9 / 47.6 %.
- **Result: within noise (~1–6% at conc1, <1% above; acceptance identical to 0.1%)** — native dynamic SD
  gives no measurable benefit here. Mechanism: `num_speculative_tokens_per_batch_size` is consumed only
  by `v1/core/sched/scheduler.py` (→ `num_spec_tokens_to_schedule`, i.e. how many drafted tokens are
  *scheduled/verified*) and by `cudagraph_utils.py` (query lengths). The V2 speculator's draft loop is
  fixed at `self.num_speculative_steps` (= config nspec), so it still runs 8 draft forwards regardless;
  trimming verification width saves little vs the draft-forward cost. At conc8 the per-step draft count
  is ~5 but the speculator loop is unchanged.
- Consequence: the `patch_dynamic_sd_cudagraph.py` overlay is now **dormant** (only hit when a schedule
  is set). Keeping the registry `spec_schedule=""` (static nspec=8) is the simpler, equal-performing
  config. The overlay stays in the entrypoint as a correctness fix in case the schedule is re-armed.

---

## 2026-09-29 (cont. 22) — V2 controller hooks LIVE; native dynamic-SD fixed via overlay; schedule ARMED

- **V2 port stage 1 works**: boot log shows `[radiance.draft] V2 speculator hooks installed` and, on
  the warmup, `V2 greedy_sample_draft ENTERED ids=(8,) conf min=0.0844 max=0.0844 preconf=True` +
  `V2 propose ENTERED cls=MTPSpeculator conf_hist=set`. So the controller is finally on the executed
  path (the draft head's FUSED+EXACTSET `preconf` is being consumed). Note conf is constant across the
  batch here (min==max) — all rows share the same prompt; fine.
- **Native dynamic SD was broken in this build** (now FIXED — see below): adding
  `num_speculative_tokens_per_batch_size=[[1,1,8],[2,3,7],[4,7,6],[8,8,5],[9,64,4]]` to
  `--speculative-config` crash-looped the EngineCore at startup:
  `v1/worker/gpu/spec_decode/autoregressive/speculator.py:143 init_cudagraph_manager` →
  `cudagraph_utils.py:254 _init_candidates` → `round_up(num_tokens, decode_query_len)` →
  `ZeroDivisionError`. The schedule itself parsed correctly
  (`build_dynamic_sd_schedule_lookup(...)= [0,8,7,7,6,6,6,6,5]`), so this was a 0.29 bug in the
  dynamic-SD × V2-speculator-cudagraph path, not our format.
- **Fix — new overlay `patch_dynamic_sd_cudagraph.py`**: `_init_candidates` expands candidate decode
  query lengths as `{num_spec + (decode_query_len - num_speculative_tokens) for num_spec in
  dense_schedule[1:]}`. That assumes the MAIN runner (`decode_query_len = num_spec+1`, offset +1).
  `AutoRegressiveSpeculator.init_cudagraph_manager` builds its per-step DECODE manager with
  `decode_query_len=1`, so the offset is `1 - num_spec` (e.g. -7 at SPEC=8) and the set becomes
  `{-2,-1,0,1}` → `round_up(...,0)`. The patch filters `_q >= 1` and falls back to
  `[self.decode_query_len]` when empty. Main runner unchanged (`{9,8,7,6}`); speculator decode manager
  collapses to `{1}` (i.e. the correct non-dynamic behaviour). Wired into `entrypoint.sh` right after
  `patch_step_trace.py` (unconditional, idempotent; marker `patch_dynamic_sd_cudagraph`). Verified in a
  throwaway container: applies once, `py_compile` OK, second run prints "already applied".
- **Re-armed `spec_schedule`** to `[[1,1,8],[2,3,7],[4,7,6],[8,8,5],[9,64,4]]` in
  `model-registry.json` (MTP entry) after the user restarted vllm-0.
- **Verified on vllm-0 boot (13:49) + smoke request**: `[dynamic-sd-cg] applied`, serve args carry
  `num_speculative_tokens_per_batch_size`, `Using V2 Model Runner`, no `ZeroDivisionError`, service
  healthy. Smoke (batch=1) shows the whole stack executing: `V2 greedy_sample_draft ENTERED … preconf=True`
  and `V2 propose ENTERED cls=MTPSpeculator`, draft head `FUSED (EXACTSET)`; `spec_decode_num_draft_tokens`
  = 64 (8 drafts × 8), accepted 33 at batch=1 — and the schedule selected `num_spec=8`, matching
  `[1,1,8]`, confirming the native schedule drives draft depth.
- Registry state: `spec_schedule=[[1,1,8],[2,3,7],[4,7,6],[8,8,5],[9,64,4]]`, MTP keeps
  `RADIANCE_DRAFT_VOCAB=keep-union-freq.txt` + `RADIANCE_DRAFT_EXACTSET=1` + `RADIANCE_DRAFT_FUSED=1`.
- **Next**: V2 port stage 2 (per-request confidence gate + n-gram tail assembly in
  `MTPSpeculator`/`AutoRegressiveSpeculator`), then A/B (FUSED/EXACTSET on/off; union-freq vs union-v2
  corpus). Custom `RADIANCE_DRAFT_SCHEDULE` is now redundant with native `spec_schedule` — keep as
  fallback only.
- **Stage-2 reconnaissance (feasibility)**: the V2 multi-step draft loop
  (`autoregressive/speculator.py::_multi_step_decode` / `_generate_fused_drafts`) is captured into the
  decode CUDA graph, so a per-request *Python* early-stop (the legacy gate's compute saving) is not
  graph-safe. What IS feasible graph-safely: post-process the returned `draft_tokens[:num_reqs]` (a GPU
  tensor) after `propose`. The legacy matcher's context source (`input_batch.token_ids_cpu_tensor` /
  `num_tokens_no_spec`) does NOT exist on V2 `InputBatch` (it only carries the scheduled `input_ids`),
  but the full history is on GPU at `model_runner.req_states.all_token_ids.gpu` (+ `num_computed_tokens`,
  `input_batch.idx_mapping` maps req→state index), so a V2 n-gram matcher is possible without host
  copies. Net: batch-size schedule is already covered by native `spec_schedule`; the remaining stage-2
  value is (a) n-gram "free win" tail extension and (b) a tau gate that decides where the tail is
  allowed — both as post-propose tensor ops. This is a sizeable port; measurement-first is an option.

---

## 2026-09-29 (cont. 21) — FINDING: the dynamic-draft controller is INERT on the V2 model runner

- **Symptom**: enabling `RADIANCE_DRAFT_EXACTSET=1 + RADIANCE_DRAFT_FUSED=1` (union-freq) gave a
  ~3x-slow first boot (transient) then ~neutral numbers (greedy 90.0 / sampled 92.7 / accept 53.2%
  vs baseline 94.0/89.5/57.8). A one-time `_local_draft` log was added and never fired.
- **Diagnosis** (one-time entry logs on `SpecDecodeBaseProposer.propose/_greedy_sample` and
  `GPUModelRunner.propose_draft_token_ids`): none ever fire, though `install()` reports
  `RADIANCE_DYNAMIC_DRAFT=ON` and the draft head's FUSED path DOES run. So the controller hooks are
  applied to classes that are never called.
- **Root cause**: `vllm_config.use_v2_model_runner` defaults **True** on ROCm except for
  `{DeepseekV32ForCausalLM, DeepseekV4ForCausalLM}` (`ROCM_DEFAULT_MRV1_ARCHITECTURES`); our
  `Qwen3_5ForConditionalGeneration` is not in it and no unsupported features apply, so 0.29 runs the
  **new** `vllm/v1/worker/gpu/model_runner.py` runner. Its drafter for `method="mtp"` is
  `MTPSpeculator` (`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`), not
  `vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer`.
- **Impact**: `radiance_draft.py`'s whole controller — the tau confidence gate, the n-gram/suffix
  gating, the per-slot schedule, the batch cap — is a **no-op** in this deployment. Only the *draft
  head* path is live (it runs via the speculator's `compute_logits`), so int2/vocab-prune/FUSED still
  apply; the decoupling in cont.19 currently has no consumer.
- **Consequence for B1**: FUSED's launch saving still applies to the head, but EXACTSET's mask no
  longer affects the gate (there is no gate), so the arm is ≈neutral; the decoupling must be
  re-validated once the controller is live.
- **Fix options**: (a) port the controller hooks to the V2 speculator (`BaseSpeculator.propose` /
  `AutoRegressiveSpeculator`/`MTPSpeculator` greedy path) — the correct fix; (b) force
  `VLLM_USE_V2_MODEL_RUNNER=0` to run the legacy runner our hooks target (big config change; other
  v2-oriented patches may break); or (c) accept the controller is off and drop its knobs. Recommend
  (a). The plan's Phase 0.5 "V1 hooks inert on V2" concern is confirmed — and it hits our own
  controller, not just the generic audit.

---

## 2026-09-29 (cont. 20) — Phase 0.5 V2-hook audit: STATIC PASS (runtime check armed)

- **Static finding: there is no V1 runner in the 0.29 image.** `vllm/worker/model_runner.py` is
  absent; the runner is `vllm/v1/worker/gpu_model_runner.py` (`GPUModelRunner`) + `vllm/v1/worker/
  gpu_worker.py` (`Worker`). So the review's "V1 hooks are inert" class cannot apply to a hook that
  targets these, and a V1-targeting hook would fail to import (visible) rather than silently no-op.
- **Every hook/patched path targets V2** (verified by reading each install and the patch scripts):
  - `radiance_draft.py` → `vllm.v1.worker.gpu_model_runner.GPUModelRunner.propose_draft_token_ids`
    (and `_sample`/bookkeeping) — the V2 runner itself.
  - `radiance_drafthead.py` → `load_weights` on `Qwen3_5MTP`/`Qwen3NextMTP`/`DFlash2Qwen3ForCausalLM`
    (arch classes) + `LogitsProcessor._apply_head`.
  - `radiance_kernels.install_load_hook` → `Fp8LinearMethod.process_weights_after_loading` (quant layer).
  - `radiance_kernels.install_attn_config_hook` → AITER `unified_attention` `select_3d/2d_config`
    (the ROCm unified-attention backend V2 uses).
  - `radiance_kernels.install_r4d_report` → `vllm.v1.worker.gpu_worker.Worker.compile_or_warm_up_model`.
  - `radiance_allreduce.install_custom_ar` → `CudaCommunicator.{__init__,all_reduce}`.
  - `radiance_vit_attn.install` → `vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa`.
  - patch scripts: targets are `vllm/v1/...` throughout (scheduler, input_processor, rejection_sampler,
    gpu_input_batch, kv_cache_utils, attention backends, gpu_worker). `patch_step_trace.py` targets
    `vllm/v1/worker/gpu_worker.py` + `vllm/v1/worker/gpu/async_utils.py` (V2).
- **Runtime proof** (install ≠ invocation) is a boot-log check per hook; script:
  `aijuus/tools/v2_hook_audit.sh [container]`. Markers: preshuffle line, attn-override line,
  `int2 draft head armed` + `INT2_DRAFT_HEAD (lazy)`, `DRAFT_VOCAB: N of M rows`,
  `RADIANCE_DYNAMIC_DRAFT=ON`, `fast-reduce hook armed`/`custom all-reduce INSTALLED`,
  `R4D kernel selection: libr4d` (fires from the V2 Worker after warmup), and item-2 `[aot-envkey]`
  + env-key hash + `new AOT dir` count. Negative control: with a hook's env toggle off its marker
  must be absent.

---

## 2026-09-29 (cont. 19) — FUSED/EXACTSET tau-gate decoupling (B1), validated

- **Problem**: `RADIANCE_DRAFT_EXACTSET=1` was unusable on MTP because it sets `_radiance_topk_only`
  and makes the returned row `-inf` outside the RERANK candidates; `_local_draft` derived the
  tau-gate confidence as `1/Σexp` over that masked row, so confidence read ≈1 and the gate never
  fired. `RADIANCE_DRAFT_FUSED=1` needs `_radiance_topk_only` (it discards the coarse row), so it was
  inert too.
- **Fix (decoupling)**: `_draft_head_int2_cand` now also emits the per-block sum-exp `SM` (verbatim
  from the already-validated `_draft_head_int2_top1`), and `_apply_vocab_fused` recovers the top-1
  softmax from the COARSE kept-vocab partials (`brow`/`S`/`max_ex`, same as the TOP1 path), stashing
  it as `lp._radiance_last_conf`. `_local_draft` consumes it and skips `capture_local` when
  `lp._radiance_conf_precomputed` (set only when `FUSED and EXACTSET`). The returned row is unchanged;
  no caller signature changed. Default-off (needs both envs).
- **Validation** (`stilldeadcode`/`juupp` `vllm-radiance:0.9.3-collect-tokens`, throwaway GPU run):
  - row equality: fused masked row == unfused masked row (mask + values) — `fused_test.py`.
  - **real lm_head** (`lm_head.weight` bf16 [248320,5120], 32768-id subset, m=8): fused conf vs the
    capture-path reference `1/Σexp` → **mean |Δ| 5.0e-7, max 6.0e-7**; finite/row 32 (=RERANK);
    argmax identical. So the confidence the tau gate sees is unchanged while the head runs in 6
    launches instead of 15.
  - block-partial reconstruction checked on CPU (S == full logsumexp to 5e-7).
- **To A/B**: set `RADIANCE_DRAFT_EXACTSET=1` + `RADIANCE_DRAFT_FUSED=1` in the MTP registry entry
  (`server_env`), restart vllm-0, compare greedy/sampled t-s + acceptance against union-freq without
  them. Recall note: for the MTP ARGMAX caller, EXACTSET caps the draft to the top-RERANK (32)
  coarse blocks, so drafted ids are not guaranteed identical to the coarse-row argmax — acceptance
  must be checked, not assumed.

---

## 2026-09-29 (cont. 18) — patch_degen idempotency fix (crash loop) + vocab keep-set decided: union-freq

- **Bug**: `patch_degen.py` step 7 (CLI options) used marker `'"degen-max-period"'`, but the inserted
  text is `"--degen-max-period"` (leading `--`). The marker never matched, so the block was re-inserted
  on **every container boot**. vllm-1 (restarting repeatedly while chasing a ThinkingCap swap) reached
  ~20 duplicate `--degen-max-period` registrations → `argparse.ArgumentError: conflicting option
  string` → crash loop. It also hung `/reload`: the controller's swap loop holds `_op_lock` until
  `SWAP_TIMEOUT` (3600s), so the reload request blocked.
- **Fix**: marker → `'"--degen-max-period"'`. Verified idempotent — applying 3× leaves the file hash
  unchanged with the count at 1; real restarts of vllm-0 (via `/reload`) and vllm-1 both boot healthy
  with the count still 1.
- **Recovery**: copied vllm-0's good `arg_utils.py` into vllm-1 (it was the only diverged file of
  2382), cleared the stuck `trigger.json`, reset state to the default blend-MTP key, restarted the
  controller then vllm-1. Both instances healthy; ThinkingCap can be retried later.
- **Vocab A/B result / decision**: warm sweep (discard the first post-boot bench, record 2× reps-4) →
  **union-freq 65,327 is the fastest greedy arm** (93.4/93.3), ahead of 96k (87.4), full 248k (85.6),
  49k seed (85.4), 24k (80.9). Chosen keep-set: `keep-union-freq.txt`. Detail in the plan's Vocab A/B
  section.

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

---

## 2026-09-30 (cont. 30) — tcclaviger dev MTP ragged-verify port: A/B on vllm-0 (negative)

Ported the two "portable" items from `tcclaviger/vllm:dev` (29.06.18) and A/B-tested on vllm-0
(`mtp-27B-MXFP4-blend`, conc 8 / single-stream, `aijuus/bench-conc.py` + `aijuus/bench-quick.py`).

### A/B mechanism
Container env is fixed at `docker create`; `docker kill && docker start` only re-runs the
entrypoint. Added `aijuus/kv-offload/ops/entrypoint.sh` hook: sources `/patches/aijuus/ab.env`
on every start (after the registry server_env), so a RADIANCE_* A/B arm is a file edit + restart,
no recreate. Remove the file to restore the baseline.

### Experiment 1 — fork's existing ragged verify (`patch_dynwidth.py`, was off in compose)
Enabled `RADIANCE_DYNAMIC_WIDTH=1` (per-request EMA(accepted)+margin cap on verify width).
conc 8 aggregate **361.9 -> 368.8 tok/s (+1.9%, within run spread)**. No effect single-stream
(gated off below 3 running). Confirms cont.26: verify width is not the lever.

### Experiment 2 — ported tcclaviger confidence early-exit (`patch_mtp_conf_exit.py`)
New overlay: captures the top-1 draft probability in-graph, keeps a per-request survival product,
stops each request's serial MTP draft at the first step below tau, and carries per-request
`draft_lens` through `DraftTokensHandler` (placeholder-length path) so the target verifies ragged
widths. Tau 0.5 (raw top-1 is ~0.9, so tcclaviger's calibrated 0.28 never fires here).
- Diagnostic: bs8 all rows drop at step 2 -> width 4 -> **2**; bs2 keeps 5; bs1 prose -> 1.
- conc 8 aggregate **361.9 -> 351.6 tok/s (-2.8%)**, single-stream **103.9 -> 95.1 (-8.5%)**,
  `ms/step` flat/up (weight-bound target). Acceptance rate rose (50% -> 58%) but tokens/step fell
  more than time. At bs1, prose dropped 67 -> 57 tok/s.
- Baseline re-confirmed after revert: **364.0 tok/s** (dormant patch is a no-op).

### Conclusion
The tcclaviger MTP update is **not profitable on this stack**: the batch-size schedule
(`spec_schedule` k=5/4) already sits at the per-concurrency optimum, the target forward is
weight-stream-bound at our concurrency (MAXSEQS=8), and proposer depth has a hard geometry cliff
below k=3 (cont.26). Raw top-1 confidence is uncalibrated; a calibrated estimator would add a lot
of machinery to chase a schedule that already wins. The PLE-fusion and QSA-ring items are
Qwen4Exp/Flash-Next-only and do not apply to this model.

Artifacts left **dormant** (env-gated, default off): `aijuus/kv-offload/patches/patch_mtp_conf_exit.py`,
the entrypoint `ab.env` hook, `aijuus/bench-conc.py`, bench-quick API-key support.
Track B (mamba conv spec-scratch zeroing) remains **unimplemented**; the Triton align-copy tail
hazard is structurally present and is the next correctness item if pursued.

### Restart reminder
`docker kill $(docker ps --format '{{.Names}}' | grep -m1 '^vllm-0') && \
 docker start $(docker ps -a --format '{{.Names}}' | grep -m1 '^vllm-0')` — the **start** must
use `docker ps -a` (a killed container is not in `docker ps`). Poll health every 20 s.

## 2026-09-30 (cont. 31) — tcclaviger mamba spec-scratch zeroing port: A/B negative (-55%)

Implemented `aijuus/kv-offload/patches/patch_mamba_scratch_zero.py` (gated
`RADIANCE_MAMBA_ZERO_SCRATCH=1`), porting the dev's `v1/worker/mamba_utils.py` change: zero the
align state-copy destination tail (columns `[num_dst_tokens, conv_width)`) in both conv layouts so
a verify at accepted offset > 0 cannot convolve the destination page's previous owner.

- Layout check: `get_conv_state_layout()` = **SD** (no `VLLM_SSM_CONV_STATE_LAYOUT` env), so only
  the SD branch executes; the DS branch is dead code here.
- Boot clean, `[mamba-zero] applied`, acceptance/output unchanged (acc/draft 1.98).
- **conc 8 aggregate 361.9 -> 161.8 tok/s (-55%)**. A single masked u8 tail store per copy cannot
  explain that volume (tail ~= token_bias*inner_size*elem = KBs); it is a Triton kernel
  occupancy/recompile side effect of the added dynamic-range loop, not store bandwidth.
- Reverted (reset `mamba_utils.py` from the image, removed `ab.env`); baseline re-confirmed
  **363.9 tok/s**.

Conclusion: the align state-copy is on the hot path, so this correctness fix must be a cheap
vectorized store (or the conv-write side `ZERO_SPARE`) and only after reachability of the stale
scratch is proven on our R4D GDN path. Not landed; `patch_mamba_scratch_zero.py` left dormant.

Both tcclaviger dev items selected as "portable" (conf-exit/ragged-verify and conv scratch
zeroing) are now A/B-closed as not profitable on this stack.

## 2026-09-30 (cont. 32) — Corrected A/B method: compile cache is performance-critical; tcclaviger retests

Advice taken: clear the compile cache on restart. Verified the caches and the effect.

### Cache layout (vllm-0)
- Host `<cache> = ~/.radiance-cache-w4a8-093-gdnm-nqft-fp8s-gnq-tp1s` -> container `/cache`
  (`vllm`, `inductor`, `triton`, `aiter`, `tunableop`). vllm-1 uses a separate `...-b` dir, so
  clearing vllm-0's cache does not disturb vllm-1. Container `/root/.cache/comgr` (ROCm compiler
  cache) also persists across `docker restart`; `/root/.cache/huggingface` holds weights (never clear).
- **Clearing the cache collapses throughput on the next boot** — baseline c1 **20.5** (vs 67),
  c8 **157.6** (vs 364) — and a **warm restart restores it** (c1 67.2; artifacts rebuilt on the
  cleared boot are reused). So the correct method is: clear -> boot1 (rebuild) -> `docker restart`
  -> boot2 (warm) -> measure. Measuring boot1 is invalid.

### Retests under clear+boot1+boot2 (vllm-0, mtp-27B-MXFP4-blend, conc-1/8, gen 256)
| Arm | c1 agg | c8 agg | notes |
|---|--:|--:|---|
| baseline (R4D) | 71.0 | 369.1 | c1 per-rep deterministic and identical across arms |
| `patch_mamba_scratch_zero` | 71.0 | 387.5 | **neutral** (c1 byte-identical run) |
| `patch_mtp_conf_exit` tau 0.5 | 66.8 | 348.7 | **negative ~-5.5%**; acc rate up 50%->57%, ttft up |

**Corrections vs cont.30/31:** the mamba arm's earlier **-55% was a stale-cache artifact** — the
align-copy tail zeroing is perf-neutral and is a viable correctness hardening. The conf-exit
result stands: negative at matched (fresh) cache. Both patches remain in the overlay; conf-exit
is env-gated off.

### P4 (R4D vs AITER attention) — BLOCKED
`R4D_ATTN=0` -> `ROCM_AITER_UNIFIED_ATTN` boots fail: `ValueError: Selected backend
...ROCM_AITER_UNIFIED_ATTN is not valid for this configuration. Reason: ['KV connector not
supported']` (our OffloadingConnector). A/B needs `KV_OFFLOAD_GIB=0`; entrypoint now honours
`R4D_ATTN` (default 1) for that future arm.

### Cache-clear procedure
```
docker kill <vllm-0>; docker run --rm --user root -v <cache>:/cache --entrypoint bash \
  stilldeadcode/vllm-radiance:0.9.3 -c 'rm -rf /cache/*'
docker start <vllm-0>   # boot1, rebuild
# wait health, then
docker kill <vllm-0>; docker start <vllm-0>   # boot2 warm, then measure
```

## 2026-09-30 (cont. 33) — X2 closed N/A, K3 neutral

### X2 reorder-threshold fix — N/A for our runner
`calculate_reorder_batch_threshold` exists only in the **V1** runner
(`v1/worker/gpu_model_runner.py:7325`); the served **V2** runner
(`v1/worker/gpu/model_runner.py`) has no reorder-threshold path at all (no `reorder_batch_threshold`
reference; builders set it but nothing aggregates it). The upstream #55894/#55898 bug and its fix
therefore do not apply to the MTP V2 path. Closed; no patch written. Keep the backend invariant as a
monitoring note only.

### K3 capture-ladder trim — neutral
Added `RADIANCE_CAPTURE_SIZES` (comma-separated override of the derived union) to the entrypoint and
tested a coarse ladder `1,4,8,16,24,32,40,48,56,64,72` (11 buckets vs ~40), clear+boot1+boot2:
- conc-8 **370.9** vs baseline 369.1 (neutral).
- `GPU KV cache size` unchanged at 171,320 tokens → capture memory was not the binding constraint.
No runtime benefit; knob left in (default off). K3 closed.

## 2026-09-30 (cont. 34) — M1 n-gram fault: standalone repro PASSES; fault is in-engine integration

Stopped vllm-0 to free card 0 and ran the new `aijuus/match_gpu_repro.py` off-server.

| Variant (card 0, isolated) | Result |
|---|---|
| B=1, ctx 512, nspec 8, iters 60 | PASS |
| B=2, ctx 512, nspec 8, iters 60 | PASS |
| B=2, cross=1 | PASS |
| B=2, maxl=64, ctx 4096 | PASS |
| B=2, invalid ids (-1/-2) / out-of-vocab | PASS |
| B=2, invalid + no per-iter sync | PASS |
| B=1, invalid | PASS |
| nspec=5 (any B) | **Triton compile error** `arange's range must be a power of 2` |

Conclusions:
- The matcher kernel is **not** faulty in isolation at B=2 — the in-engine `HSA_STATUS_ERROR_EXCEPTION`
  is an **integration/aliasing** effect (engine buffers/views, memory pool, or launch context), not the
  match_gpu arithmetic. Repro must move in-engine (NGRAM=1 boot with targeted instrumentation of the
  `ctx` view, buffer pool, and launch context), accepting a possible wedge per attempt.
- Latent constraint found: the matcher's `tl.arange(0, NSPEC)` requires **NSPEC a power of two**.
  The engine passes `cap = num_speculative_steps = 8` (po2) so it is safe today, but any future
  non-po2 n-gram tail width would hard-fail at compile.
- vllm-0 restarted to baseline (healthy).

## 2026-09-30 (cont. 35) — index refresh, M1 narrowed, static items closed

- **M1 narrowed** (cont.34): standalone matcher passes B=1/B=2 and variants; fault is in-engine
  integration. Recorded. Latent rule: matcher `NSPEC` must be power-of-two.
- **H2 closed**: keep the n-gram code inert (`NGRAM=0`); retained safety changes stay.
- **H4 partial**: root README/DOCKERHUB are upstream-owned → deployment-reality corrections added to
  `aijuus/README.md` (V2 inertness of TAU/SCHEDULE/NGRAM, HEAD_TOP1 dropped, R4D/AITER, cache method).
- **New harness**: `aijuus/bench-prefill-ttft.py` (long-prompt TTFT/prompt-eval t/s) for PF1/P2/P3;
  smoke-tested on vllm-0 (2048 tok → ~472 ms warm TTFT, ~4.2k prompt tok/s).
- Still-open GPU items queued: PF1 profile, K1 calibrate, V1 per-row fallback, PF5 AR A/B, M1 in-engine.
- Background: tcclaviger HIP kernel package analysis (TF1/TF3/ST7) running.

## 2026-09-30 (cont. 36) — tcclaviger HIP package analysis (TF1/TF3/ST7)

Background analysis of `/tmp/kilo/tccla-dev-root/app` + `site-packages` vs our stack:
- All tcclaviger HIP packages are **py3.14/torch2.11+rocm10** vs our **py3.12/rocm7.14** → ABI-bound,
  not drop-in; source shipped only for libr4d/gdn_hip-ish, not fp8hip/parohip.
- **TF1 `gdn_verify_r`: not an upgrade** — tcclaviger's own libr4d chain beats it 1.2-1.5×; we run libr4d.
  Only the interface idea (single dispatch, raw int32 spec metadata, in-kernel per-token snapshots) ports.
- **TF3 `clav_ar`/`clav_ag`: N/A** — TP collectives; we are DP; x1 card defeats BAR P2P.
- **TF2 `clav_attn`: optional A/B** after rebuilding `clav_attn_C` for py3.12/rocm7.14 (direct d256/fp8-KV/MTP alternative to R4D).
- **KB1 libr4d bump** would inherit generic `r4d_pq_*` + dense `r4d_gemm_w4a8_nt_m64`; must rebase
  `r4d_radiance_extras*.patch` and re-validate GDN (their r4d GDN NaNs on our model).
- `q4hc`/`plehip`/`r4d_qsa|ple|mhc|dsfp|dflash2` are Qwen4Exp/Flash-Next/DSV4-only → N/A.

## 2026-09-30 (cont. 37) — PF1 prefill chunk-size result + cache-method refinement

### PF1: prefill throughput is chunk-independent at 8k tokens
`aijuus/bench-prefill-ttft.py`, 7974-token prompt, gen 8, warm reps:
- chunk **16384**: TTFT 452.0 ms, prompt **17.64k tok/s** (single chunk), decode 93.5 tok/s.
- chunk **4096**: TTFT 453.9 ms, prompt **17.57k tok/s** (two chunks), decode 37.2 tok/s.
So splitting an 8k prefill into 2×4k chunks costs nothing measurable; the 16k-chunk **decode**
difference is a compile-key artifact, not a chunk effect. Prefill work (PF2 GDN scan, PF3 R4D
scheduling) must be measured at larger prompts/context — the 8k case is already at parity.

### Cache-method refinement (important)
Clearing `~/.radiance-cache-…-tp1s` then boot1+boot2 did **not** always restore throughput: after the
PF1 clears the server sat at ~21/161 t/s even after a "warm" restart, then recovered to **67.8 / 369.1**
on the *next* restart (seeded the Triton cache from the intact vllm-1 dir `…-tp1s-b`, arch-keyed, plus
another reboot). Lesson: after a cache clear, **verify c1 ≈ 68 t/s before trusting any measurement**,
and treat a cleared cache as multi-boot until warm. vllm-1's cache is separate and was never cleared.

Baseline restored: conc1 67.8, conc8 369.1 (R4D, chunk 16384, no ab.env).

## 2026-09-30 (cont. 38) — V1 static-verified; backlog triage

- **V1 statically verified**: `radiance_draft.py:758-780` per-row windowed fallback is correct
  (miss rows re-scan with `base=0`, non-miss rows get an empty window, `pk[sel]` copies back only
  miss rows). It only executes under `NGRAM=1`+window, so it is gated by M1.
- **Backlog triage** (`aijuus/OPEN-TASKS-INDEX.md`): the quick/config/A-B items are done
  (M2, X2, K2, K3, K4/P4, PF1, T1, T2, V1); what remains is large or blocked —
  kernel projects (PF2/PF3, S4/KB2-KB4), image/ABI rebuilds (KB1, X1, TF2), offload code (OS1-3),
  the private fork (TF5), the in-engine M1 repro (risky), GPU-gated validations (K1, V3, ST1/ST2/ST6),
  open research questions (§15), and the H1 commit (awaiting explicit go).

## 2026-09-30 (cont. 39) — ST6 acceptance-by-position (no restart)

Per-position acceptance from `vllm:spec_decode_num_accepted_tokens_per_pos_total` (added to
`bench-conc.py`). Conditional P(accept pos p | accept p-1):

| arm | p0 | p1 | p2 | p3 | p4 | p5-7 |
|---|--:|--:|--:|--:|--:|--:|
| conc1 | 0.78 | 0.67 | 0.72 | 0.53 | 0.50-0.75 (n=1) | 0 (not scheduled; k=5) |
| conc8 | 0.78 | 0.75 | 0.65 | 0.60 | 0.01 | 0 (not scheduled; k=4) |

Finding: acceptance decays slowly to ~0.6 by position 3, so the current schedule (k=5 at bs1-2, k=4
at bs3-8) is well matched; positions 5-7 are never *scheduled* (not merely never accepted), so
"depth beyond 4-5" cannot be judged from these counters without raising K. p4 at conc1 (0.5-0.75,
n=1) suggests depth 5 is justified there. Supports keeping SPEC=8 ceiling with the dynamic schedule.

## 2026-09-30 (cont. 40) — offload E2 anchor found; ST1/VC2 already answered

### E2 (OS2) anchor and why it is not a one-line flip
`v1/worker/gpu/kv_connector.py:86 ActiveKVConnector.post_forward(finished_req_ids,
wait_for_save=True)` calls `self.kv_connector.wait_for_save()` on the critical path, and
`v1/worker/gpu/model_runner.py:1860/1994/2017` call it with the default `wait_for_save=True` every
step (incl. the final prefill step — the measured ~18% cold-prefill stall). A naive
`wait_for_save=False` would let freed/reused blocks race the D2H → KV corruption. The correct fix is
the ADAPTIVE-KV E2 design: a reserved/device-ready/host-ready/committed frontier, advance durability
only over contiguous K+V D2H completions, retire blocks only after the last referencing completion.
Scoped but **not implemented** — medium-high effort on the live offload path, needs GPU correctness
validation (cold prefill + multi-turn + block reuse). Deferred rather than risk the serving path.

### ST1 / VC2 status
- ST1 (T0 per-phase split): already answered by prior instrumentation — MTP is host-CPU-bound with a
  ~28 ms/step `_bookkeeping_sync` (verify-forward wait) and ~16.5 ms propose; re-running step trace
  on 0.29 would restate it. No new run.
- VC2 (EXACTSET+FUSED re-run): prior R9700 evidence stands (EXACTSET renormalizes the V1 tau-gate
  confidence; inert on V2 where the gate is not ported). No new run.

## 2026-09-30 (cont. 41) — rebuild triage + design-item deep research launched

Rebuild items triaged per directive (overlay if possible, else park):
- **PARKED** (no runtime overlay possible / needs a libr4d or image rebuild to validate):
  PF2 (GDN scan), PF3 (R4D prefill scheduling) — kernels live in the libr4d clone, not this repo;
  KB1 (libr4d bump — possible as a build overlay but low value + GDN-NaN caveat); KB2-KB4 (cp314-only);
  X1 (flash_attn, low value); TF2 (clav_attn — source not shipped, genuinely impossible).
- Remaining actionable work is the design+validation set; launched deep research before implementing:
  1. E2 (offload store decoupling / committed frontier) — the ~18% prefill store await.
  2. E1 (suffix-only invalidation on draft rejection; MTP-draft-group offload inclusion).
  3. E3 (staged host-snapshot GDN rollback) + M1 (in-engine n-gram fault cause/instrumentation).

## 2026-09-30 (cont. 42) — E2 implemented + A/B NEGATIVE; E1/E3/M1 research in

### E2 (offload lazy-commit) — implemented, measured, reverted
`aijuus/kv-offload/patches/patch_offload_lazy_commit.py` (wired into `apply-kv-patches.sh`, gated
`RADIANCE_OFFLOAD_LAZY_COMMIT`, default off): Hunk A submits the store D2H at creation in
`post_forward` instead of deferring a step; Hunk B stops a finishing request's self-flush
(`offloading/scheduler.py` `if req.is_finished()`), leaving the existing `_block_id_to_pending_jobs`
fence to fire on first real reallocation. Applies cleanly + idempotent on our built image.

A/B, cold 18k prefill (`bench-prefill-ttft.py`, unique salt per rep), warm c1≈68:
| arm | TTFT | prompt t/s |
|---|--:|--:|
| gate off (baseline) | 7529 / 7602 ms | 2328 / 2306 |
| gate on | **8031 ms** | 2181 |

**+5.6% slower** — the eager submit puts the (large) first-chunk KV D2H in flight concurrently with
the second prefill chunk's compute, so it contends for bandwidth rather than overlapping sampling.
The earlier "18% finish-time store await" was mis-attributed: the real await is
`pre_forward -> handle_preemptions -> worker.wait(jobs_to_flush)` and for a 2-chunk prefill the
blocking await was already small. Reverted (ab.env removed, baseline restored 7602 ms). Patch left
dormant.

### Deep research completed (implement next)
- **E1** (suffix-only invalidation): medium; rejection currently invalidates **nothing** (counter
  rollback only); volatility is handled by withholding the trailing chunk, which caps the *target*
  group's external hit by one chunk every turn. Needs a new `manager.invalidate` API + tombstones
  (races with in-flight load/write) + tiering/fs cascade + a scheduler hook. Detailed change set in
  the research (WORKLOG refs); not yet implemented.
- **E3** (staged host-snapshot GDN rollback): medium runtime patch; root cause of the lazy-GDN
  multi-turn corruption identified — the libr4d materialize kernel **fails open** (`r=0` branch
  stores the base state as the checkpoint) whenever a prefix hit invalidates the stash. Design:
  2-slot GPU stage + pinned host ring keyed by token frontier, fail-**closed** gate. Needs a small
  libr4d edit or a runtime Triton validator. Not yet implemented.
- **M1** (in-engine n-gram fault): ranked hypotheses H1 (untested windowed path `base>0`) / H2
  (`_match_gather` OOB row read at ML=160k) / H3 (matcher scratch allocated during a FULL-graph
  replay) / H4 (UVA/int32 context vs repro VRAM/int64) / H5 (bad continuation corrupts the next
  step). Instrumentation plan (dump inputs, per-kernel sync split, pre-warm buffers, id range check)
  ready.

## 2026-09-30 (cont. 43) — E1/E3 implementation decisions (parked), backlog status

- **E3 (lazy GDN rollback) — PARKED (rebuild-class).** Root cause is a libr4d materialize kernel that
  fails open (`r=0` stores the base as the checkpoint) on stash invalidation. A correct fix is either
  a libr4d edit (rebuild) or a new runtime Triton validator kernel. Per the overlay/rebuild rule this
  is parked. Design captured in cont.42.
- **E1 (suffix-only invalidation) — PARKED (needs extended validation).** Verified all anchors
  (`scheduler.py:2048-2066` rejection rollback, `offloading_connector.py:144-150` delegate,
  `base.py:262 touch`, `cpu/manager.py:96/124/181/244`, `tiering/manager.py:185-332`). Safe Stage-1
  plumbing is inert (the volatile tail is not stored under the current drop, so invalidation is a
  no-op); the benefit only exists with the Stage-2 include path, which changes the KV-consistency
  contract and needs the multi-turn byte-identical oracle (turnbench --exact) before it can ship.
  Not safely completable in-session; parked with the full change set recorded.
- **M1 — PARKED (risky).** In-engine instrumentation requires an `NGRAM=1` boot that can wedge the
  card per probe; hypotheses/plan recorded (cont.42).

### Backlog status
Actionable non-destructive items are complete: the tcclaviger ports A/B-closed (conf-exit/ragged
negative, mamba neutral), E2 implemented+A/B-negative+dormant, PF1/ST6/V1 measured, M2/X2/K2/K3/P4/T1/T2
closed, rebuild items parked. Remaining are parked (E1/E3/M1 + all rebuild/kernel items) or need
user decisions. Baseline preserved: vllm-0 healthy (c1≈68, 18k TTFT≈7600 ms), no ab.env.

## 2026-09-30 (cont. 44) — M1 hypotheses H1/H4 ruled out off-server

Extended `aijuus/match_gpu_repro.py` with the engine's untested inputs (window `base>0`, UVA/pinned
int32 context) and ran off-server on the freed card 0 (vllm-0 stopped, restarted after):

| variant (B=2) | result |
|---|---|
| H1 windowed path: `--window 16384 --ctx-len 40000` | PASS |
| H1+H2 invalid: `--window 16384 --ctx-len 40000 --invalid 3` | PASS |
| H4 UVA/int32 ctx: `--uva 1 --window 16384 --ctx-len 40000` | PASS |

Both H1 (untested `base>0` window) and H4 (UVA/int32 context source) are **ruled out** as the
standalone cause. Remaining hypotheses need in-engine instrumentation (can wedge): **H3** matcher
scratch allocated inside a FULL-graph replay, **H2** `_match_gather` OOB row read at ML=160k,
**H5** bad continuation corrupting the next step. Next step is the cont.42 in-engine plan
(pre-create/pre-compile buffers before capture first, as it is the most likely and is a safe change).

## 2026-09-30 (cont. 45) — E1 Stage 1 implemented + validated inert

`aijuus/kv-offload/patches/patch_offload_suffix_inv.py` (wired into `apply-kv-patches.sh`, gate
`RADIANCE_OFFLOAD_SUFFIX_INV`, default off). Stage 1 = full invalidation plumbing, end to end:
- `v1/core/sched/scheduler.py`: on `num_rejected>0` (post-rollback) call `connector.on_draft_rejected`.
- `offloading_connector.py`: delegate.
- `offloading/scheduler.py`: `on_draft_rejected` — for eagle/MTP groups, `first_stale = b//C-1`,
  `manager.invalidate(stale_keys)`, trim `offload_keys`, clamp `next_stored_chunk_idx`.
- `kv_offload/base.py`: `OffloadingManager.invalidate` default no-op.
- `cpu/manager.py`: conservative removal (ready, `ref_cnt==0` only; no tombstones needed yet).
- `tiering/manager.py`: primary + cascade to secondary tiers.
- `tiering/fs/manager.py`: remove on-disk file + `_lookup_manager.invalidate` (forget).

Fix during bring-up: fs tier lacks `Collection` import → annotation-free signature (boot had
crash-looped, now fixed). Validation: gate=1 vs gate=0, `temperature=0 seed=1`, two prompts →
**byte-identical** outputs, boot healthy, no crash. Stage 1 is confirmed inert (the volatile tail is
not stored under the current drop, so there is nothing to invalidate).

Next: **Stage 2** (`RADIANCE_OFFLOAD_EAGLE_INCLUDE`) — remove the store-side trailing-chunk drop and
the load-side extra-chunk pop so MTP groups are included; needs the multi-turn byte-identical oracle.

## 2026-09-30 (cont. 46) — E1 Stage 2 implemented + correctness-validated

`aijuus/kv-offload/patches/patch_offload_eagle_include.py` (wired into `apply-kv-patches.sh`, gate
`RADIANCE_OFFLOAD_EAGLE_INCLUDE`, default off): removes the two withholding mechanisms so eagle/MTP
groups are included — the store-side trailing-chunk drop (`storable_chunks`) and the load-side extra
query/pop (`query_max`+`required_window`+`num_hit_chunks`). Correctness relies on Stage 1 invalidation.

Validation (both gates on, `temperature=0 seed=1`, two prompts): outputs **byte-identical** to the
gate-off baseline; per-position acceptance unchanged (p0≈0.78 … p3≈0.61, p4≈0.01); boot healthy, no
crash. conc-8 aggregate was cache-confounded (new RADIANCE_* env → fresh AOT key), so the offload
hit-rate benefit was NOT measured here — that needs the multi-turn/prefix-reuse oracle
(`turnbench --exact`) with a warm cache. Both E1 patches are left **dormant** (default off), so the
served configuration is unchanged. Baseline restored (c1 67.8).

- Both E1 stages are runtime patches only; no image/libr4d rebuild.
- E3 remains the one item needing a (small, incremental, non-image) `r4d.so` rebuild via the rx10
  extras patch, or a runtime Triton fail-closed validator.

## 2026-09-30 (cont. 47) — E1 inclusion: workload measurement (turnbench --concurrent)

With `RADIANCE_OFFLOAD_SUFFIX_INV=1` + `RADIANCE_OFFLOAD_EAGLE_INCLUDE=1`, three-agent workload
(`turnbench --concurrent`, MAXSEQS 2, sessions growing to ~110k tokens):
- **COMPLETED 21/21 turns, HEALTH PASS** (no empty replies, no repeat loops).
- Served decomposition: **GPU 5.0%, tier 71.0%, recomputed 24.0%** of 1,402,062 prompt tokens — the
  offload tier carries the bulk of prefix reuse at long context. No crash; inclusion is stable.
- The `--exact` multi-turn oracle (byte/logprob-identical CACHED vs COLD twins, the mandated
  correctness gate for inclusion) is running in the background; result to be folded in next.

## 2026-09-30 (cont. 48) — E1 inclusion: turnbench --exact attribution + benefit

Ran the `turnbench --exact` multi-turn oracle twice (gate-on E1 vs gate-off baseline), three-agent
workload to ~110k tokens/session, CACHED vs COLD twins.

| path (turn) | gate-off baseline | E1 on (SUFFIX_INV+EAGLE_INCLUDE) |
|---|---|---|
| COLD (1) | EXACT | EXACT |
| GPU (2-3) | **DIFF** first-div 15-111 | **DIFF** first-div 15-87 |
| TIER (4+) | **DIFF** | **DIFF** |
| TIER ext tokens/turn (A) | 50,160 @A4 … 94,160 @A7 | 51,040 @A4 … 95,040 @A7 (**+880 = one extra chunk/turn**) |

Conclusions:
- The CACHED-path divergence (GPU partial-hit, and TIER) is **pre-existing in our stack** — baseline
  diverges identically. It is NOT introduced by E1. (turnbench's docstring flags the GPU partial-hit
  case as a known open issue; TIER divergence appears to inherit it via the cached-history feedback.)
- **E1 works as designed**: at every TIER turn it adds exactly **+880 external tokens (one chunk) per
  turn** — the volatile MTP group's trailing chunk is now stored and served from the tier, i.e. the
  one-chunk-per-turn hit-rate cap is lifted. No crash; health PASS.
- Both E1 patches remain **dormant** (default off). Enabling them is safe w.r.t. this oracle, but the
  pre-existing cached-path divergence should be understood independently before trusting inclusion
  for multi-turn correctness at high hit rates.

Open follow-up (new): the pre-existing multi-turn cached-path divergence (GPU partial-hit at fp16,
and TIER) — recorded in `aijuus/OPEN-TASKS-INDEX.md` as a real correctness item, separate from E1.

## 2026-09-30 (cont. 49) — E1 enablement rationale, MT1/MT2 definitions, rebuild-vs-overlay

### Why E1 is currently dormant (decision: do NOT enable yet; do NOT sidestep MT1/MT2)
Output correctness is NOT the blocker. The volatile trailing chunk is the **EAGLE/MTP draft attention
group**; `is_eagle_group` is used only in KV-cache/offload bookkeeping, never by target attention or
sampling, and every draft is target-verified with target KV. So a stale draft-KV reuse can only lower
**draft acceptance**, never change a served token. E1 Stage 1 also deletes the stale keys and clamps
`next_stored_chunk_idx`, so the chunk is recomputed/re-stored on the next opportunity — a stale hit is
**self-healing via the normal miss/recompute path**, i.e. "prefill the bad parts normally" is already
what happens. The cross-group hit is reconciled to the **min across groups**, and a group's hit is
bounded by its own lookup, so including the MTP group cannot make the **target** skip recomputing a
chunk it did not itself hit.

Reasons to wait anyway:
1. The only workload-level oracle (`turnbench --exact`) is **red on the untouched baseline** (MT1/MT2,
   below), so E1's acceptance/benefit parity cannot be measured with a trusted gate. Do not sidestep
   MT1/MT2: the same multi-turn reuse path is E1's operating regime and where the tier carries ~71%.
2. Stage 1 invalidation is conservative (removes only ready, `ref_cnt==0` CPU-primary entries; skips
   in-flight/in-use) — a store/load race can leave a stale draft block for one step. Bounded and
   self-healing, but a known gap; tombstones are the clean fix.
3. Benefit is narrow (+1 tier chunk/turn on multi-turn long-context), not decode throughput; needs a
   real multi-turn A/B to justify.

### MT1 / MT2 (short)
- **MT1 — GPU partial-hit divergence.** Turn 2 hits ~23k of ~37k tokens in the prefix cache; the rest
  is recomputed and the CACHED output then differs from the COLD twin of the identical messages
  (first-div 15-111, max|dlogprob| 0.04-0.19). COLD-only turns are EXACT. The harness names it a known
  open issue (`gpu-partial-hit-divergence.md`, missing from the repo). Likely: the un-cached tail is
  prefilled under a different batching/chunk shape than a full cold prefill, so not bit-identical.
- **MT2 — TIER divergence.** TIER-served turns also differ, though a TIER path "must be EXACT" (a byte
  copy of an already-computed state). Open question: independent tier bug, or inherited from MT1 via
  the diverged cached history (twins get identical text, so an independent cause is plausible).
- Both mean **multi-turn long-context KV reuse is not numerically reproducible on the current stack**,
  independent of E1.

### Rebuild vs overlay (answer)
- **E1 needs no rebuild at all** — pure runtime Python patches (`patch_offload_suffix_inv.py`,
  `patch_offload_eagle_include.py`).
- **E3 needs no *image* rebuild** either: edit `r4d_radiance_extras_rx10.patch` and rebuild **just
  `r4d.so`** through the existing `AUTO_R4D`/`R4D_KEY=…-rx10` path (host-cached under
  `~/.cache/radiance-libr4d/<key>/`, mounted into `/r4d`, copied over the image's at boot), or avoid
  even that with a runtime Triton fail-closed validator.

## 2026-09-30 (cont. 50) — MT1 narrowed (writer faithful); E3 fix route chosen

### MT1/MT2 oracle (`aijuus/kv-offload/ops/hit_oracle.py`, vllm-0 mtp-blend, 24k prompt, temp 0 + logprobs)
| comparison | result |
|---|---|
| writer(cold) vs cold_twin | bit-identical |
| full-hit vs cold_twin | bit-identical |
| partial-hit (prefix + new suffix) vs cold | bit-identical |
| multi-turn: turn2 reusing an **assistant reply written by MTP decode** vs cold | **bit-identical** (160 tokens, max|dlp| 0) |

Conclusion: the KV **writer is cold-faithful** on the base path — full hit, hit-anchored suffix prefill,
and decode-written reply reuse all reproduce a cold recompute exactly. So MT1 is **not** base
prefill/KV or reply-reuse accumulation. The turnbench divergence must come from a condition the single
-session oracle does not exercise: **cross-session shared prefixes / partial external (offload) hits**
(turnbench's three sessions share the stdlib corpus), or scale (pool pressure/eviction). `patch_offload_mixed_hit.py`
/ `patch_reconcile_reask.py` are the prime suspects for a wrong-prefix serve, not numerics. MT2 remains
likely inherited.

Next MT1 step: a cross-session partial-external-hit repro (two sessions sharing a prefix, cold twin
for a later turn), then bisect the offload patches (`RADIANCE_OFFLOAD_MIXED_HIT=0`,
`RADIANCE_RECONCILE_REASK=0`) against `turnbench --exact`.

### E3 fix route (research)
Fail-open confirmed at `r4d_radiance_extras_rx10.patch:1503-1509` (`r=0` on header mismatch) followed by
an unconditional store at `:1529-1531`. `radiance_gdn_lazy.py` is **runtime-copied** from /patches each
boot (`entrypoint.sh:227`), so a Python fix needs only a restart — **no rebuild**.
Routes: (3a) Python header check → detect (raise/log) but does not repair; (3b) skip the store = a trap
(`dst` then holds stale q/k/v bytes); (3c) **host-snapshot ring + 2-slot GPU stage keyed by token
frontier, restore on mismatch** = the only variant that *fixes* it, pure Python, no rebuild; (K2)
kernel `restore_ok` flag + fallback via the incremental `r4d.so` rebuild (higher moving parts: ABI bump,
cache-key bump, must compile with the image's hipcc). Chosen route: **3c** (self-contained overlay),
with 3a as an interim loud-failure if desired. Lazy is currently off (`RADIANCE_GDN_LAZY=0`), so E3 is a
memory optimization (≈865 MB/req device → ≈260 MB), not a correctness emergency.

## 2026-09-30 (cont. 51) — MT1 is single-session; alignment/phase hypothesis

`turnbench --exact --sessions A` (single session, to ~108k, warm pool, gates off) STILL diverges:
COLD A1 EXACT, then A2 GPU partial hit DIFF (first-div 28, max|dlp| 1.05e-01), decreasing over turns
(A4 4.99e-03). So MT1 is **not cross-session** — it is the single-session GPU partial-hit resume, and
my `hit_oracle` (synthetic 24k prompt, reply reuse) was exact because it did not hit the problematic
alignment/shape that turnbench's stdlib prompts do.

Leading hypotheses, now narrowed to the resume geometry (not the KV writer, which the oracle showed
faithful):
1. **Chunk-phase / block alignment.** The resumed suffix prefill starts on a grid anchored at the hit
   boundary; cold starts at 0. GDN scan is chunk-local (internal `CHUNK=64`; block 880; gcd=16) and
   R4D/MXFP4 splits are per-launch, so phase-shifted grids can round differently, amplified to whole
   ulp flips by fp8 KV. `patch_sched_align_last_block` (`RADIANCE_ALIGN_PROMPT_LAST_BLOCK=1`) fixes the
   prompt's last block only, not the resume boundary's state write.
2. A genuine writer/reader mismatch that appears only at specific lengths (e.g. reply block reused
   before re-prefill, or a state block written at a hit boundary different from cold).
Either way it is *resume-specific*, not a general KV corruption.

Decisive next test (research §4 tertiary): a **phase sweep** — pick prompt lengths so the resume/hit
boundary is an exact multiple of `lcm(880, 64, chunk)` vs deliberately misaligned, and see if aligned
is EXACT and misaligned DIFFs. If aligned is exact → chunk-phase (inherent; E1 validation should be on
acceptance, not bit-exactness). If misaligned-exact too and only specific lengths diverge → writer-side
bug to bisect (`patch_sched_align_last_block` / reconcile / mamba stride).

## 2026-09-30 (cont. 52) — MT1 reproduces at all scales; trigger is turnbench-flow-specific

`turnbench --exact --sessions A --t1 6000 --step 4000 --turns 4` (all prompts <16k, one/two chunks,
warm pool, gates off): COLD A1 EXACT, A2 GPU partial hit DIFF (first-div 61, max|dlp| 1.48e-01),
A3/A4 DIFF, drift 61/56/82, HEALTH PASS. So the divergence:
- is **single-session** and **independent of scale/chunk count** (appears at ~10.6k tokens),
- is **not** the KV writer (the `hit_oracle` full-hit/partial/reply-reuse tests were all bit-identical),
- but the minimal oracle does **not** reproduce it, so the trigger is in turnbench's specific flow.

Narrowing: turnbench's distinguishing features vs the oracle are (a) a stdlib-file prompt (varied,
unaligned content) rather than a repeated synthetic one, (b) `max_tokens=320` replies and
`top_logprobs`, (c) reuse of the previous turn's **decode-written assistant reply** as a cached prefix
next turn under the radiance MTP/align patches, (d) the cold twin runs later (cache warm).

The two remaining classes are now: **inherent resume geometry** (a partial-hit suffix is prefilled with
different attention/GEMM split shapes than cold, so not bit-identical; fp8 KV amplifies to token flips)
vs a **radiance writer bug** in the align/reconcile/mamba-stride patches that only this flow exercises.
Decisive next step: byte-level `statecmp` at the hit boundary (dump the reused KV/state bytes for the
cached vs cold A2 and diff), or a `turnbench --exact` bisect with `RADIANCE_ALIGN_PROMPT_LAST_BLOCK=0`,
`RADIANCE_RECONCILE_REASK=0`, `RADIANCE_MAMBA_STORE_STRIDE` variations.

Status: MT1 root-cause narrowed to a specific flow; the fix (or the "inherent" verdict) needs the
statecmp/bisect. MT2 unmeasured but expected to inherit the same mechanism through the tier.

## 2026-09-30 (cont. 53) — MT1/MT2 RESOLVED: inherent resume-vs-cold prefill numerics

Bisect (short single-session `turnbench --exact --t1 6000 --step 4000 --turns 4`), isolating the cause:
| arm | A2 result |
|---|---|
| baseline (align=1, mtp) | DIFF (first-div 61) |
| `RADIANCE_ALIGN_PROMPT_LAST_BLOCK=0` | DIFF (first-div 42) — align patch exonerated |
| `SPEC_METHOD=none` (spec decode OFF) | **DIFF (first-div 53)** — not spec-decode-related |
| writer oracle (`hit_oracle.py`) | full-hit / partial / reply-reuse all **bit-identical** |
| COLD-only turns, always | EXACT |

Conclusion: MT1 (and MT2) is **inherent**, not a bug and not E1. Cause: on a partial GPU prefix-cache
hit the suffix is prefilled as one launch of `total-hit` tokens, while the cold twin prefills `total`
in chunk-sized launches — different attention/GEMM split geometry and accumulation order — and fp8 KV
(3 mantissa bits) turns the sub-ULP differences into whole-ulp flips. Spec-decode, the align-last-block
patch, and the KV writer are all exonerated. The harness's premise "a TIER path must be EXACT" is false
under chunked prefill + fp8 KV: byte-identical KV does not imply bit-identical output.

Implications:
- **turnbench --exact is not a valid gate for E1** (it is red on the untouched baseline by construction).
  E1 correctness must be judged on **acceptance / served-token semantics**, not bit-exactness.
- MT1/MT2 closed as inherent (documented). If bit-exactness across partial hits is ever required, it
  needs a resume that reuses the cold chunk grid (a scheduler change), not a KV fix.

## 2026-09-30 (cont. 54) — Can resume be made bit-identical? (deep research verdict)

**Yes, two routes — but making it default is not worth it.**

Root gate refined: not attention/GEMM shape per se, but the **GDN recurrent path anchored to the launch
start** (S1 `radiance_gdn.py:40,55,201-263` CHUNK=64 launch-local; S2 `conv_prep` per-chunk cumsum
`:376`; S3 the SSM state narrowed to **fp16 at chunk ends** `:699-702,714` / `--mamba-ssm-cache-dtype
float16` `serve-mxfp4.sh:700-702`). fp8 KV (e4m3, 3-bit) is only the **amplifier** (S8). Cold's actual
chunk stride is **C = B·floor((CHUNK−draft_slots)/B) = 880·floor(16384/880) = 15840**, not 16384
(`scheduler.py:550-557`; `mamba_has_prefill_checkpoint_blocks` is False under Eagle/MTP `:448-457`).

- **Route A (runtime overlay, bit-identical):** quantize every prefix hit DOWN to C (raise
  `cache_hit_alignment_tokens`, `single_type_kv_cache_manager.py:77,783`; `kv_cache_coordinator.py:666`)
  and align the Mamba/offload store grid to C (`patch_mamba_stride.py`). **Cost:** forfeits up to
  C−1 ≈ 15.8k tokens of reuse per hit; **`prompt ≤ CHUNK` forces hit = 0** (full recompute). Likely
  destroys most partial-hit benefit and negates E1's tier inclusion.
- **Route B (kernel, heavy):** absolute-phase-anchor the GDN scan (libr4d rebuild) **and** make cold
  round/reload the recurrent state on a fixed absolute grid (runner/kernel change that *alters cold
  numerics*). Preserves hit granularity; needs a `r4d.so` rebuild and touches the hot path.
- **Cheap partial mitigations (not exact):** existing 3520 store grid (`RADIANCE_MAMBA_STORE_STRIDE=4`),
  pin MXFP4 kernel selection across M (S4), bf16 KV (kills the fp8 amplifier, halves concurrency).

**Recommendation (adopted):** do NOT pursue bit-exactness. It is over-strict on this stack by
construction: fp8 KV sets a ~6 %/element noise floor, chunked prefill re-partitions reductions, and
R4D explicitly opts out of batch-invariance (`radiance_r4d_attn.py:260 supports_batch_invariance=False`,
`backend.py:200,321`). Validate E1 on **acceptance / served semantics**, not bit equality. Route A is
available as a gated overlay if bit-exactness ever becomes a hard requirement.

## 2026-09-30 (cont. 55) — MT1/MT2 consolidated note (canonical)

**MT1/MT2 = INHERENT, not a bug, not E1.** A partial GPU prefix-cache hit resumes by prefilling its
suffix as one launch of `(total−hit)` tokens, while a cold twin prefills `total` in chunk-sized
launches. The GDN fused chunk-scan/conv grid is launch-local (`radiance_gdn.py:40,55,201-263`,
`CHUNK=64`) and the SSM state is narrowed to fp16 at cold's chunk ends (`:699-702,714`), so the two
computations group the carry differently; fp8 KV (e4m3, 3-bit) amplifies sub-ulp differences into
whole-ulp flips. Evidence: COLD-only turns EXACT; divergence persists with the align patch off AND with
`SPEC_METHOD=none`; the `hit_oracle` (full-hit/partial/reply-reuse) is bit-identical, so the KV writer
is faithful. **Consequence:** `turnbench --exact` is an unsound gate on this stack (fp8 KV noise floor
~6 %/element; R4D opts out of batch-invariance `radiance_r4d_attn.py:260`). **Solve?** only (A) a
pure-Python overlay quantizing hits to cold's 15840-token stride (forfeits up to ~15.8k tokens/hit;
hit=0 for prompt≤chunk) or (B) a kernel absolute-offset arg (incremental `r4d.so`) + unsupported
prefill-checkpoint-block work under spec decode. Neither pursued. Validate E1 by acceptance/served
semantics.

## 2026-09-30 (cont. 56) — OS1 E1 acceptance validation (done)

`turnbench --concurrent` (3-agent, ~1.4M prompt tokens), gates off vs on:
| arm | GPU | tier | recomputed |
|---|--:|--:|--:|
| gates off (baseline) | 7.7% | 67.3% | 25.0% |
| E1 on (SUFFIX_INV+EAGLE_INCLUDE) | 5.0% | **71.0%** | 24.0% |
Both HEALTH PASS, 21/21 turns. E1 shifts ~3.7 pp of prompt tokens to the tier and cuts recompute ~1 pp
(consistent with +880 tokens/turn). Byte-identical single-turn (cont.45/46); no new divergence vs the
inherent baseline (cont.53). **OS1 done: validated on acceptance/served semantics; gates default off —
enabling is a deploy decision.**

## 2026-09-30 (cont. 57) — OK3 tierbench + closure pass

**OK3 (`tierbench --yes --phases cold,gpu,fs`, 90k prefix):** cold RECOMPUTE 52.2 s, GPU hit 1.6 s,
**FS/tier OFFLOAD 17.7 s**. Tier cuts a cold 90k prefill ~3× and is ~13× slower than a GPU hit, so the
tier is a real but secondary promotion cost; capacity: GPU 227,981 tok, CPU primary 560 blocks
(~476k tok). **OK5 (tier compaction) is not justified by wait** (the cost is promotion latency, not
bytes); only revisit if tier *capacity* becomes the constraint.

**OK4:** the fs promotion path is functional (the 64 s queue-depth stall was the fixed fanout bug) —
no further action.

**Closure/decision pass:**
- **V2** (watch acceptance after vocab prune) — no regression seen across all runs; keep as a standing
  monitor, no action.
- **T3** (n-gram tail-rate) — blocked by M1 (NGRAM off).
- **PF5** (AR-quant A/B) — not pursued: `RADIANCE_USE_R4D_AR_QUANT` is not bit-identical and only
  helps a TP>1 all-reduce we do not run (DP, world_size 1), so it is inert here.
- **PF6** — recommendation: keep chunk 16384 (PF1 showed prefill is chunk-independent at 8k; the 16k
  decode delta was a cache-key artifact). The 8k compromise is only worth revisiting with a proper
  N>1 GDN-occupancy profile at long context.
- **PF7** — co-scheduling more sequences per step is a batching property; not a knob; documented.
- **ST1** — per-phase split is already answered by prior `RADIANCE_STEP_TRACE`/PHASE data (host-bound:
  ~28 ms `_bookkeeping_sync` + ~16.5 ms propose); no new run.
- **ST2** — V2-MTP boot blockers answered statically (V2 runs MTP; capture/FULL-graph decode working).
- **ST3** (matcher-context staleness, C4) — V1-only path (`radiance_draft.py`), inert on the served V2
  runner; no action.
- **ST4** (TOP1 confidence contract, C5) — subsumed by M2 (arm dropped).
- **P10** — harness exists (`bench-conc.py`, `bench-prefill-ttft.py`, `hit_oracle.py`, BetterBench).
- **P2/P6/P7** — P2 is a libr4d/R4D kernel-table item (rebuild); P6/P7 are conditional on the A1
  confirmation and currently low value; left parked.

## 2026-09-30 (cont. 58) — M1 RESOLVED: n-gram bs>=2 fault fixed via overlay

Deep research (cont.55 area) named the engine-only ops the standalone repro never exercised:
`st.all_token_ids.gpu.index_select(0, idx)` (a torch gather over the 160k-column **UVA host-mapped**
aperture — the prime suspect on gfx1201), plus a real `_nblk(base vs size)` bug that inflated the scan
grid ~9x. Implemented the safe fixes in `patch_dynamic_depth.py` `_radiance_ngram_extend`:
1. stage the context via the **pinned CPU source of truth** (`_uva_buf.cpu`) + H2D instead of the UVA
   torch gather;
2. read `num_computed_tokens` from its CPU mirror (`num_computed_tokens_np`);
3. pass the window **size** to `gpu._nblk` (was the base);
4. clamp `n` to the row width.

Result: **`RADIANCE_DRAFT_NGRAM=1` now runs bs>=2 cleanly** — conc8 x3 reps, HEALTHY, no HSA, matcher
active (`[ngram] rows/extended_rows/appended`). Off-server engine-exact repro (`aijuus/eng_ngram_repro.py`:
UVA `index_select`, host gather, matcher, all under a global-pool graph replay) PASSES every mode, so the
fault needs the real engine context; the fix works empirically.

**Policy: NGRAM stays default-off** — warm A/B with the fix: conc1 62.7 vs 67.8, conc8 310.5 vs 369
(-7.5% / -16%). The per-row matcher + host-staging syncs outweigh the ~8% extended-row draft gain on
this mix. T3 measured: ~8% extended rows on the bench prompt (repetitive content would be higher).
The fault no longer blocks enabling n-gram for repetitive/code workloads; a device-side gather (Triton)
would remove the sync cost if we ever want it on.

## 2026-09-30 (cont. 59) — M1 fix CORRECTED: it is the `_nblk` grid bug, one line

Follow-up experiment isolated the cause: reverting the context staging to the original **sync-free
on-device** `index_select` (keeping only the grid fix + clamp) still ran bs>=2 cleanly; then dropping the
clamp (grid fix only) also ran clean (conc8 x2, HEALTHY). So the fault is **not** the UVA gather and
**not** host syncs — it is the `_nblk(base vs size)` bug alone: the engine passed the window *base*,
inflating the scan grid ~9x so `q` ran past `ML` and the unmasked suffix load read outside the row
(gfx1201 HSA). The fix is one line in `_radiance_ngram_extend`:
`nblks = [gpu._nblk(int(n_np[i]), _win) for i in range(R)]` (window **size**, not `base_np[i]`).

Perf (warm, NGRAM=1 with the one-line fix): conc1 **62.8** vs 67.8, conc8 **313.6** vs 369 (-7.4% /
-15%). The per-row matcher launches (not the syncs) outweigh the ~8% extended-row draft gain on our
mix, so **NGRAM stays default-off**. The fault no longer blocks enabling n-gram for repetitive/code
workloads (T3: ~8% here). The earlier host-staging variant is unnecessary and was reverted.

## 2026-10-01 (cont. 60) — A1/A4/A5 executed

**A1 — E1 ENABLED (registry).** `RADIANCE_OFFLOAD_SUFFIX_INV=1` + `RADIANCE_OFFLOAD_EAGLE_INCLUDE=1`
added to both MTP entries' `server_env`; vllm-0 restarted, env confirmed, patches active. Warm E1-on
baseline: **c1 71.3 / c8 371.6** vs E1-off 67.8/369 → neutral-to-slightly-positive, no regression.

**A4 — EXACTSET/FUSED.** Warm A/B (FUSED=1): EXACTSET=0 → c1 67.3 / c8 370.4; EXACTSET=1 →
c1 71.3 / c8 371.6, and higher acceptance. **Keep EXACTSET=1** (registry already correct); FUSED=1.

**A5 — fusion confirmation (prefill, 15.6k prompt).** fusion on → **2230 tok/s** (TTFT 6979 ms);
fusion off (`FUSE_RMS_QUANT=0`+`FP8_STREAM=0`) → 1949 tok/s (TTFT 7989 ms). **−12.6%** → the fusion
earns its keep; **keep it on**. P6 (folding the attention-output→o_proj quant) would chase headroom the
fusion already captures; deprioritised.

## 2026-10-01 (cont. 61) — A2/A3 n-gram workload A/B

Repetition-heavy code prompt (~3.7k tok), warm:
| arm | c1 | c8 |
|---|--:|--:|
| NGRAM=0 (generic mix ref) | 46.6 | 82.1 |
| NGRAM=1 (M1 fix) | **50.9 (+9%)** | **109.1 (+33%)** |

So n-gram is a **large win on repetitive/code content** (+33% c8) and a loss on the generic mix
(-15%, cont.59). Workload-dependent → keep default-off unless the deployment is code/repetitive-heavy;
A3 (depth+ngram) is the same axis and is covered by this. (extended_rows ~3% here.)

## 2026-10-01 (cont. 62) — adaptive n-gram gate implemented + enabled

Deep research (dynamic n-gram gating) → implemented a **per-request productivity gate** in
`patch_dynamic_depth.py` (`RUNNER_TAIL`): per-request EMA of extend-rate + warmup + periodic probe,
per-row skip, and a **batch-level early return** (no gather/launches/syncs when the whole batch is
cold). Gated `RADIANCE_DRAFT_NGRAM_ADAPT`; output lossless (cold rows keep their MTP draft).

Warm A/B (c1 / c8):
| workload | NGRAM=0 | NGRAM=1 static | NGRAM=1 + ADAPT |
|---|--:|--:|--:|
| generic | 67.8 / 369 | 62.8 / 313.6 | **64.9 / 380.3** |
| repetitive code | 46.6 / 82.1 | 50.9 / 109.1 | **52.2 / 115.6** |

So the gate **recovers the generic loss** (c8 ≈ baseline) while **retaining the repetitive win**
(c8 +41% vs NGRAM=0). Enabled **by default** in the registry (`RADIANCE_DRAFT_NGRAM=1`,
`RADIANCE_DRAFT_NGRAM_ADAPT=1` on both MTP entries); vllm-0 restarted, env confirmed, HEALTHY.
This turns the manual workload toggle into automatic per-request behavior.

## 2026-10-01 (cont. 63) — variant B: independent n-gram depth (enabled)

Deep research: the "row width == scheduled K" rule is a stale workaround for the misdiagnosed HSA
fault — the worker-driven width means a request's draft list length is free (≤ SPEC=8, capture ladder
must cover 1+M). So an independent n-gram depth is overlay-only. Implemented variant B in
`patch_dynamic_depth.py`: `RADIANCE_DRAFT_NGRAM_DEPTH` (M), `_BS_MAX` (apply M only at bs≤2),
`_NGRAM_MIN` (decoupled gate: `mlen>=STRONG and cl>=MIN`, emit `min(cl,M)`), `W=max(K,M)` on fired
steps. Adaptive gate/`NGRAM=1` unchanged.

Warm A/B on the repetitive code probe (M=8 at bs≤2 vs adaptive M=K):
| arm | repetitive | generic |
|---|--:|--:|
| adaptive (M=K) | c1 52.2 | c1 64.9 / c8 380.3 |
| variant B | **c1 56.8 (+8.8%)**, c2 96.3 | **c1 70.7 (+9%)**, c8 361.9 (M not applied at R>2; ≈noise) |
No wedge; HEALTHY. So there **is** headroom at bs1-2 (K was binding on repetitive), and M>K is safe
now that `_nblk` is fixed. **Enabled by default** in the registry (`NGRAM_DEPTH=8`, `NGRAM_BS_MAX=2`).

Note: the `cl/mlm` histogram print was finicky (the `RADIANCE_DRAFT_NGRAM_HIST` env didn't reach the
EngineCore, so I made recording unconditional; the print still didn't surface in `docker logs` this
session). Variant B's own gain demonstrates the headroom; the histogram can be revisited if needed.

## 2026-10-01 (cont. 64) — C1/C2: batched n-gram matcher (bit-identical, perf-neutral, kept)

**Motivation.** `_radiance_ngram_extend` ran `gpu.match_gpu` once per *armed* row (B=1) with one
`.cpu()` sync per row, i.e. R launches + R D2H syncs per propose step on repetitive/code workloads.

**Change** (`patch_dynamic_depth.py` `RUNNER_TAIL`). One batched `match_gpu` call over all armed rows,
one max-block grid, one D2H -- gated `RADIANCE_DRAFT_NGRAM_BATCH` (default **1**). The grid uses
`max(nblk_i)`; a shorter row's extra blocks are masked (`alive = q < n-1`) so they emit no key. The
old per-row B=1 path is kept as the `=0` fallback.

**Safety.** Standalone in-container test (`batched_ngram_equiv.py`: 40 iters, B 1..8, random
n/window, seeded suffix repeats) -- batched pack is bit-identical to per-row and raises no gfx1201
HSA fault. Acceptance/per-position rates identical in the live A/B.

**Warm A/B** (each arm verified warm, c1 ≈ 68, before measuring; `ab.env` BATCH 0 vs 1):

| workload | per-row | batched |
|---|--:|--:|
| repetitive c1 | 61.8 | 61.9 |
| repetitive c8 | 130.8 | 132.1 |
| generic c1 | 67.9 (base) | 67.8 |
| generic c8 | 357.2 (base) | 362.7 |

All within run spread. So the per-row launches/syncs are **not** on the critical path at these
context sizes -- the target forward dominates, and the adaptive gate already skips the matcher
entirely on generic content. Batching is a lossless reduction in per-step ops with **no regression**;
kept as the default (expected to matter more at long context, where the window scan is larger). No
registry change needed (default on); `ab.env` removed.

**Ops note (re-confirmed).** After a reload the *first* boot can be ~3x slow (observed c1 **22 t/s**)
while AOT/graph state settles; the next warm restart returned **71.0 t/s**. Always re-verify c1 ≈ 68
before measuring, never trust the first post-reload boot.

## 2026-10-01 (cont. 65) — A9/OS3 lazy-GDN: overlay repaired, but **blocked on an r4d.so rx10 rebuild**

**What I set out to do.** Implement E3 route 3c (fail-closed lazy-GDN rollback). The empirical first
step is to make `RADIANCE_GDN_LAZY=1` *boot*, then reproduce the multi-turn corruption.

**Finding 1 — the overlay had drifted and was un-appliable (crash-loop risk).**
Enabling lazy aborted entrypoint (`patch_gdn_lazy.py` `apply` raises) → the container **crash-looped
(RestartCount 17)**. `_patchlib.apply` reported `anchor matched 0x` on `abstract.py`. Audit of all 16
anchors: **8 mismatched**. Upstream had changed the source (`num_speculative_blocks` now guards
`cache_config.use_kda_recoverssm` and uses `vllm_config.num_speculative_tokens`; kernel sigs gained
`TEMPORAL_TILES`; `initialize_from_forward_context` now delegates to `_populate_metadata`;
`get_mamba_groups` return annotation changed). I repaired all 8 anchors in `patch_gdn_lazy.py`; the
audit is now **16/16 OK** and the patch applies (`patch_gdn_lazy: done`).

**Finding 2 — the base image's libr4d has NO lazy kernels.** With the patch applying, the engine logged:
`[radiance.gdnmerge] gdn fused counter init failed: RuntimeError('RADIANCE_GDN_LAZY=1 but this libr4d
has no gdn_lazy_update kernel (needs rx10+)')`, then HSA-faulted during CUDA-graph capture. The mounted
`/r4d/r4d.so` (v0.5.0-w4a16, 1.9 MiB) predates the `r4d_gdn_lazy_update_k128_v128` extras in
`r4d_radiance_extras_rx10.patch`; the host cache has only `b9e42ab-rx6`, `b9e42ab-rx9`, `v0.5.0-w4a16`
— **no `…-rx10`**.

**Consequence.** OS3/A9 is **not** a no-rebuild item: the *Python* side is runtime-only (now repaired
and ready), but the *kernel* side needs `libr4d` rebuilt with the rx10 extras (`AUTO_R4D` /
`R4D_KEY=…-rx10`) and mounted at `/r4d`. The configured `vllm_config.num_speculative_tokens` plumbing
means `abstract.py` must be patched for the kernel too. **Reclassified: rebuild-class (libr4d rx10),
user-owned.**

**Measured payoff (motivates the rebuild).** With lazy enabled the KV pool grew
**171,320 → 190,157 tokens (+11%, 1.07× → 1.19× concurrency for 160k)** because the per-request spec
blocks drop 9 → 3. Worth the rx10 rebuild.

**Recovery.** lazy reverted (`ab.env` removed), warm restart: health 200, c1 **70.9** (cold boot was
21.3 as usual). Site-packages keep the now-inert lazy patched code (all runtime-gated by
`RADIANCE_GDN_LAZY`).

**Ops hazard noted.** A single stale env-gated overlay aborts entrypoint under `set -e` and crash-loops
the container. Consider making optional overlays loud-but-non-fatal, or validating anchors at apply
time before the service is torn down.

## 2026-10-01 (cont. 66) — libr4d rx10 built; A/B: b9e42ab-rx10 **+6%**; lazy still blocked (3 overlay defects)

**Build.** Cloned `codeberg.org/StillDeadcode/libr4d.git`, `checkout b9e42ab`, applied
`r4d_radiance_extras_rx10.patch` (clean), ran `./build.sh` (GFX_ARCH=gfx1201) inside
`juupp/vllm-radiance:0.9.3-collect-tokens`. ~1 min. Published
`~/.cache/radiance-libr4d/b9e42ab-rx10/r4d.so` (1,947,888 B, sha 2393c5d0); exports
`gdn_lazy_update` + `gdn_lazy_materialize`. Recipe verified against `serve-mxfp4.sh:559-573`.

**A/B.** Swapped `~/.cache/radiance-libr4d/v0.5.0-w4a16/r4d.so` (the mount source the entrypoint
copies over site-packages on every start); stock saved to `/tmp/kilo/r4d_stock.so`. Warm arms
(c1≈warm before measuring):

| arm | libr4d | lazy | c1 | c8 | acc/draft |
|---|---|:--:|--:|--:|--:|
| B0 | stock v0.5.0 | off | 71.3 | 366.4 | 1.93 / 1.93 |
| B1 | b9e42ab-rx10 | off | **75.9** | **386.4** | 2.17 / 2.01 |

= **c1 +6.5%, c8 +5.5%**. The `no matching narrow-state kernel` warning is gone and the GDN fused
update now resolves for fp32/bf16/fp16 state — the stock mount had been declining the fp16 GDN fused
path to the FLA fallback. libr4d now reports **0.4.0, 22 kernels, 16/18 queries**. Quality sanity on
rx10: `17*23=391`, correct Fibonacci. Caveat: acceptance ROSE (1.93→2.17), so the win is partly
numerics-driven (rx9 fp32-accumulate/RTNE state) and outputs are **not** bit-identical to stock.

**Lazy (B2) still cannot run — three independent overlay defects, only two fixed:**
1. stale anchors in `patch_gdn_lazy.py` (fixed cont.65).
2. **non-idempotent** `patch_gdn_lazy.py`: sentinels were `SENT + " word"` but replacements contain
   only bare `SENT`, so a second apply re-inserted/consumed anchors → `FAIL` → crash-loop
   (RestartCount reached 18; abstract.py had 19 stacked helper copies). **Fixed:** every sentinel is
   now a stable substring of its own replacement; `abstract.py` reset to pristine and re-applied
   (1 helper). Re-run confirms all-but-abstract NOOP.
3. **`radiance_gdn_lazy.py` runtime API drift (NOT fixed):** with the kernel present, engine init
   dies at `_Tables.__init__` (`radiance_gdn_lazy.py:49`) with
   `IndexError: tuple index out of range` on `copy_funcs[st_idx]` — the mamba state-copy spec
   structure changed since this module was written.
Plus lazy is still **fail-open corrupt** (`r4d_radiance_extras_rx10.patch:1503-1509`) and needs
route 3c. So lazy stays OFF; the rebuild's realized value is the rx9 narrow-state perf (+6%).

**State now:** rx10 mounted (the `v0.5.0-w4a16` dir repurposed; rename/compose point pending), lazy
off, health 200, warm c1 75.8. Compose still names `v0.5.0-w4a16` — should be repointed to
`b9e42ab-rx10` for honesty.

## 2026-10-01 (cont. 67) — (c) fixed: lazy BOOTS (+11% KV); (3) route 3c premise is wrong

**(c) `radiance_gdn_lazy._Tables` runtime drift — FIXED.** The module indexed
`copy_funcs[st_idx]` on `tuple(ctx._radiance_copy_funcs)`, but `ctx._radiance_copy_funcs` is now the
per-type `MambaStateCopyFuncsByType` dict, so `tuple(...)` gave its KEYS and indexing raised
`IndexError`. Fix: per layer, `mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, name)` (V2
mamba_utils) then `copy_funcs = tuple(ctx._radiance_copy_funcs[mamba_spec.mamba_type])`, zipped
positionally with `layer.kv_cache`. Runtime-copied from /patches, so no rebuild.

With `RADIANCE_GDN_LAZY=1` the engine now **boots clean**:
`[radiance.gdn.lazy] materialize tables: 48 temporal states, H 48 Hg 16 st_head 16384 state
torch.float16`; `gdn_lazy_update` fp32/fp16 kernels resolve; **GPU KV cache 190,157 tokens (+11%
vs 171,320)**; health 200; single-turn output coherent (17*23=391, correct Fibonacci). So (c) is
done: lazy runs, the memory win is real.

**(3) route 3c is NOT implementable as a pure-Python overlay — premise corrected.** The fail-open is
decided *on-device* inside `r4d_gdn_lazy_materialize_kernel` (`r4d_radiance_extras_rx10.patch:1503-1509`:
`r=0` on a stash-header mismatch, then the unconditional store). Python cannot observe that decision
without a per-step readback, and it cannot *repair* it: the correct value is the state after `count`
candidates of the previous step, whose only source is the candidate inputs in the stash block — the
very thing that went invalid. A host ring holding that data is ~18 MB/req/step (infeasible). So the
"host-snapshot ring + 2-slot GPU stage, restore from Python" plan does not hold.

Two concrete paths from here (both need an `r4d.so` edit + the ~1 min rx10 rebuild):
- **3a (safe, small):** kernel writes a per-request `stale` bit when it would fail open with
  `count>0`; the V2 runner reads it (async) and raises/logs. Lazy then fails **loud**, never silently
  corrupts — makes it safe to enable for the memory win even before the repair.
- **K2 (repair):** dedicate ONE extra state page per request as a backup stash (3 -> 4 pages vs 9
  stock) that survives block reuse; materialize reads it on mismatch. Real fix, needs multi-turn
  validation via `turnbench`.

**State:** rx10 live, lazy OFF, health 200. Until the compose redeploy, vllm-0 runs rx10 via the
repurposed `v0.5.0-w4a16` dir.

## 2026-10-01 (cont. 68) — lazy prefill invalidation wired (overlay-only); multi-turn CLEAN

**Fix (overlay, no rebuild).** The intended "prefill invalidates its stash" step was dead metadata
(cont.67 §8). Wired it:
- `radiance_gdn_lazy.invalidate()` + `_lz_invalidate_kernel` (Triton): per prefilling row, zero the
  4-byte magic of every head region of the request's stash block (`bt[state_idx+1]`), so a stash
  written against a reused physical base_slot in an earlier context can never replay.
- `patch_gdn_lazy.py` patches `mamba_hybrid.preprocess_state` to call it (for `input_batch.is_prefilling_np`
  rows) right before `run_fused_precopy`.

**Bring-up bugs (all fixed):**
1. Offsets: `state_ptrs` are raw BYTE addresses in the triton kernel, so `slot_strides` (bytes) and
   `st_head * itemsize` are byte offsets — using elements wrote the wrong slots.
2. `idx_mapping` must be passed as the **tensor**; a `.data_ptr()` int is a triton *scalar*, giving
   `CompilationError: Unsupported ptr type triton.language.int64 in tl.load`.
3. Patch idempotency skipped the updated 6-arg hook (sentinel already present) → reset
   `mamba_hybrid.py` to pristine and re-applied.

**Result.** lazy + fix boots clean: `materialize tables: 48 temporal states`, **KV 190,157 (+11%)**,
health 200. **8-turn multi-turn CLEAN** (no empty replies, no repeat loops). A turn-5 empty reply in
the first run was a reasoning token-cap artifact (`finish=length`, ct=600), not corruption.
`turnbench --concurrent` could not complete (harness `ConnectionResetError`; the engine stayed
healthy), so the canonical gate is still outstanding.

**Status.** Not yet an exactness comparison vs lazy-off, and the canonical gate is incomplete; lazy
kept **OFF** (rx10, no `ab.env`) pending stronger validation. Files: `radiance_gdn_lazy.py`,
`patch_gdn_lazy.py`.

## 2026-10-01 (cont. 69) — lazy validation + enabled in the registry

**Gate.** `mt_gate.py`: a ~3k-token shared document + 10-turn growing conversation (prefix-cache reuse
every turn), temp 0, thinking off, run under lazy-off (control) and lazy-on (+fix):

| arm | result |
|---|---|
| lazy OFF | CLEAN (10/10 turns, no empty/loop) |
| lazy ON (+fix) | CLEAN (10/10) |

Outputs are **semantically equivalent** (same edge case, same O(n), same functions) but **lexically
divergent** (mean char-similarity 0.40) — expected: lazy's fp32 replay is explicitly not bit-identical
to eager, and greedy decoding cascades the first differing token. Health (the corruption signature)
is clean.

**turnbench is unusable in this build:** its default endpoint is `127.0.0.1:8080` (not our server), and
pointed at the right base its token sizer hits `/tokenize` → HTTP 404. So the canonical
`turnbench --concurrent` gate could not run; the multi-turn gate above is the available evidence.

**Enabled.** `RADIANCE_GDN_LAZY=1` added to `mtp-27B-MXFP4-blend.server_env` (commit `05288ea`);
boots via the registry with no `ab.env`: KV **190,157 (+11%)**, sanity correct. Perf: c8 **+3.2%**,
c1 −1.8%. Revert = remove that one key.

## 2026-10-01 (cont. 70) — post-enable housekeeping

- **lazy is live via the registry** (`RADIANCE_GDN_LAZY=1` on mtp-blend, commit `05288ea`); KV
  **190,157 (+11%)**, sanity OK. Revert = remove that one key.
- **BetterBench** (v0.4.0 at `~/betterbench/.venv/bin/betterbench`) command for the running vllm-0
  recorded (all three phases, `config/default.json`, `--note lazy=on`). For a comparable A/B run it
  once lazy-on and once eager (remove the key + reload). `turnbench` remains unusable in this build
  (default endpoint `127.0.0.1:8080`; `/tokenize` → 404).
- **Loose end — libr4d mount.** Compose now points at `b9e42ab-rx10`, but until the next redeploy the
  running container still uses the repurposed `v0.5.0-w4a16` dir (which holds the rx10 `.so`). After
  the redeploy, restore `~/.cache/radiance-libr4d/v0.5.0-w4a16/r4d.so` to the stock build
  (`/tmp/kilo/r4d_stock.so`, sha `b83307c8`) so the dir is honest.

## 2026-10-01 (cont. 71) — vllm-0 crashed under BetterBench: prefill-chunk OOM (KV over-provision)

First proper BetterBench run on vllm-0 (lazy on), `aijuus/bench/vllm0-lazy.betterbench.{json,html}`:

- **single-stream** combined decode **90.3 t/s** median, update p99 **54.3 ms**, TTFT p50 **83 ms**
  (chat 70.2 · code 91.9 · file_edit 104.0 · json 110.1 · math 103.7 · prose 64.1 · reasoning 81.3 ·
  summarization 99.4).
- **prefill** PP t/s median 1924 (2k) / 2388 (8k) / 2570 (16k) — then **32k and 64k FAILED**.

**Failure.** At 12:16:47 (prefill depth ~32000) EngineCore hit
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.03 GiB. GPU 0 has a total capacity of
31.86 GiB of which 0 bytes is free ... 27.61 GiB allocated by PyTorch ... 2.45 GiB reserved but
unallocated`. Stack: `radiance.mxfp4_linear_pq` (`radiance_mxfp4.py:610`) allocating the `(M, N)` bf16
output of the W4A8 GEMM for the 16384-token chunk. The engine died → every 32k/64k prefill retried 500,
then connection-refused; the concurrency sweep (1/2/4/8, all 0/48) is **invalid** (engine already gone).
The container auto-restarted via its restart policy (Kilo did not restart it) and came back healthy.

**Cause: KV pin is over-provisioned vs `max_model_len`.** Same 6.5 GiB pin now yields **190,157 tokens**
(36,704 B/tok) but `max_model_len=160,000` → **30,157 tokens = ~1.03 GiB of KV is unusable** — exactly
the failed allocation. (Lazy lowered per-token KV cost so the same bytes hold +11% more tokens; the
device KV bytes are unchanged, so lazy is not the trigger, the pin is.) The 16k depth fits, 32k (2nd
chunk with the growing KV in flight) does not.

**Fix options (all need a restart, which the user owns):**
1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (server_env) — reclaims the **2.45 GiB**
   reserved-but-unallocated fragmentation the error itself flags; >1.03 GiB shortfall with margin,
   leaves chunk 16384 and KV 6.5 GiB untouched. Lowest behavioural risk. **Recommended first.**
2. Right-size KV: `kv_cache_memory` 6.5 GiB → **~5,873,000,000** (~160k tokens) frees the ~1.03 GiB
   over-provision (user-preferred axis) but is an exact-fit with no margin.
3. Chunk 16384 → 12288 shrinks the peak activation (0.77 GiB) — a milder cut than the rejected 4k.

**Decision (user).** Lever 3: `max_num_batched_tokens` **16384 → 12288** (KV pin unchanged) — staged in
the registry. Pending the user's restart; re-run BetterBench (prefill at least) to confirm 32k/64k pass
and to re-measure single-stream/concurrency.

## 2026-10-01 (cont. 72) — calibrate-kv.sh is not aligned with our image (calibration path blocked)

Tried the faithful calibration (`IMAGE=juupp/vllm-radiance:0.9.3-collect-tokens TP=1 SNAP=…blend…
RADIANCE_GDN_LAZY=1 … ./calibrate-kv.sh`, chunks 2560/4096/12288). With our image the **lazy** patch
now applies, but every run dies at `patch_ar_maxbytes.py`, which targets `radiance_allreduce.py` — a
file our lineage **deleted** (`aijuus/patches/010-dockerfile.patch:67`). `serve-mxfp4.sh` runs that
patch under `set -e` (`:1086-1092`) so the FAIL is fatal; `aijuus/kv-offload/ops/entrypoint.sh:258`
runs the identical patch **without** `set -e`, so our real deployment tolerates it.

Image comparison (checked directly with `docker run --entrypoint`):

| image | `model_executor/layers/mamba/abstract.py` | `radiance_allreduce.py` |
|---|---|---|
| `stilldeadcode/vllm-radiance:0.9.3` | OLD: `num_speculative_blocks=(speculative_config.num_speculative_tokens if speculative_config else 0)` | present |
| `juupp/vllm-radiance:0.9.3-collect-tokens` | NEW: `num_speculative_blocks=(0 if use_kda_recoverssm else num_speculative_tokens)` | **missing** |

So neither image satisfies `serve-mxfp4.sh`'s repo-root patch stack for our config (lazy-on, TP=1,
blend): stock can't enable lazy; ours dies at the AR patch. `serve-mxfp4.sh`'s default
`IMAGE=stilldeadcode/vllm-radiance:0.9.3` (`:231`) is also stale for our lineage. **calibrate-kv.sh
cannot measure our stack.** Viable paths: (A) analytic pin + one restart + a CHUNK-sized prefill probe
on the live model-controller; (B) an aijuus live-stack calibration harness. Also: the calibration pin
lands in `~/.cache/radiance-mxfp4/kv-profiles.local.tsv`, which our registry does not read — the bytes
must be ported into `aijuus/model-registry.json` by hand.

## 2026-10-01 (cont. 73) — our own calibration harness: aijuus/calibrate-kv-live.py

Added **`aijuus/calibrate-kv-live.py`** to replace the upstream `calibrate-kv.sh` path (cont.72 showed
that one cannot drive our image). It drives the **real deployment**:
1. writes `kv_cache_memory` into `aijuus/model-registry.json` (the compose bind-mounts it at
   `/model-registry.json`, `coolify-compose-2gpu.yml:565`);
2. `POST /reload {"model":…,"instance":"vllm-0"}` to the model-controller (bearer auth) → it restarts
   the instance so `entrypoint.sh` re-reads the registry — the same path a hot-swap uses;
3. polls `/health`, then runs upstream's PASS test (one **CHUNK-sized prefill** + a short decode);
4. reads the boot's `GPU KV cache size` / `Initial free memory` / OOM from container logs and writes
   the best pin back to the registry (restores the original registry on abort).

Modes: default sweep (profile → +2% × 6 → back off 1 step); `--quick` (profile only); `--dry-run`;
`--no-reload` (probe the running server, change nothing); `--pins a,b,c` (explicit bytes). Env:
`MODEL_KEY`, `INSTANCE` (vllm-0), `CONTROLLER` (172.18.0.5:8101), `CHUNK`, `STEP`/`MAX_STEPS`/
`BACKOFF_STEPS`, `RELOAD_TIMEOUT`.

It reloads (restarts) the target instance — a deployment op, so the user runs it. Verified compiling +
`--dry-run` only so far (the compose was down at time of writing). Because we want the *minimum* pin
that holds 160k tokens (not the max), an explicit `--pins 5.87e9,6.06e9,6.5e9` run is the most direct.

## 2026-10-01 (cont. 74) — OOM fix VALIDATED on the live server; harness controller auto-discovery

- **`calibrate-kv-live.py` fix:** the controller has no published port, so the hardcoded
  `172.18.0.5:8101` was stale after the compose restart. It now **auto-discovers** the controller by
  compose-service label and picks its IP on the same network as the instance (`container_networks`),
  `CONTROLLER=` override still wins. Also fixed the CHUNK-sized probe prompt (it was ~1/9 of a chunk).
- **Validation (live vllm-0):** `'max_num_batched_tokens': 12288`, `GPU KV cache size: 190,157 tokens`.
  `calibrate-kv-live.py --no-reload` → CHUNK-sized prefill **HTTP 200**, `oom=False`. Then a direct
  **32k-token prefill** (the exact BetterBench case that failed at cont.71) → **HTTP 200 in 8.1 s**.
  The OOM is gone; chunk 16384→12288 is the confirmed fix.
- Registry pin was restored by the `--no-reload` run (it changes nothing); the registry still carries
  `kv_cache_memory=6979321856` (190,157 tokens). Optional next: reclaim the ~1.03 GiB that 160k can
  never use via `calibrate-kv-live.py --pins 6.06e9` (≈165k tokens) — a reload, user-run.

## 2026-10-01 (cont. 75) — calibrate-kv-live crash handling fixed; profiling is infeasible

**Defect found the hard way.** The first `calibrate-kv-live.py` used vLLM **profiling** as pass 1
(`kv_cache_memory=""`). On this model that cannot work: the profiler charges the transient/cudagraph
peak and sizes **4.05 GiB**, below the **5.45 GiB** vLLM says is needed for `max_seq_len=160000`
(`estimated maximum model length is 115280`), so the engine **refuses to boot**. The harness then sat
in its 20-min health wait while the container **crash-looped**, leaving the registry at `""`. SIGINT ran
the `finally`, restored `6979321856`, and the container re-read the registry on its next restart and
came back (health 200, KV 190,157).

**Fixes:**
- **No profiling.** The sweep starts from a known-good pin — the current registry value, or `--start`.
  Pass 1 now *verifies* that pin serves a CHUNK-sized prefill before raising.
- **Fast-fail.** `attempt()` watches the new boot's logs for engine-start failures
  (`Engine core initialization failed`, `larger than the available KV cache memory`, OOM, ...) and
  bails immediately instead of waiting on the controller's 3600 s `SWAP_TIMEOUT`.
- **Rescue.** On any failed pin it restores the last known-good pin and restarts, so the instance is
  never left crash-looping. Also waits for the container's `RestartCount` to advance before scoring a
  boot (the controller drains first, so the old pin keeps answering `/health` for a while).

**Useful datum:** vLLM reports the minimum pin directly — **5.45 GiB for 160k**, i.e. the floor;
6979321856 (6.5 GiB) gives 190,157 tokens, so there is headroom to raise.

## 2026-10-01 (cont. 76) — calibrate-kv-live works; KV ceiling explored to 7.6 GiB

After the `State.StartedAt` fix the harness drives the real stack correctly (registry write -> controller
`/reload` -> wait for the new boot -> CHUNK-sized prefill probe -> apply). Explicit run:

| pin | KV tokens | x for 160k |
|---|--:|--:|
| 6,979,321,856 (6.5 GiB, start) | 190,157 | 1.19 |
| 7,000,000,000 | 190,994 | 1.19 |
| 7,200,000,000 | 196,858 | 1.23 |
| 7,400,000,000 | 201,884 | 1.26 |
| **7,600,000,000 (applied)** | **207,748** | **1.30** |

All four probes PASSed, so the **ceiling is still above 7.6 GiB** (not found). ~36.6 kB/token, so the
160k floor is ~5.85e9 and each GiB buys ~27k tokens. **Caveats:** the CHUNK probe tests the worst-case
single step (bounded by `max_num_batched_tokens`) but not long-context fragmentation/concurrency — the
cont.71 failure mode — so a passing pin still needs a BetterBench long-context/concurrency run. KV past
160k only helps concurrency/prefix reuse. Note `--pins` keeps the edge and applies no backoff; the sweep
(`START=<bytes> MAX_STEPS=N`) backs off 2% for margin. Pin is provisional.

## 2026-10-01 (cont. 77) — quick KV-pin OOM validator; 7.6 GiB pin validated

Added `aijuus/bench-kv-validate.py`: the fast, purpose-built check for a `kv_cache_memory` pin --
a cold long-prefill sweep (the transient activation peak) plus a concurrency sweep (KV pressure), with
a PASS/FAIL verdict and OOM/500/refused detection. Unlike BetterBench it does no quality/plotting; it
answers "did the engine survive".

**Validated the applied 7.6 GiB pin (207,748 tokens):**
- prefill 16k / 64k / **160k** (155,105 actual tok) -> OK, PP 2381 / 2137 / 1677 t/s
- concurrency 8 x 4096-token prompts + 128 gen -> 8/8 OK, 33.9 agg t/s
- **VERDICT: PASS -- no OOM.**

## 2026-10-01 (cont. 78) — BetterBench @ chunk=4096, kv=7.6 GiB: 32k/64k prefill + concurrency VALID

Loaded `chunk=4096` (was 12288) keeping `kv=7,600,000,000` (207,748 tokens) via registry + controller
`/reload`; health 200. Full BetterBench 0.6.0 pass (~44 min), reports
`aijuus/bench/vllm0-kv76-chunk4k.betterbench.{json,html}`.

- **Single-stream** combined decode **89.1 t/s** (vs 90.3 at 12k/6.5G) -> decode unchanged; TTFT p50
  **84 ms**, update p99 **55.0 ms**.
- **Concurrency (valid this time):** level 1/2/4/8 = **37.2 / 70.0 / 135.3 / 199.5** agg t/s, **48/48 ok**
  each (the cont.71 run showed 0/48 because the engine was already dead).
- **Prefill (valid, no OOM):** 2k 1851 / 8k 2251 / 16k 5159 / 32k 3061 / 64k 7560 PP t/s. The 32k and
  64k rows are the ones that OOM'd before. PP is non-monotonic (16k/64k sit above 8k/32k) -> treat the
  prefill medians as noisy and repeat before drawing chunk-vs-throughput conclusions.

**Net:** 4k chunk + 7.6 GiB is robust end-to-end (long prefill + concurrency + decode), confirming it is
a safe operating point; decode is not affected by the chunk change.

## 2026-10-01 (cont. 79) — n-gram draft tail: root cause + trust-gated fix (F1/F2/F5/F4)

**Symptom:** concurrent/normal agent traffic showed MTP acceptance collapsing to <5% (unusable decode),
and responses felt degraded/cut. Isolated probes (math, tool calls, 48k-token retrieval, prefix-cache
reuse, greedy 8x determinism) were all correct on BOTH models -> the model/kernels were fine; the
regression was the **draft policy**.

**Root cause (evidence).**
- Two n-gram implementations exist. `radiance_draft.py::slot_decide` has the good policy (agreement OR
  long-match OR recency, + confidence/TAU) but hooks the legacy `SpecDecodeBaseProposer._greedy_sample`,
  which the **vLLM-0.29 V2 runner never calls** (`greedy_sample ENTERED` never logs) -> it is dead code.
- The live path is runner-side `patch_dynamic_depth.py::_radiance_ngram_extend`: on `mlen >= STRONG(8)`
  (variant-B: `clen >= NGRAM_MIN(3)`) it **replaces the whole MTP draft** with the verbatim continuation
  -- no agreement, no confidence, no frequency.
- The captured agent request repeats tool-schema boilerplate (`"type":"object","properties":{"<name>"`,
  `"$schema":...`, `"description":"` x9). The matcher finds a long exact suffix match on the boilerplate
  but the continuation is a DIFFERENT field name -> the target rejects it. Matches saturate
  (`clen8 1803/2000`, `mlen32 1365/2000`), so the override fires ~every step and throws away the
  reliable MTP draft -> per-position-0 acceptance bimodal: **0.97 healthy vs 0.028 collapse**, identical
  on both models. Turning `RADIANCE_DRAFT_NGRAM=0` restored steady 48-70% (A/B confirmed live).
- `RADIANCE_DRAFT_NGRAM_ADAPT` cannot help: its EMA signal is `ext_flags` = "took a tail" (always ~1),
  not "was accepted" -> it never disarms.
- Matcher kernels are correct: `batched_ngram_equiv.py` (batched == per-row) and the GPU selftest pass,
  so this is **selectivity**, not a kernel bug. (Safety invariant holds: proposals cannot change output.)

**External research (why match length alone is wrong).** vLLM 0.29 ships native Suffix Decoding (Arctic
`SuffixDecodingCache`, `min_token_prob`=0.1) -- a **frequency-weighted suffix tree** that would not
speculate on boilerplate. TensorRT-LLM *combines* a suffix automaton with the neural drafter
(`sa_spec_threshold`) and notes neural is better for novel content. MNN lookahead uses frequency x length;
SpecDec++/Xu gate on predicted rejection/confidence. `arctic_inference` is not installed here, so the
frequency idea is ported as a cheap proxy (top-2 determinism).

**Fix design (ranked).** F1 agreement gate (tail slot-0 == MTP slot-0) · F2 determinism gate (top-2
matches agree on slot-0) · F3 confidence gate (needs V2 conf wiring) · F4 fix adaptive feedback
(agreement-rate EMA) · F5 threshold hygiene · F6 frequency-weighted suffix matcher (port Arctic).

**Implemented (this cont.):** F1, F2, F4 + F5 partial in `patch_dynamic_depth.py` (runtime overlay, no
rebuild), behind new knobs `RADIANCE_DRAFT_NGRAM_AGREE` / `_DET` (default ON), `_EMA_MIN` 0.02->0.10.
`AGREE=0 DET=0` restores the pre-fix behaviour for A/B. F3/F6 deferred. Offline policy test added.

**Validation (offline, no GPU):**
- `aijuus/patch_dynamic_depth_policy_test.py` -- extracts + execs the injected `RUNNER_TAIL`, asserts
  the gate declines the boilerplate case and accepts the echo case, and the AGREE/DET toggles. **PASS (9/9).**
- `ngram_draft_selftest.py` (legacy `slot_decide`) **ALL PASS**; patch file `ast.parse` OK.
- Applied the patch to the **pristine image** files in a throwaway container: anchors matched, both
  `speculator.py` and `model_runner.py` patch + `ast.parse` clean -> restart will apply it.

**F3 (confidence gate) is blocked on the served path:** the decode loop is a replayed full CUDA graph
(`radiance_draft.py` V2_CONF comment), so the draft-head confidence is not visible in Python at
serving time. F6 (true frequency-weighted suffix matcher, Arctic `min_token_prob`) remains the proper
end-state but needs kernel work + GPU validation; F2's top-2 determinism is the cheap proxy for now.

**Enable recipe (after you restart):** set in the MTP entry `server_env` `RADIANCE_DRAFT_NGRAM=1`
(AGREE/DET default ON; optional `RADIANCE_DRAFT_NGRAM_DEPTH=0` to disable variant B). Watch
`SpecDecoding metrics` per-position acceptance + `[ngram] extended_rows/appended`. Rollback = `=0`.

**Review follow-up (same cont.):** self-review found 3 refinements, all fixed + re-validated (12/12):
1. DET now takes the other candidate's match length and only vetoes against another **STRONG** match, so
   a short unrelated 2nd match cannot suppress a valid long one.
2. The F4 agreement EMA now guards on `clen>0`, so a row whose matcher did not run (pk zeros -> cont 0)
   cannot fake agreement when MTP's token id is 0.
3. `RADIANCE_DRAFT_NGRAM_STRONG<=0` restored as an explicit kill-switch (the old `>0` guard was implicit).

## 2026-10-01 (cont. 80) — F6 frequency-weighted n-gram gate (overlay); F3 confirmed blocked

**F6 implemented as a runtime overlay** (Arctic Suffix-Decoding `min_token_prob` idea, no rebuild):
- `radiance_draft_gpu.py` (overlay, `/patches` = repo root): new `_match_count` Triton kernel +
  `match_count()` wrapper + `occ`/`agree` int32 buffers in `make_match_buffers`. Per row it scans the
  context and counts **other occurrences** of the top-1 matched suffix (`L >= Lref`) and how many share
  the reference continuation's first token. Deterministic (one program per row, no atomics).
- `patch_dynamic_depth.py`: new knobs `RADIANCE_DRAFT_NGRAM_FREQ` (default **1**), `_MIN_FREQ` (1),
  `_MIN_PROB` (0.5). `_radiance_ngram_ok` is now: fast path (F1/F2) OR -- when MTP disagrees or the top-2
  are ambiguous -- allow **only** if `agree_other >= MIN_FREQ` and `(agree+1)/(occ+1) >= MIN_PROB`.
  Candidate 2 keeps the F1/F2 gates only (freq is measured for the top-1 suffix). The host runs
  `match_count` on the armed rows and feeds the counts in. `FREQ=0` = F1/F2 only; `AGREE=0 DET=0 FREQ=0`
  = pre-fix.

**Validation:** `aijuus/ngram_freq_gpu_test.py` -- real Triton `_match_count` vs a CPU longest-suffix
reference -- **PASS (40 iters, B 1..8, windows 0/256/1024)**; `batched_ngram_equiv.py` still **PASS**;
policy test **17/17**; patch applies to the pristine image (`model_runner.py` + `speculator.py`) and both
`ast.parse` clean.

**F3 confirmed BLOCKED (evidence):** the run args show `cudagraph_mode: FULL_AND_PIECEWISE`, so
`init_cudagraph_manager` sets the draft-decode mode to `FULL_DECODE_ONLY`
(`speculator.py:154-158`) -- draft decodes are graph-replayed and the Python sampling body
(`_greedy_sample_draft`) never runs at serving, so draft-head confidence is not host-visible. Enabling it
would require baking a confidence kernel into the captured draft graph (buffer indexed by
`current_draft_step`); parked since F1 (agreement) + F6 (frequency) already cover the failure.

**Enable recipe:** `RADIANCE_DRAFT_NGRAM=1` (AGREE/DET/FREQ default ON). Tune `_MIN_PROB` upward for
stricter override; `_FREQ=0` to fall back to agreement-only; `_NGRAM=0` to disable.

## 2026-10-01 (cont. 81) — CRITICAL: `docker start` did NOT re-apply the overlay; gates were never live

**Symptom:** after "restarting to apply" the F1/F2/F6 gates, acceptance still collapsed (position-0
0.029, avg 1.3%) and `[ngram] extended_rows` stayed 54-74%. The env change applied but the code did not.

**Root cause:** `docker kill && docker start` reuses the container's **writable layer**. `entrypoint.sh`
applies `patch_dynamic_depth.py`, but `edit()` returns early when `MARK="patch_dynamic_depth"` is already
present in `model_runner.py`. The container was created 12:34 with the OLD tail; every subsequent
`docker start` saw MARK and skipped, so the injected `_radiance_ngram_ok`/F6 host code was never the new
version. (`radiance_*.py` IS copied unconditionally, so the F6 kernel file was fresh but unused.)
**A fresh container (recreate) is required for patch-code changes** -- a plain restart only re-applies env.

**Fix applied (overlay, no recreate):** re-inject just the `RUNNER_TAIL` into the running container's
`model_runner.py` (cut at the tail header `# ---- RADIANCE dynamic draft depth + n-gram tail`, append the
new `RUNNER_TAIL` extracted from `patch_dynamic_depth.py`), `ast.parse`, then `docker restart`. Anchors are
unchanged between old/new so only the tail needed replacing. Done for both vllm-0 and vllm-1; backups at
`/tmp/kilo/mr_backup.py` and `/tmp/kilo/mr_backup_v1.py`. The entrypoint now skips (MARK present) and keeps
the new tail.

**Live A/B (same model Thinkingcap, same env `_DEPTH=0 _MIN_FREQ=2 _MIN_PROB=0.75`):**
- **OLD code (vllm-1, before re-inject):** mean avg-acceptance **47.5%**, collapses to **1.3%**
  (position-0 0.029), `extended_rows` 54-74%.
- **NEW code (vllm-0):** **7 windows, mean 67.0%, min 55.7%, max 74.8%, zero windows <15%**, position-0
  0.83-1.00 -- at/above the `NGRAM=0` baseline (~66%) with n-gram still enabled.
**Conclusion:** the F1/F2/F6 gates work; the earlier "tightening didn't help" was entirely the stale
overlay. Both instances re-injected + restarted and re-verified (`_radiance_ngram_ok` present).

## 2026-10-01 (cont. 83) -- Arctic SuffixDecoding backend: implemented, offline-validated, LIVE CRASHED (HSA)

Implemented the research-recommended Arctic hybrid backend as an overlay and enabled it; it crashed the
engine on first live boot, so it is reverted to the Triton backend pending debugging.

- **Dependency:** `pip install --no-deps arctic-inference` (0.3.0, cp312, pure-python wheel built OK) into
  both containers. `from arctic_inference.suffix_decoding import SuffixDecodingCache` works; API:
  `start_request(id, prompt_ids)`, `add_active_response(id, ids)`, `speculate(id, context, max_spec_tokens,
  max_spec_factor, min_token_prob) -> draft{token_ids, score, match_len}`, `stop_request(id)`; **inputs must
  be int32**.
- **Overlay:** `patch_dynamic_depth.py` gained `_radiance_arctic_extend` + `RADIANCE_DRAFT_NGRAM_BACKEND`
  (`triton`|`arctic`), `_NGRAM_TAU` (score gate = expected accepted length), `_ARCTIC_DEPTH/_FACTOR/_MIN_PROB`.
  Per-request `SuffixDecodingCache`, fed incrementally from `req_states.all_token_ids.gpu` (D2H of the
  prompt once, then deltas + the last `depth` pattern); takes the suffix draft only when
  `score >= tau`, else keeps MTP (hybrid). Runtime `InputBatch` uses `idx_mapping_np`,
  `num_computed_tokens_np`, `prefill_len_np` (NOT `token_ids_cpu`/`num_prompt_tokens`).
- **Offline:** `aijuus/arctic_hybrid_test.py` (fake InputBatch) -- repeat -> Arctic tail taken,
  novel -> MTP kept, `tau` too high -> MTP kept. PASS.
- **Live:** first boot with `BACKEND=arctic` reached full graph capture then **`Queue error:
  HSA_STATUS_ERROR_EXCEPTION` + `GPU coredump` during `[dyn-depth] propose#5`** (bs=1). Engine wedged
  (health 000); reverted `BACKEND=triton` + restart -> healthy (200). Root cause of the HSA is NOT
  established: candidate is the per-row/per-step `.cpu()` D2H on `all_token_ids.gpu` inside
  `propose_draft_token_ids`, but the Triton path also does D2H there; GPU0 is `THROTTLED` (possible
  hardware instability). Needs a controlled repro before re-enabling.

**TODO (arctic):** reproduce on a quiesced/cool GPU; try batching the D2H (one gather/step) or feeding the
context from `input_batch.input_ids` instead of slicing `all_token_ids`; verify the draft width stays within
`num_speculative_steps`. Keep `RADIANCE_DRAFT_NGRAM_BACKEND=triton` until then.

## 2026-10-01 (cont. 84) -- Arctic HSA root-caused + fixed; result: prompt-lookup loses to MTP here

- **Repro:** the HSA is fully deterministic at `propose#5` (bs=1) on every boot. Bisection: run the arctic
  plumbing with `TAU=1e9` (never adopt) -> **HEALTHY**, so the D2H/speculate plumbing is fine; the fault is
  the **adopted draft**. Failing kernel: `at::native::indexSelectSmallIndex<c10::BFloat16,long,...>` = a
  token-embedding `index_select` with an out-of-range index.
- **Root cause:** during speculator warmup the arctic cache speculates degenerate/short drafts
  (`[0]`, `[0,0]`, `[0,0,0]`). An arctic-only row shorter than K is `-1`-padded, and the verify's embedding
  `index_select` loads `-1` as a token id -> OOB -> HSA on gfx1201. (Not the raw values; `-1` pad.)
- **Fix (overlay):** merge the arctic prefix INTO the full MTP row: `di` stays length K with **no -1**
  (override slot j only for valid, in-vocab arctic tokens, stop at the first invalid); plus an optional
  slot-0 agreement gate `RADIANCE_DRAFT_NGRAM_ARCTIC_AGREE`. After this, `BACKEND=arctic` boots and serves
  with **no HSA**.
- **Result (harness `ngram_ab_probe.py`, 5x1024, warm):**
  | prompt | MTP-only | Triton(agree) | Arctic AGREE=1 | Arctic AGREE=0 |
  |---|---|---|---|---|
  | repeat | 123.1 | 115.2 | 111.7 | 32.2 |
  | agent  | 100.5 | 103.4 | 94.3  | 40.3 |
  | novel  | 53.6  | 54.2  | 51.1  | 47.3 |
  Arctic `AGREE=1` adopts only ~1% of rows and costs ~10% (CPU `speculate()` on the step); `AGREE=0`
  adopts ~62% but those drafts are rejected -> 2-4x slower. Same shape as the ungated Triton n-gram.
- **Conclusion:** prompt-lookup (Triton matcher OR real Arctic suffix tree) does **not beat MTP** on this
  blend/Thinkingcap agentic workload -- MTP-only is fastest and simplest. Registry set
  `RADIANCE_DRAFT_NGRAM=0` (MTP-only). Arctic code retained (`BACKEND=arctic`, `ARCTIC_AGREE=1` = stable
  demo). The HANDOFF gates F1/F2/F6 remain in code for A/B. The real lever for this workload is elsewhere
  (e.g. DFlash, rebuild-class), not prompt-lookup.

## 2026-10-01 (cont. 85) -- prompt-lookup CEILING is high; the gap is policy (override), not the workload

Research subagent verdict was "policy-flawed + fundamentally capped"; our own empirical oracle **refutes the
"capped" half** and localizes the fix.

- **Ceiling probe** (`aijuus/ngram_ceiling_probe.py`, in-container, tokenizes prompt+gen, longest-suffix
  oracle): at the SERVED temp 0.7:
  | prompt | match>=8 on | first-token hit | mean oracle accepted len |
  |---|---|---|---|
  | repeat | 49.2% | 95.2% | 6.59 |
  | agent  | 67.7% | 97.0% | 7.02 |
  | novel  |  2.4% | 94.7% | 6.11 |
  i.e. when a match exists the continuation is right ~95-97% and is worth ~6.6-7 tokens -- MORE than MTP's
  ~3.5. There IS large headroom on repetitive/structured spans; the earlier "n-gram loses" is not the
  workload's fault.
- **Matcher correctness** (`aijuus/ngram_matcher_oracle_probe.py`): on a synthetic periodic sequence the
  Triton matcher equals the oracle 25/26 (only misses L<MIN=3). So the matcher is sound in isolation.
- **The real flaw is the POLICY**: `_radiance_ngram_extend` REPLACES the whole MTP row (`di = c[:K]`) --
  zero-sum, and it fires on boilerplate whose continuation diverges. Also the live context fed to the
  matcher can include unverified draft tokens.

**RECOMMENDED FIX (ranked, kernel overlays allowed):**
1. **Prefix-preserving EXTENSION** (not override): keep `MTP[0:K]`; append the suffix continuation only when
   it is consistent with the MTP prefix, at slots `K..num_speculative_steps-1`. Monotone-safe (cannot
   regress accepted length). This is the fix that harvests the measured ceiling.
2. **Draft-conditioned lookup kernel**: find a context occurrence whose continuation starts with the MTP
   draft `[0:K]`, then append its tail -- gives a trustworthy extension when MTP is right.
3. **Continuation-probability scoring** (extend `_match_count` to a continuation histogram) to decide how
   many append slots to spend.
4. Acceptance-based adaptive gate; drop the zero-sum override entirely.
Tree verification is NOT overlay-feasible in VLLM 0.29 (flat chain, no parent indices).

## 2026-10-01 (cont. 86) -- F7 prefix-preserving extension implemented; single-chain verify is the ceiling

- **Built** `_radiance_ngram_extend_row` (pure, unit-tested) + `RADIANCE_DRAFT_NGRAM_EXT`: keep the WHOLE
  MTP row and append the suffix continuation `cont1[K:clen]` ONLY when it agrees with the full MTP prefix
  (never overrides MTP -> monotone-safe). 21/21 policy tests pass; applied live, no HSA.
- **Result: fires ~0.1% of rows** (`extended_rows=5/4000`). `[ngram-hist]` shows the live matcher gives
  `clen8` on 29% of rows and `mlen>=8` on ~13% overall (≈45% of the rows the matcher actually runs on,
  consistent with the ceiling probe) -- so matching is fine; **full-K agreement with MTP is what's rare**.
- **Throughput** (5x1024 warm): repeat 110-117 (MTP 123), agent 92-101 (MTP 100.5), novel 51-53 (~MTP).
  i.e. neutral-to-slightly-negative; no win.
- **DEFINITIVE CONCLUSION for overlay prompt-lookup:** vLLM 0.29's spec decode verifies a SINGLE token
  chain. So any suffix proposal must either (a) OVERRIDE MTP -> regresses when MTP is right (the anti-
  correlation the research flagged; measured 2-4x slowdowns), or (b) require full-prefix agreement ->
  almost never fires. The measured high ceiling (cont[0] correct 95%, ~6.6 accepted) cannot be harvested
  without **tree verification**, which is not overlay-feasible in this version. Therefore **MTP-only is the
  optimum** for this deployment among overlay options; all prompt-lookup variants (Triton gates, Arctic,
  extension) are neutral-or-worse.
- Registry left at `RADIANCE_DRAFT_NGRAM=0` (MTP-only). All prompt-lookup code (F1/F2/F6 gates, F7
  extension, Arctic backend) remains in the overlay for future use/A-B.

## 2026-10-01 (cont. 87) -- TREE-VERIFICATION scope (planned; gated on an offline gain test)

**Why:** the only route that can harvest the measured prompt-lookup ceiling (cont.85: cont[0] correct
95%, ~6.6 accepted tokens) is to verify the MTP chain AND the suffix chain together, because single-chain
verify forces a zero-sum override (regresses) or full-prefix agreement (~0.1% fire). DFlash is out of scope.

**Scope (vLLM 0.29 V2 spec-decode path is chain-only everywhere):**
1. **Proposer** -- emit a shallow tree: root prefix; branch A = MTP chain `m[0:K]`; branch B = suffix chain
   `c[0:E]` (optionally the merged path). Generate parents[] + node token ids. (overlay-friendly)
2. **Runner input layout** (`v1/worker/gpu/input_batch.py:449 combine_sampled_and_draft_tokens` +
   `_combine_sampled_and_draft_tokens_kernel`) -- must lay out nodes with per-node parent/position and build
   a tree attention mask instead of the contiguous linear chain. Core change (Triton).
3. **Attention backend** (`attention_backend='R4D'`, custom gfx1201 decode kernel) -- must apply per-query
   ancestor masking. New/changed kernel. **Biggest risk.**
4. **Rejection sampler** (`v1/worker/gpu/spec_decode/rejection_sampler.py` `rejection_sample`) -- chain-indexed
   today; needs a tree verify (longest accepted root->leaf path). New Triton kernel.
5. **Metadata/commit** (`v1/spec_decode/metadata.py` `SpecDecodeMetadata`) -- add parent indices / tree
   structure; update `combine_sampled_and_draft_tokens` + `get_num_sampled_and_rejected` + the slot/position
   bookkeeping.
6. **Cudagraph** (`FULL_DECODE_ONLY`) -- a variable tree breaks capture; must use a FIXED-shape shallow tree
   (e.g. 2 branches, depth <= K) or fall back to eager for tree steps. Feasibility hinges on this.

**Feasibility verdict:** rebuild-class reimplementation of EAGLE-3-style tree spec inside this fork; NOT a
runtime overlay (contradicts the overlay-only workflow) and high engine risk. EAGLE-3/tree support was
removed in this V2 rewrite (`grep -rn tree|parent|branch` in `v1/worker/gpu/spec_decode/` is empty;
`medusa.py` has no tree code).

**DECISION GATE:** before any of the above, run the bounded OFFLINE gain quantification (cont.88) -- if a
tree cannot beat MTP-only by a meaningful margin in an oracle simulation, the rebuild is unjustified.

## 2026-10-01 (cont. 88) -- TR0 tree-gain quantification: NEGATIVE (tree adds ~nothing); MTP-only confirmed

- **Probe:** `RADIANCE_DRAFT_NGRAM_PROBE=1` (env-gated, measurement-only; returns MTP unchanged) logs, per
  step, the MTP draft `m`, the suffix continuation `c`, and the tokens actually committed. Analyzer:
  `aijuus/tree_gain_analysis.py` (3320 paired steps over repeat/agent/novel).
- **Result (correct alignment, drafts = committed minus the first bonus token):**
  - mean `A_mtp` (MTP-only accepted drafts) = **2.715**
  - mean `A_suffix` (suffix accepted drafts) = **0.930**
  - mean tree oracle `max(A_mtp, A_suffix)` = **2.715** -> **gain 0.000 tok/step (0%)**
  - on `mlen>=8` steps (563): `A_mtp 4.806` vs `A_suffix 4.442` -> MTP already wins.
- **Verdict:** the tree (MTP chain + suffix chain) would add **nothing** on this workload -- MTP already
  matches or beats the suffix exactly where long matches exist (the anti-correlation the research flagged).
  The earlier "high ceiling" (cont.85) was an artifact: it measured how well the suffix matches the target
  WITHOUT comparing to MTP, which is just as good there. **The tree rebuild (TR2-TR5) is UNJUSTIFIED; do
  NOT build it. MTP-only is confirmed optimal for this deployment.**
- **Incidental real bug:** the matcher's continuation is **off-by-one** vs the MTP draft -- `c[0]` aligns to
  the already-decided bonus token, `c[1:]` to `m`. This is why F1 (`c[0]==m[0]`) and F7 (`c[:K]==m[:K]`)
  almost never fired. If prompt-lookup is ever revisited, compare `c[1:]` to `m`, or run the matcher on a
  context one token longer.
- Registry left at `RADIANCE_DRAFT_NGRAM=0` (MTP-only). Model-registry n-gram knobs trimmed to just
  `RADIANCE_DRAFT_NGRAM=0` (the experimental knobs removed).

## 2026-10-01 (cont. 89) -- A/F/G/H: libr4d M=64 kernels (A) built+deployed; F N/A; G/H config-gated

Boot-log review of vllm-1 surfaced four items.

**A (done): small-M GEMM kernels.** The deployed r4d.so was the lean rx build (no m64/w4a16) -- but the
BAKED image's r4d.so HAD `gemm_bf16_nt_m64` / `w4a16_nt_m64` / `w4a8_nt_m64`; mounting the lean build
dropped them. Root: the rx10 host tree was based on an older libr4d commit (`b9e42ab`) that predates those
kernels; upstream tag **v0.5.0** has them.
- Built **rx11 = libr4d v0.5.0 + `r4d_radiance_extras_rx10.patch`** (merged 3 rejects: build.sh UNITS,
  r4d_module.hip pybind defs, r4d_registry.hip `cAr3Exact`; kept the v0.5.0-only units quant_act_i8 +
  dflash_conv). Source tree `/tmp/kilo/libr4d-v0.5.0`; artifact `~/.cache/radiance-libr4d/b9e42ab-rx11/r4d.so`
  (sha 8d9b8096..., 2,629,000 B, `r4d.__version__=0.5.0`, exports all four gemm kernels + gdn lazy/fused +
  ar_3rank).
- **Deployed** to the LIVE mount dir `~/.cache/radiance-libr4d/v0.5.0-w4a16/r4d.so` (both instances mount
  this, NOT b9e42ab-rx10 as the compose grep suggested); backup `r4d.so.lean-backup`. Restarted vllm-0:
  **`libr4d 0.5.0, 27 kernels, 22/23 queries`; `gemm_nt M=64 bf16/w4a16/w4a8` now RESOLVE** (no fallback),
  and the `[radiance.gemm] no gemm_nt kernel` + `[radiance.w4] no w4a16` disabled-lines are gone.
- Throughput quick-check on v0 (rx11, MTP-only 5x1024): repeat 112.0 / agent 95.4 / novel 53.4 vs lean
  123.1/100.5/53.6 -- within the ~+-5% GPU-throttle noise; **no resolvable gain** but the fallback is gone.
- **vllm-1 must be restarted** to pick up rx11 (same mount). Revert = restore `v0.5.0-w4a16/r4d.so.lean-backup`.

**F (parked): cuteDSL/CUTLASS.** `ll_bf16.is_available()` imports `cutlass` + `cutlass.cute`, and the tuned
configs are SM100f (NVIDIA Blackwell). **Not applicable on gfx1201/ROCm** -- parked.

**G (config-gated): fused CUDA GDN decode.** `_fused_gdn_decode_unsupported_reason` requires
`recurrent_state_dtype in FUSED_GDN_STATE_DTYPES = (float32, bfloat16)`; we pass
`--mamba-ssm-cache-dtype float16` -> blocked, so `gdn_decode_kernel` resolves to `triton`. Enabling needs
`mamba_ssm_cache_dtype` -> bf16 or fp32 (conv cache is already bf16). Interacts with GDN-lazy (fp16 state)
and doubles SSM-state memory for fp32. A/B needed (accuracy+throughput).

**H (investigate): mamba dtype mismatch.** Model config says `mamba_ssm_dtype='float32'`; we override the
SSM state to `float16` (kept for memory / GDN-lazy / possible stochastic-rounding). fp16 is NOT in
FUSED_GDN_STATE_DTYPES (blocks G) and is the least accurate state dtype. No stochastic-rounding env is set
(currently), so fp32/bf16 is switchable. A/B `fp16 vs bf16 vs fp32` (acceptance/quality + tok/s) needed.

**TODO:** bake the overlay so this can't recur -- either clear the injected tail at boot before patching,
or make `edit()` replace an existing tail instead of skipping.

**Positive-case validation (cont.81, new code live):** a large repetitive/structured generation
(60 near-identical Python functions) on vllm-0 gave **mean 78.1%, up to 90-100%** sustained (22/26
windows >=70%) -- the n-gram echo tails are accepted. Across the whole boot (varied flows) **42 windows,
mean 72.1%, 0 windows <15%**, 0 matcher errors. `[ngram] rows=500 extended_rows=301` (dominated by the
repetitive gen) and `[ngram-hist]` mlen32 only 16/500 (vs the old 1365/2000 boilerplate saturation), so
the gates now filter rather than fire on every long match.

## 2026-10-01 (cont. 82) -- same-prompt A/B (NGRAM=0 vs 1) + spec-decode optimality research

**A/B harness:** `aijuus/ngram_ab_probe.py` -- fixed prompts {repeat=60-function Python file,
novel=1200-word story, agent=40-entry JSON tool-schema continuation}, REPS=5, max_tokens=1024,
temperature 0.7, against vllm-0 (blend) directly, registry-toggle + restart + warm re-run.

| prompt | NGRAM=1 (warm) | NGRAM=0 | delta |
|---|---|---|---|
| repeat | 115.2 tok/s | 123.1 | n-gram **-6.4%** |
| novel | 54.2 | 53.6 | +1.1% |
| agent | 103.4 | 100.5 | +2.9% |

**Verdict: within noise -- the gated n-gram is a WASH vs MTP-only on this workload.** Confounded by
both GPUs reporting `THROTTLE_STATUS: THROTTLED` (~+-5% run variance; one cold boot gave an anomalous
52 tok/s that recovered to 115 on re-run). So: safe (no collapses) but **not demonstrably beneficial**;
the earlier 78-100% "win" on repetitive content was MTP itself, not the n-gram.

**Optimality research (subagent):** our design is the right family (hybrid neural + prompt-lookup) but
**not globally optimal**. Our F1 slot-0-agreement gate makes it too conservative to rescue MTP misses;
the principled published form is TensorRT-LLM **SA+MTP** (`sa_spec_threshold`) and the Arctic Suffix
Decoding paper's **hybrid tau gate** (suffix tree first, fall back to neural when `SCORE <= tau`), with
frequency x prefix-length scoring, adaptive `MAX_SPEC`, and a **cross-request** suffix cache -- we only
have a single-suffix top-1 frequency proxy (F6). Ranked: (1) drive the tail from the real Arctic
`SuffixDecodingCache` (overlay + `pip install arctic-inference`; expected +10-30% accepted length on
repetitive agentic segments, neutral on prose); (2) keep ours + tune (marginal); (3) **DFlash** (big win
on MI355X, but image rebuild + checkpoint + `TRITON_ATTN` and the ROCm concurrency bug -- high risk);
(4) EAGLE3 tree (no head for this arch -> training project); native `method: suffix`/`ngram` **replaces**
MTP and would regress the general fraction. Revert stays `RADIANCE_DRAFT_NGRAM=0`.

## 2026-10-01 (cont. 90) -- rx12: bf16 lazy-state kernel + v0.5.0 rebase; fresh-deploy path wired

**Why.** A bf16 `--mamba-ssm-cache-dtype` state could not run with `RADIANCE_GDN_LAZY=1`:
`radiance_gdn_lazy._Tables` maps only `{fp16: f16state, fp32: fp32state}` and looks up an exported
`r4d.gdn_lazy_materialize_k128_v128_bf16_<tag>`, and rx10/rx11 shipped no `_bf16state` sibling for the
lazy family (recurrent + fused already had one). Python cannot build a kernel, so this was rebuild-class
(libr4d rebuilds are permitted).

**Kernel change (rx12).** The lazy kernels are already generic over `STDT` (`lz_launch<STDT>` /
`lz_materialize_launch<STDT>`, `r4d_gdn_state.h` already implements `R4D_ST_BF16` load/store), so the
addition is a thin TU + plumbing:
- `r4d_gdn_lazy_update_k128_v128_bf16_bf16state.hip` (new): calls `lz_launch<R4D_ST_BF16>` /
  `lz_materialize_launch<R4D_ST_BF16>`.
- `r4d.h`: declare both symbols; `r4d_module.hip`: two `m.def`.
- `r4d_registry.hip`: `cGdnLazyUpdateBf16St` (`state_dtype=bf16`) + the matching `ROW`
  (**mandatory** -- `select()` matches on `state_dtype`, else it can hand back the wrong-width kernel).
- `build.sh`: one `UNITS` entry (the v0.5.0 link line is auto-generated from UNITS).
- `radiance_gdn_lazy.py`: `torch.bfloat16: "bf16state"` in the tag map (runtime-copied from /patches).

**Packaging.** `r4d_radiance_extras_rx12.patch` = rx10 extras **rebased onto libr4d `v0.5.0`** (folds in
the 3 anchors rx10 needed merged by hand, and brings the v0.5.0-only `gemm_*_m64` + `quant_act_i8` +
`dflash_conv` units) **+** the bf16 lazy TU. 22 files; `git apply --check` clean against a fresh v0.5.0.
Built `GFX_ARCH=gfx1201` in `juupp/vllm-radiance:0.9.3-collect-tokens` (2,712,064 B, sha
`01d4f90b…`). Verified: `kernels 28`, `select("gdn_lazy_update", state_dtype="bf16", head_k=128,
head_v=128) -> gdn_lazy_update_k128_v128_bf16_bf16state`, both `_bf16state` symbols exported,
`gemm_bf16_nt_m64` / `gemm_w4a16_nt_m64` present.

**Deployed.** Live mount `~/.cache/radiance-libr4d/v0.5.0-w4a16/r4d.so` (both instances) overwritten with
rx12; prior rx11 saved as `r4d.so.rx11-backup`. Cache copy at `~/.cache/radiance-libr4d/v0.5.0-rx12/r4d.so`.
Revert = restore the `.rx11-backup`.

**Fresh-deploy wiring.** `serve-mxfp4.sh` now selects `$R4D_PIN_RX12-rx12` (source pin `v0.5.0`, default
`R4D_PIN_RX12=v0.5.0`) for `RADIANCE_GDN_LAZY=1`, via a new `R4D_SRC_PIN` so the checkout pin follows the
key; rx9/rx10 paths are unchanged. `coolify-compose-2gpu.yml`: both vllm services' `R4D_SO` + `/r4d` mount
moved `b9e42ab-rx10` -> `v0.5.0-rx12`. README `R4D_PIN` row notes the lazy override.

**Caveat (design, not a bug).** `r4d_gdn_state.h` argues fp16 still beats bf16 for this state (10 vs 7
mantissa bits; bf16's extra exponent range is unreachable for the O(1e-2) leaky integrator). rx12 makes
bf16 *possible*, not preferable -- the M10 fp16-vs-bf16 A/B via `bench-eval.py` is the decider.

**Pending.** Engine restart to load rx12 (user-owned). Then: confirm boot log shows the bf16 materialize
fn under `MAMBA_SSM_DTYPE=bfloat16` + `RADIANCE_GDN_LAZY=1`, and run the M10 A/B.

## 2026-10-02 (cont. 91) -- MTP acceptance gate (low-acceptance bail-out) + acceptance discriminator

**Diagnosis (live).** vllm-0 was pinned on one 20k-context request drafting 5/step and accepting
~0.3-1.9 tok/s (p0 0.03-0.26, p1-p4 ~0, gen 10 tok/s), while vllm-1 was healthy (p0 0.70-0.88).
Root cause of the throughput loss: **no acceptance feedback at small batch**. `patch_dynwidth`'s
per-request width cap is gated to `running >= RADIANCE_DYNW_MIN_BATCH` (3) and floored at 2, and the
V2 runner's draft depth K comes only from `num_speculative_tokens_per_batch_size` (bs 1-2 -> 5). So a
lone low-acceptance stream pays the full K=5 draft cost forever. Also confirmed the width cap is
largely inert under sync scheduling: the runner sets K from `speculator._radiance_dyn_k` and slices
`draft_tokens[:, :_rd_k2]`; it never forwards `SchedulerOutput.num_spec_tokens_to_schedule`.

**Discriminator (one-shot, `aijuus/acc_gate_check.py`).** Same drafter, three regimes on vllm-1:
- greedy/predictable (counting): 137 tok/s, accepted/update 4.90, per-pos 1.00 1.00 0.99 0.97 0.94
- greedy/high-entropy (random words): 61.7 tok/s, accepted/update 1.61, 0.99 0.35 0.17 0.07 0.03
- sampled/high-entropy (temp 0.9): 67.8 tok/s, accepted/update 1.90, 0.90 0.52 0.26 0.14 0.07
=> **drafter is healthy** (p0 ~1.0 on predictable text); the collapse is HIGH-ENTROPY output, not a
draft/state bug. So an acceptance-gated depth cut is the correct fix, not a correctness investigation.

**Fix (`patch_dynamic_depth.py`).** Added an acceptance gate inside the speculator's `propose`, which
already receives `num_sampled`/`num_rejected`: when `RADIANCE_ACC_GATE=1` and `num_reqs <=
RADIANCE_ACC_GATE_BATCH` (default 2), track an EMA of `max_i(num_sampled_i - 1)` (accepted drafts,
max-across-requests so a good co-scheduled request is never dragged down; accept-full-width observes
+1 so the EMA can climb out) and shrink `_rd_k` to `max(1, ceil(ema)+RADIANCE_ACC_GATE_MARGIN)`
(margin default 1). Sets both `_radiance_eff_k` (loop bound) and `_radiance_dyn_k` (runner slice).
Above the batch threshold the size-only schedule is untouched (weight-stream flat zone). Defaults
off; `RADIANCE_ACC_GATE_DIAG=1` logs `[acc-gate] bs accmax ema k`. `RADIANCE_ACC_GATE_BATCH=2`
because bs=1-2 is per-request/effectively so; bs>=3 stays batch-size-only (batch-K coupling would
hurt mixed batches). Applied-tested in a pristine container: patches + `ast.parse` clean.

**Wiring.** `aijuus/model-registry.json`: `RADIANCE_ACC_GATE=1`, `RADIANCE_ACC_GATE_BATCH=2` added to
both MTP server_env blocks (`mtp-27B-MXFP4-blend`, `mtp-27B-MXFP4-Thinkingcap`).

**Status.** NOT live yet: the user's 08:03 restart (Thinkingcap switch) booted before these edits
(installed speculator grep `RADIANCE_ACC_GATE` = 0; PID1 env has DYNAMIC_DEPTH but not ACC_GATE).
Takes effect on the next restart. Verify: boot log `[dyn-depth] applied`, then `RADIANCE_ACC_GATE_DIAG`
`[acc-gate]` lines should show `k` dropping on low-`accmax` lone streams, and high-entropy bs=1
throughput rising toward the ~60+ tok/s regime seen in the discriminator.

## 2026-10-02 (cont. 92) -- acceptance gate MEASURED, refuted, disabled

**Live-apply.** The overlay patches are marker-idempotent and `docker start` reuses the container
filesystem, so the new gate never applied on restart (`[dyn-depth] speculator.py already applied`).
Applied the gate delta directly to both containers' installed speculator via
`aijuus/apply_acc_gate_live.py` (+6 refs each); a changed patch needs a container RECREATE for the
entrypoint path to pick it up. Gate confirmed live: `drafts/update` fell 5.00 -> 2.84/3.34 on vllm-1.

**Measured on vllm-1 (blend, gate on vs the gate-off baseline earlier same instance):**
| regime | baseline tok/s | gate tok/s | baseline drafts/upd | gate drafts/upd | baseline acc/upd | gate acc/upd |
|---|---|---|---|---|---|---|
| greedy/predictable | 137.2 | 133.2 | 5.00 | 4.97 | 4.90 | 4.92 |
| greedy/high-entropy | 61.7 | **65.3 (+6%)** | 5.00 | 3.34 | 1.61 | 1.61 |
| sampled/high-entropy | 67.8 | **56.8 (-16%)** | 5.00 | 2.84 | 1.90 | 1.23 |

**Verdict: net-negative on the live default (sampled, temp 0.7/top_p 0.95/top_k 20); disabled.**
Step time fell only 42.7 -> 39.3 ms (-8%) while tokens/step fell 2.90 -> 2.23 (-23%). The draft loop
is only ~8% of the step (each draft forward ~1.6 ms = 849 MB bf16 MTP block; K=5 ~= 8 ms of ~43 ms);
the target weight stream (~24 ms) dominates. Marginal drafts have POSITIVE expected value under
sampled/stochastic acceptance, so trimming K loses more accepted tokens than the time it saves.
Greedy/zero-acceptance streams are the exception (dead drafts removed -> +6%).

**Corrected premise.** The earlier "lone low-acceptance stream = 4x loss from the draft loop" was
wrong; the gate can save at most ~8-15% even at zero acceptance, and it HURTS sampled throughput. The
vllm-0 10 tok/s was thermal (97 C junction, sclk 3364 MHz) + 17k context, not the draft loop. Registry
set `RADIANCE_ACC_GATE=0` on both MTP models; the code stays as an off-by-default option
(`apply_acc_gate_live.py` / `patch_dynamic_depth.py`), margin/EMA knobs documented. If ever re-enabled
for strict-zero streams, use a much higher margin (e.g. ceil(ema)+3) and gate on ema < ~1.

**Takeaway for MTP optimization.** Draft depth is a sub-10% lever. Future throughput work should
target the TARGET forward (decode GEMM split-K/BK at small M, attention, GDN) or reducing verify
rows -- not K. GPU0 thermal (LACT undervolt/clock cap <=3.3 GHz) is the real vllm-0 problem.
