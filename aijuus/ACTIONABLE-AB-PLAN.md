# Actionable items — A/B & validation plan

Status: planning. Every item here is **runtime-overlay / no rebuild** and testable on vllm-0.
Method: env knobs are flipped with `aijuus/ab.env` (sourced at container start), then
`docker kill && docker start` (env change ⇒ new AOT key ⇒ fresh compile; **restart warm until
`bench-conc --conc 1` shows c1 ≈ 68 t/s before measuring**). Auth `juup-123`; endpoint
`http://172.18.0.4:8000`. Recording: `aijuus/WORKLOG.md` + `OPEN-TASKS-INDEX.md`.

Tooling: `aijuus/bench-conc.py` (decode: agg t/s, ms/step proxy, acc/draft, per-position acceptance),
`aijuus/bench-prefill-ttft.py` (long-prompt TTFT), `aijuus/bench-quick.py`, `aijuus/hit_oracle.py`,
`aijuus/kv-offload/ops/turnbench.py --exact|--concurrent`, `aijuus/kv-offload/ops/tierbench.py`.

---

## A1 — E1 enablement (finalize) — **DECISION, evidence complete**

- **Goal:** decide whether to ship E1 on (offload suffix invalidation + eagle/MTP inclusion).
- **Knobs:** `RADIANCE_OFFLOAD_SUFFIX_INV=1`, `RADIANCE_OFFLOAD_EAGLE_INCLUDE=1`.
- **Evidence already in hand:** byte-identical single-turn; per-position acceptance unchanged;
  `turnbench --concurrent` gates-off 67.3% tier / 25.0% recompute → gates-on **71.0% tier / 24.0%
  recompute**, HEALTH PASS 21/21; `--exact` adds exactly **+880 external tokens/turn**. MT1/MT2 are
  inherent, so bit-exactness is not a gate.
- **What's left to do:** the only remaining step is a **deploy decision**. To ship: add the two vars to
  the model entry's `server_env` in `aijuus/model-registry.json` (or compose), restart both instances.
  To A/B once more on the live workload: set both in `ab.env`, warm-boot, `turnbench --concurrent`
  vs gates-off, compare tier%/recompute%/HEALTH.
- **Gate:** HEALTH PASS and no acceptance drop; tier ≥ +3pp. Risk: low (draft-group KV only; target
  re-verifies). **Recommend: enable** (or keep dormant if you want zero offload-contract change).

## A2 — n-gram enablement on repetitive/code workloads — **A/B (workload-specific)**

- **Goal:** does M1's now-safe n-gram help where it should (repetitive/code), even though it is
  net-negative on the generic bench mix?
- **Knobs:** `RADIANCE_DRAFT_NGRAM=1` (default 0).
- **Procedure:** baseline (0) warm → c1/c8 `bench-conc`; then `NGRAM=1` warm → c1/c8; **plus** a
  repetition-heavy probe (send the same ~2–4k-token code block repeated, or a `turnbench` session on
  the stdlib corpus) and compare acc/draft and tok/s. Watch the `[ngram] extended_rows/appended`
  counters (≈8% on the generic prompt; higher on repetitive).
- **Gate:** on a repetitive workload, NGRAM=1 must beat NGRAM=0 by more than the matcher overhead
  (currently ≈ −15% c8 on the generic mix). **Decision: enable per-workload only if it wins there.**

## A3 — V3: MTP dynamic depth + n-gram tail validation — **A/B (unblocked by M1)**

- **Goal:** validate the combination (dynamic depth `spec_schedule` 5/4 + n-gram tail) after the M1 fix.
- **Knobs:** `RADIANCE_DYNAMIC_DEPTH=1` (already default) ± `RADIANCE_DRAFT_NGRAM=1`.
- **Procedure:** warm A/B {depth, depth+ngram}; `bench-conc` c1/c4/c8 + acceptance; confirm no fault at
  bs≥2 with ngram on.
- **Gate:** depth-only is the reference; ngram adds only if acceptance/tok-per-step rises. Low priority
  (A2 likely decides the same question).

## A4 — VC2: EXACTSET + FUSED battery re-run — **A/B (confirmation)**

- **Goal:** confirm the prior "EXACTSET off / FUSED on" verdict on the current build.
- **Knobs:** `RADIANCE_DRAFT_EXACTSET=0|1`, `RADIANCE_DRAFT_FUSED=0|1` (registry currently sets
  EXACTSET=1 — worth verifying).
- **Procedure:** warm A/B of the 2×2 (or just EXACTSET 1 vs 0 with FUSED=1) on `bench-conc` c1/c8;
  acceptance + tok/s.
- **Gate:** keep the better cell; if EXACTSET=1 is neutral-or-better now, leave the registry as-is.

## A5 — P7: A1 fusion confirmation — **A/B (prefill)**

- **Goal:** does disabling the RMS/quant fusion lose prefill throughput (i.e. is the fusion earning
  its keep)?
- **Knobs:** `RADIANCE_FUSE_RMS_QUANT=0` + `RADIANCE_FP8_STREAM=0` vs on (default).
- **Procedure:** warm A/B (both vars) with `bench-prefill-ttft.py --prompt-tokens 16000+` (long prompt,
  unique salt) — the fusion only shows up with real prefill work.
- **Gate:** if fusion-off loses ≫2% prompt t/s, fusion stays; otherwise it's removable.

## A6 — P6: fold attention-output → o_proj quant — **CONDITIONAL on A5**

- Only worth it if A5 shows the fusion is *not* already capturing that headroom. Implementation is a
  runtime overlay in the attention epilogue; do A5 first.

## A7 — TF4: TunableOp GEMM sweep — **N/A for our stack**

- `CLAV_TUNABLEOP_SWEEP` is tcclaviger-fork-only; our compose uses a fixed `tunableop/tunableop-skinny0.csv`
  with `PYTORCH_TUNABLEOP_TUNING=0`. No env to sweep → mark N/A (would need a different table/recomp).

## A8 — TF6 / TF7: reduced-vocab + draft-head replicate — **need fork specifics**

- TF6 (tokens outside `draft_keep_file`) and TF7 (`CLAV_DRAFT_HEAD_REPLICATE` for 2-GPU DP) depend on
  tcclaviger-fork internals not shipped; no direct A/B without the private fork. Park pending TF5.

## A9 — OS3: E3 lazy-GDN fail-closed (route 3c) — **IMPLEMENT + validate (no A/B)**

- **Goal:** make `RADIANCE_GDN_LAZY=1` correct (currently fail-open → multi-turn corruption).
- **Design:** in `radiance_gdn_lazy.py` (runtime-copied), keep a bounded per-request temporal-state
  snapshot (2-slot GPU stage + pinned host ring keyed by token frontier); on a header mismatch restore
  from the ring instead of accepting the base. Pure Python, no rebuild.
- **Validate:** injected-stale-header unit test (old code → base; new → restored/raise) + multi-turn
  health (no repeat-loop/empty replies from ~turn 5) + `turnbench --concurrent` HEALTH PASS. Lazy is
  off today, so this is a **memory** win (~865→260 MB/req), not a perf one.

## A10 — PF6: 8k chunk decision — **profile, then decision**

- **Goal:** decide chunk 16384 vs 8192 (GDN occupancy vs deferral). PF1 showed prefill is
  chunk-independent at 8k; the question is long-context N>1 occupancy.
- **Procedure:** `bench-prefill-ttft.py` at 16k/32k and `bench-conc` c8 with `REG_CHUNK` 16384 vs 8192
  (each clear+warm). Compare prompt t/s, TTFT, deferrals.
- **Gate:** switch to 8192 only if it clearly wins on the long-context/concurrency profile.

## A11 — Dormant ports D1/D2 — **decision**

- **D1** `patch_mtp_conf_exit` (conf-exit/ragged) — A/B negative; keep dormant unless revisiting with a
  calibrated acceptance estimator at higher concurrency.
- **D2** `patch_mamba_scratch_zero` — perf-neutral correctness hardening; land if we want the defensive
  fix (reachability unproven). A/B is "no regression".

---

## Fastest ordering
1. **A1** enable E1 (evidence complete) — the only item with a measured gain and no cost.
2. **A4** EXACTSET/FUSED confirm (cheap; may reveal a registry misconfig).
3. **A2/A3** n-gram on a repetitive/code workload (decides the n-gram question).
4. **A5** fusion confirmation (cheap, prefill).
5. **A10** 8k decision; **A9** E3 implementation (larger).
6. A6 conditional; A7/A8 N/A/parked.
