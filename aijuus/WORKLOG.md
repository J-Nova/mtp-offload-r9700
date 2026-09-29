# AIJUUS work log

A running, dated log of what was changed, why, and how it was verified. Newest entry first.
Complements (does not replace) `TCCLA-VLLM-MTP-RESEARCH.md` and
`TCCLA-VLLM-MTP-IMPLEMENTATION-PLAN.md`, which hold the analysis and the plan.

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
