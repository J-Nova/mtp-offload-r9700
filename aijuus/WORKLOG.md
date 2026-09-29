# AIJUUS work log

A running, dated log of what was changed, why, and how it was verified. Newest entry first.
Complements (does not replace) `TCCLA-VLLM-MTP-RESEARCH.md` and
`TCCLA-VLLM-MTP-IMPLEMENTATION-PLAN.md`, which hold the analysis and the plan.

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
