# AIJUUS work log

A running, dated log of what was changed, why, and how it was verified. Newest entry first.
Complements (does not replace) `TCCLA-VLLM-MTP-RESEARCH.md` and
`TCCLA-VLLM-MTP-IMPLEMENTATION-PLAN.md`, which hold the analysis and the plan.

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
