# Reorder Batch Threshold Fix Plan

## Background

Upstream vLLM issue [#55894](https://github.com/vllm-project/vllm/issues/55894) documents a bug
where `GPUModelRunner.calculate_reorder_batch_threshold` takes the minimum across all attention
groups' `reorder_batch_threshold` values. When FlashInfer (threshold=1) is mixed with GDN/Mamba
builders (threshold=1+k), the global threshold drops to 1, causing draft-decode rows to be
misclassified as prefills → state slot corruption → garbage output.

The upstream fix is [PR #55898](https://github.com/vllm-project/vllm/pull/55898), open since
Sep 8, 2026 (22 days as of Sep 30), awaiting review from 10 code owners with no approvals yet.

## Current Deployment Safety

The current deployment is safe by configuration:
- `serve-mxfp4.sh:784` pins target attention to `ATTN=R4D`
- `serve-mxfp4.sh:438` pins dflash drafter to `TRITON_ATTN`
- `serve-mxfp4.sh:831` pins MTP drafter to `ATTN=R4D`
- R4D (`radiance_r4d_attn.py:146`) subclasses `TritonAttentionMetadataBuilder`; neither sets
  `reorder_batch_threshold`, inheriting `None` from `backend.py:592`
- GDN reports `1+SPEC` (5 or 8 depending on `RADIANCE_SPEC`)
- Global threshold stays at `1+SPEC` — no corruption under current config
- Production entrypoint (`aijuus/kv-offload/ops/entrypoint.sh:177,179`) hardcodes R4D/TRITON_ATTN

The risk is configuration drift: if the drafter backend changes to FlashInfer, or a new
threshold-1 backend is introduced, corruption could occur silently.

## Proposed Fix: Runtime Patch Script

Create `aijuus/kv-offload/patches/patch_reorder_threshold.py` — an idempotent Python script
applied at container boot (same pattern as `patch_gdn_metadata.py`, `patch_dynamic_depth.py`, etc.).

### Why runtime patch instead of build-time overlay patch

| Factor | Runtime patch | Build-time overlay patch |
|--------|---------------|--------------------------|
| Docker rebuild | Not needed | Required |
| Consistency | Matches existing pattern (`patch_gdn_metadata.py`, etc.) | Different from existing runtime patches |
| Removal when upstream merges | Delete script + restart | Revert overlay patch + rebuild |
| Iteration speed | Edit script + restart | Edit patch + rebuild + redeploy |

### Patch Contents

The patch mirrors PR #55898's changes:

1. **Add attribute to base class** (`vllm/v1/attention/backend.py`):
   - Add `requires_decode_ordering: bool = False` to `AttentionMetadataBuilder`

2. **Set True on hybrid attention builders**:
   - `GDNAttentionMetadataBuilder` (`vllm/v1/attention/backends/gdn_attn.py`)
   - `BaseMambaAttentionMetadataBuilder` (`vllm/v1/attention/backends/mamba_attn.py`)
   - `LinearAttentionMetadataBuilder` (`vllm/v1/attention/backends/linear_attn.py`)

3. **Modify threshold calculation** (`vllm/v1/worker/gpu_model_runner.py`):
   - Change `calculate_reorder_batch_threshold` from:
     ```python
     return min_none_high([g.reorder_batch_threshold for g in self._attn_group_iterator()])
     ```
   - To:
     ```python
     all_thresholds = [g.reorder_batch_threshold for g in self._attn_group_iterator()]
     required_thresholds = [
         g.reorder_batch_threshold
         for g in self._attn_group_iterator()
         if getattr(g, "requires_decode_ordering", False)
     ]
     if not required_thresholds:
         return min_none_high(all_thresholds)
     return max(min_none_high(all_thresholds), max(required_thresholds))
     ```
   - Add `logger.info` when the threshold is raised by the required set

### Idempotency

- Use a marker string (e.g., `# patch_reorder_threshold`) to detect prior application
- On re-run, detect marker and skip (print "already applied")

## Implementation Steps

1. **Create script**: Write `aijuus/kv-offload/patches/patch_reorder_threshold.py` following the
   pattern of `patch_gdn_metadata.py` (idempotent, marker-based, guarded with `ast.parse`)
2. **Wire into entrypoint**: Add invocation in `aijuus/kv-offload/ops/entrypoint.sh` alongside
   other patch scripts
3. **Verify clean apply**: Run against current vLLM source in a throwaway container; confirm
   `py_compile` passes and markers are present
4. **Test**: Run CPU regression tests to confirm no behavioral change under current config
5. **Deploy**: User restarts containers (Coolify redeploy or `/reload`)
6. **Monitor upstream**: Track PR #55898; when merged, remove the patch script and entrypoint
   invocation, then redeploy

## Rollback

- Set the patch to no-op or remove the entrypoint invocation
- Restart containers
- Standard aijuus revert workflow applies

## Risk Assessment

- **Low risk**: Under current config, the patch is a no-op (threshold already 5/8)
- **Conflict risk**: If upstream changes `backend.py` or `gpu_model_runner.py` before PR #55898
  merges, the patch may need updating; the idempotency marker prevents double-application
- **No behavioral change**: The patch only raises the threshold when a threshold-1 backend is
  present alongside a hybrid attention group; current config has no threshold-1 backend

## Status

- [x] Analysis complete (all claims verified against actual code)
- [x] Plan documented
- [ ] Script implementation (pending user approval)
- [ ] Testing
- [ ] Deployment
- [ ] Upstream PR monitoring
