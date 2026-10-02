"""Lazy GDN state snapshots: the align-mode copy hooks.

vLLM's align mode keeps ONE running state per request in the mamba cache and, under speculative
decode, one extra block per draft token so every candidate can snapshot its state (2 + SPEC pages
per request per mamba layer group). RADIANCE_GDN_LAZY=1 keeps one extra block instead: a stash of
the last step's candidate inputs, replayed on the next step by libr4d's gdn_lazy_update kernel
(see r4d_gdn_lazy_update_k128_v128.h). Per-request cost drops from 9 pages to 3 at SPEC=7.

What vLLM still expects from the cache are its two temporal-state copies (vllm/v1/worker/
mamba_utils.py): before a forward, a request whose running block moved across a mamba block
boundary has the accepted candidate's state copied into the new block; after a step, a request
whose accepted tokens crossed a boundary has the boundary candidate's state copied into the
aligned checkpoint block. Neither candidate state exists in a lazy cache, so patch_gdn_lazy.py
makes the fused Triton copies skip temporal states and this module supplies them as
"base + replay to that candidate" through gdn_lazy_materialize -- one launch for all 48 layers,
driven by the same GPU-resident decision inputs, no host sync. The conv state is unchanged
(its rolling per-candidate columns live in the running block) and keeps vLLM's copies.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_GDN_LAZY", "0") == "1"
STASH_COL = 1                    # window column of the stash block (0 = running / base)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:                # pragma: no cover
    _HAVE_TRITON = False


def _log(msg):
    sys.stderr.write(f"[radiance.gdn.lazy] {msg}\n")
    sys.stderr.flush()


class _Tables:
    """Device pointer tables for the materialize kernel, in the ctx's state order."""

    def __init__(self, ctx, kv_cache_config, forward_context):
        import radiance_gdn
        from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec
        from vllm.v1.worker.mamba_utils import _get_mamba_spec_for_layer
        ptrs, strides, groups, alogs, dtbs = [], [], [], [], []
        self.keep = []                       # the fp32 gate copies must outlive the tables
        geom = None
        for g_local, gid in enumerate(ctx.mamba_group_ids):
            kv_cache_group = kv_cache_config.kv_cache_groups[gid]
            for name in kv_cache_group.layer_names:
                layer = forward_context[name]
                # copy funcs are per MambaSpec type (MambaStateCopyFuncsByType), aligned
                # positionally with layer.kv_cache (mamba_utils pairs them the same way).
                mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, name)
                copy_funcs = tuple(ctx._radiance_copy_funcs[mamba_spec.mamba_type])
                a_log, dt_bias = radiance_gdn._gate_params(layer)
                self.keep.append((a_log, dt_bias))
                for st_idx, (cf, state) in enumerate(zip(copy_funcs, layer.kv_cache)):
                    if cf is not get_temporal_copy_spec:
                        continue
                    ptrs.append(state.data_ptr())
                    strides.append(state.stride(0) * state.element_size())
                    groups.append(g_local)
                    alogs.append(a_log.data_ptr()); dtbs.append(dt_bias.data_ptr())
                    g = (layer.num_v_heads // layer.tp_size, layer.num_k_heads // layer.tp_size,
                         layer.head_k_dim, layer.head_v_dim, state.stride(1), state.dtype)
                    if geom is None:
                        geom = g
                    elif g != geom:
                        raise RuntimeError(f"lazy GDN: layer {name} geometry {g} != {geom}")
        dev = torch.device("cuda")
        self.state_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=dev)
        self.slot_strides = torch.tensor(strides, dtype=torch.int64, device=dev)
        self.group_idx = torch.tensor(groups, dtype=torch.int32, device=dev)
        self.alog_ptrs = torch.tensor(alogs, dtype=torch.int64, device=dev)
        self.dtb_ptrs = torch.tensor(dtbs, dtype=torch.int64, device=dev)
        self.n = len(ptrs)
        self.H, self.Hg, self.K, self.V, self.st_head, self.dtype = geom
        import r4d
        tag = {torch.float16: "f16state", torch.bfloat16: "bf16state",
               torch.float32: "fp32state"}.get(self.dtype)
        if tag is None:
            raise RuntimeError(f"lazy GDN: no materialize kernel for a {self.dtype} state cache")
        self.fn = getattr(r4d, f"gdn_lazy_materialize_k128_v128_bf16_{tag}")
        _log(f"materialize tables: {self.n} temporal states, H {self.H} Hg {self.Hg} "
             f"st_head {self.st_head} state {self.dtype}")


def materialize(ctx, mode, num_reqs, a, b, c, idx_mapping):
    """mode 0 (pre-forward): a=state_idx (dst col), b=src_col, c=token_bias.
       mode 1 (post-step):   a=num_accepted, b=state_idx (running col), c=new_num_computed."""
    tb = getattr(ctx, "_radiance_lazy_tables", None)
    if tb is None:
        tb = _Tables(ctx, ctx._radiance_kv_cfg, ctx._radiance_fwd_ctx)
        ctx._radiance_lazy_tables = tb
    im = idx_mapping.data_ptr() if idx_mapping is not None else 0
    if mode == 0:
        args = (a.data_ptr(), b.data_ptr(), c.data_ptr(), 0, 0)
    else:
        args = (b.data_ptr(), 0, 0, a.data_ptr(), c.data_ptr())
    tb.fn(tb.state_ptrs.data_ptr(), tb.slot_strides.data_ptr(), tb.group_idx.data_ptr(),
          tb.alog_ptrs.data_ptr(), tb.dtb_ptrs.data_ptr(), ctx.block_table_ptrs.data_ptr(),
          int(ctx.block_table_stride_req), im, int(mode), *args, int(ctx.block_size),
          int(num_reqs), tb.n, tb.H, tb.Hg, tb.K, tb.V, int(tb.st_head), tb.K ** -0.5, 20.0,
          torch.cuda.current_stream().cuda_stream)


if _HAVE_TRITON:
    @triton.jit
    def _lz_invalidate_kernel(
        state_ptrs, slot_strides, bt_ptr, bt_stride,
        state_idx, idx_mapping, mask, num_reqs, H, st_head,
        HBLK: tl.constexpr, HAS_IM: tl.constexpr,
    ):
        b = tl.program_id(0)
        si = tl.program_id(1)
        if b >= num_reqs:
            return
        if tl.load(mask + b) == 0:
            return
        if HAS_IM:
            req = tl.load(idx_mapping + b)
        else:
            req = b
        if req < 0:
            return
        col = tl.load(state_idx + req)
        stash_slot = tl.load(bt_ptr + req * bt_stride + col + 1)
        if stash_slot <= 0:
            return
        sbase = tl.load(state_ptrs + si)
        sstr = tl.load(slot_strides + si)
        base = sbase + stash_slot.to(tl.int64) * sstr
        offs = tl.arange(0, HBLK)
        m = offs < H
        addrs = (base + offs.to(tl.int64) * st_head).to(tl.pointer_type(tl.uint32))
        tl.store(addrs, tl.zeros((HBLK,), dtype=tl.uint32), mask=m)


def invalidate(ctx, num_reqs, state_idx, idx_mapping, mask, block_table):
    """radiance lazy gdn: zero the stash header of every masked (prefilling) row, so a stash written
    against a physical base_slot in an EARLIER context (a reused prefix-cache block) can never be
    replayed. Uses the request's post-advance window column (state_idx + 1). No-op unless enabled."""
    if not ENABLED or not _HAVE_TRITON or num_reqs == 0:
        return
    tb = getattr(ctx, "_radiance_lazy_tables", None)
    if tb is None:
        try:
            tb = _Tables(ctx, ctx._radiance_kv_cfg, ctx._radiance_fwd_ctx)
            ctx._radiance_lazy_tables = tb
        except Exception as e:                       # tables not ready yet: skip, not fatal
            _log(f"invalidate skipped: {e}")
            return
    # NOTE: triton needs the *tensor* so it sees a pointer; a raw .data_ptr() int is a scalar.
    im = idx_mapping if idx_mapping is not None else 0
    hblk = max(1, triton.next_power_of_2(int(tb.H)))
    # state_ptrs are raw byte addresses here, so every offset we add must be in bytes.
    st_head_bytes = int(tb.st_head) * int(tb.dtype.itemsize)
    _lz_invalidate_kernel[(int(num_reqs), int(tb.n))](
        tb.state_ptrs, tb.slot_strides, block_table, int(ctx.block_table_stride_req),
        state_idx, im, mask, int(num_reqs), int(tb.H), st_head_bytes,
        HBLK=hblk, HAS_IM=idx_mapping is not None,
    )
