#!/usr/bin/env python3
"""Deep trace for the KV-offload <-> speculative-decoding interaction (diagnostic).

Complements patch_offload_debug.py with the data needed to pin the SSM/Mamba
group store-vs-load (mis)alignment:

  * TRACE GEOM      -- the exact per-group offload geometry resolved by
                       SchedulerOffloadConfig.from_spec (tokens_per_chunk,
                       sliding_window_size_in_chunks, alignment_chunk_count,
                       is_eagle_group, blocks_per_chunk).
  * TRACE STORE_REACH -- every is_store_reachable_swa_chunk() decision, with the
                       absolute chunk index and the alignment segment it lands
                       in. Summarized after the first N calls.
  * TRACE SWLOOKUP  -- the tail hit/miss pattern of every _sliding_window_lookup()
                       call: group index, slice length, required window, the
                       returned end index, and the block-hash tail of the keys so
                       it can be joined against what was stored.
  * TRACE K2LOAD    -- the per-group keys_to_load length computed in
                       update_state_after_alloc().

Append-only, idempotent, gated by RADIANCE_OFFLOAD_TRACE=1. Server log only;
no control-flow change. Remove once the alignment is understood.
"""
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
SCHED = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
SENTINEL = "# radiance-offload-trace"

BLOCK = '''

# radiance-offload-trace
def _radiance_offload_trace_install():
    import logging
    import os

    if os.environ.get("RADIANCE_OFFLOAD_TRACE", "0") in ("", "0"):
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        scheduler as S,
    )
    from vllm.v1.kv_offload.base import (
        get_offload_block_hash,
        get_offload_group_idx,
    )

    log = logging.getLogger("vllm.offloadtrace")
    log.setLevel(logging.INFO)

    state = {"reach": 0, "sw": 0, "lookup": 0, "k2load": 0}
    reach_agg = {}

    # ---- geometry -----------------------------------------------------------
    _from_spec = S.SchedulerOffloadConfig.from_spec.__func__

    def from_spec(cls, spec, vllm_config, kv_cache_config):
        cfg = _from_spec(cls, spec, vllm_config, kv_cache_config)
        try:
            log.info("TRACE GEOM blocks_per_chunk=%d num_workers=%d prompt_only=%s",
                     cfg.blocks_per_chunk, cfg.num_workers, cfg.offload_prompt_only)
            for gc in cfg.kv_group_configs:
                log.info(
                    "TRACE GEOM group=%d tpb=%d tpc=%d hashes_per_chunk=%d "
                    "sw_in_chunks=%s align_chunks=%s eagle=%s",
                    gc.group_idx, gc.tokens_per_block, gc.tokens_per_chunk,
                    gc.hashes_per_chunk, gc.sliding_window_size_in_chunks,
                    gc.alignment_chunk_count, gc.is_eagle_group,
                )
        except Exception as exc:  # noqa: BLE001
            log.info("TRACE GEOM failed: %r", exc)
        return cfg

    S.SchedulerOffloadConfig.from_spec = classmethod(from_spec)

    # ---- store reachability --------------------------------------------------
    _reach = S.is_store_reachable_swa_chunk

    def reach(abs_idx, storable, align, sw, eagle):
        r = _reach(abs_idx, storable, align, sw, eagle)
        key = (align, sw, eagle)
        agg = reach_agg.setdefault(
            key, {"n": 0, "kept": 0, "drop": 0, "last_kept": [], "last_drop": []}
        )
        agg["n"] += 1
        agg["kept" if r else "drop"] += 1
        bucket = agg["last_kept" if r else "last_drop"]
        bucket.append((abs_idx, storable))
        del bucket[:-4]
        state["reach"] += 1
        if state["reach"] <= 60 or state["reach"] % 400 == 0:
            pos = None if align is None else abs_idx % align
            seg = None if align is None else abs_idx - pos
            actual = None if align is None else min(align, storable - seg)
            log.info(
                "TRACE STORE_REACH abs=%d storable=%d align=%s sw=%s eagle=%s "
                "pos_in_seg=%s seg_start=%s seg_len=%s -> %s",
                abs_idx, storable, align, sw, eagle, pos, seg, actual, r,
            )
        if state["reach"] % 400 == 0:
            log.info("TRACE STORE_REACH summary=%s", reach_agg)
        return r

    S.is_store_reachable_swa_chunk = reach

    # ---- sliding-window lookup pattern --------------------------------------
    _sw = S.OffloadingConnectorScheduler._sliding_window_lookup

    def sw_lookup(self, keys, window, ctx):
        keys = list(keys)
        orig_lookup = self.manager.lookup
        pattern = []

        def spy(k, c):
            r = orig_lookup(k, c)
            pattern.append((get_offload_block_hash(k)[:6].hex(), r))
            return r

        def _hit_idx_list():
            return [
                i
                for i, (_, r) in enumerate(pattern)
                if getattr(r, "name", str(r)).startswith("HIT")
            ]

        try:
            self.manager.lookup = spy
        except Exception:  # noqa: BLE001
            spy = None
        try:
            out = _sw(self, keys, window, ctx)
        finally:
            if spy is not None:
                try:
                    self.manager.lookup = orig_lookup
                except Exception:  # noqa: BLE001
                    pass
        state["sw"] += 1
        if state["sw"] <= 24 or state["sw"] % 200 == 0:
            group = get_offload_group_idx(keys[0]) if keys else -1
            flags = "".join(
                "H" if getattr(r, "name", str(r)).startswith("HIT") else "M"
                for _, r in pattern
            )
            log.info(
                "TRACE SWLOOKUP #%d req=%s group=%s nkeys=%d window=%d out=%s "
                "hits=%d misses=%d flags(tail48)=%s hit_idx=%s tail=%s",
                state["sw"], ctx.req_id, group, len(keys), window, out,
                flags.count("H"), flags.count("M"), flags[-48:],
                _hit_idx_list()[:60],
                [(h, getattr(r, "name", "?")) for h, r in pattern[-4:]],
            )
        return out

    S.OffloadingConnectorScheduler._sliding_window_lookup = sw_lookup

    # ---- mamba store: which chunk hashes are actually offloaded -------------
    _rad_init = S.OffloadingConnectorScheduler.__init__
    mstore_n = {"n": 0}

    def _init(self, spec, vllm_config, kv_cache_config):
        _rad_init(self, spec, vllm_config, kv_cache_config)
        try:
            mamba_groups = {
                gc.group_idx
                for gc in self.config.kv_group_configs
                if gc.sliding_window_size_in_chunks == 1
            }
            orig_prepare_store = self.manager.prepare_store

            def prepare_store(keys, ctx):
                out = orig_prepare_store(keys, ctx)
                mstore_n["n"] += 1
                if mstore_n["n"] <= 80:
                    per = {}
                    for k in keys:
                        g = get_offload_group_idx(k)
                        if g in mamba_groups:
                            per.setdefault(g, []).append(
                                get_offload_block_hash(k)[:6].hex()
                            )
                    if per:
                        log.info(
                            "TRACE MSTORE #%d req=%s mamba_hashes=%s",
                            mstore_n["n"], ctx.req_id, per,
                        )
                return out

            self.manager.prepare_store = prepare_store
        except Exception as exc:  # noqa: BLE001
            log.info("TRACE MSTORE install failed: %r", exc)

    S.OffloadingConnectorScheduler.__init__ = _init

    # ---- final lookup result -------------------------------------------------
    _lookup = S.OffloadingConnectorScheduler._lookup

    def lookup(self, req_status):
        out = _lookup(self, req_status)
        state["lookup"] += 1
        if state["lookup"] <= 24 or state["lookup"] % 200 == 0:
            log.info(
                "TRACE LOOKUP #%d req=%s computed=%d prompt=%d num_tokens=%d -> %s",
                state["lookup"], req_status.req.request_id,
                req_status.num_locally_computed_tokens,
                req_status.req.num_prompt_tokens, req_status.req.num_tokens, out,
            )
            if state["lookup"] <= 6:
                log.info(
                    "TRACE BHASH req=%s n=%d hashes=%s",
                    req_status.req.request_id,
                    len(req_status.req.block_hashes),
                    [h[:6].hex() for h in req_status.req.block_hashes],
                )
        return out

    S.OffloadingConnectorScheduler._lookup = lookup

    # ---- per-group keys_to_load ---------------------------------------------
    _usa = S.OffloadingConnectorScheduler.update_state_after_alloc

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        orig_lookup = self.manager.lookup  # noqa: F841
        try:
            group_offload_keys = [
                gs.offload_keys for gs in self._req_status[
                    request.request_id
                ].group_states
            ]
        except Exception:  # noqa: BLE001
            group_offload_keys = None
        out = _usa(self, request, blocks, num_external_tokens)
        state["k2load"] += 1
        if state["k2load"] <= 24 or state["k2load"] % 200 == 0:
            try:
                rs = self._req_status[request.request_id]
                n_cached = rs.num_locally_computed_tokens + num_external_tokens
                det = []
                for gc, gs in zip(self.config.kv_group_configs, rs.group_states):
                    import math as _m
                    num_chunks = _m.ceil(n_cached / gc.tokens_per_chunk)
                    det.append((gc.group_idx, len(gs.offload_keys), num_chunks))
                log.info(
                    "TRACE K2LOAD #%d req=%s ext=%d computed=%d (group, keys, chunks)=%s",
                    state["k2load"], request.request_id, num_external_tokens,
                    rs.num_locally_computed_tokens, det,
                )
            except Exception as exc:  # noqa: BLE001
                log.info("TRACE K2LOAD detail failed: %r", exc)
        return out

    S.OffloadingConnectorScheduler.update_state_after_alloc = update_state_after_alloc

    log.info("TRACE installed: offload trace active")


_radiance_offload_trace_install()
'''


def main() -> None:
    text = SCHED.read_text()
    if SENTINEL in text:
        print(f"[patch] offload-trace: already applied ({SCHED.name})")
        return
    SCHED.write_text(text + BLOCK)
    print(f"[patch] offload-trace: applied to {SCHED}")


main()
