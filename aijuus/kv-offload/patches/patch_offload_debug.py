#!/usr/bin/env python3
"""Temporary instrumentation for the KV-offload <-> speculative-decoding interaction.

WHY. With SPEC_METHOD=dflash the native OffloadingConnector allocates its CPU region and
stores tens of GB, yet vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} stays
0.0 -- nothing is ever promoted, so a prefix evicted from the GPU pool is recomputed from
scratch. With spec decoding off the same connector promotes and restores correctly. The only
spec-dependent branch in the offloading scheduler is the EAGLE/MTP group handling
(SchedulerOffloadConfig.from_spec): when speculative_config.use_eagle() is true and no
kv_cache_group carries is_eagle_group, it falls back to marking EVERY group as an eagle
group, which changes both storable_chunks() and the load path's per-group chunk pop.

HOW. Append an idempotent monkeypatch to two site-packages modules that counts store and
lookup traffic per KV group and logs the group configuration. This is a diagnostic only: it
changes no control flow and is installed only when serve-mxfp4.sh is run with
RADIANCE_OFFLOAD_DEBUG=1.

Remove once the interaction is understood.
"""
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
MANAGER = SP / "vllm/v1/kv_offload/cpu/manager.py"
SCHEDULER = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
SENTINEL = "# radiance-offload-debug"

MANAGER_BLOCK = '''

# radiance-offload-debug
def _rad_offload_debug_install_manager():
    import logging
    log = logging.getLogger("vllm.offloaddbg")

    from vllm.v1.kv_offload.base import (
        get_offload_block_hash,
        get_offload_group_idx,
    )

    totals = {"hit": 0, "miss": 0, "pending": 0, "other": 0}
    groups = {}
    calls = {"store": 0, "store_keys": 0, "filtered": 0}

    _lookup = CPUOffloadingManager.lookup

    def lookup(self, key, req_context):
        res = _lookup(self, key, req_context)
        name = str(res).rsplit(".", 1)[-1].lower()
        if name not in totals:
            name = "other"
        totals[name] += 1
        gi = get_offload_group_idx(key)
        g = groups.setdefault(gi, {"hit": 0, "miss": 0, "pending": 0, "other": 0})
        g[name] += 1
        n = totals["hit"] + totals["miss"] + totals["pending"] + totals["other"]
        if n <= 40:
            log.info(
                "offloaddbg LOOKUP #%d %s hash=%s group=%s",
                n, name, get_offload_block_hash(key)[:12].hex(), gi,
            )
        if n % 200 == 0 and n:
            log.info("offloaddbg LOOKUP totals=%s per_group=%s", totals, groups)
        return res

    CPUOffloadingManager.lookup = lookup

    _store = CPUOffloadingManager.prepare_store

    def prepare_store(self, keys, req_context):
        keys = list(keys)
        calls["store"] += 1
        calls["store_keys"] += len(keys)
        if calls["store"] <= 3:
            heads = {}
            for k in keys:
                gi = get_offload_group_idx(k)
                heads.setdefault(gi, []).append(get_offload_block_hash(k)[:12].hex())
            per = {gi: len(h) for gi, h in heads.items()}
            log.info(
                "offloaddbg STORE call=%d keys=%d per_group=%s total_keys=%d heads=%s",
                calls["store"], len(keys), per, calls["store_keys"],
                {gi: h[:3] for gi, h in heads.items()},
            )
        out = _store(self, keys, req_context)
        try:
            kept = len(getattr(out, "keys_to_store", []) or [])
            if calls["store"] <= 8 or calls["store"] % 25 == 0:
                log.info("offloaddbg STORE call=%d kept=%d", calls["store"], kept)
        except Exception:
            pass
        return out

    CPUOffloadingManager.prepare_store = prepare_store


_rad_offload_debug_install_manager()
'''

SCHEDULER_BLOCK = '''

# radiance-offload-debug
def _rad_offload_debug_install_scheduler():
    import logging
    log = logging.getLogger("vllm.offloaddbg")

    _init = OffloadingConnectorScheduler.__init__

    def __init__(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        try:
            for gc in self.config.kv_group_configs:
                log.info(
                    "offloaddbg GROUP idx=%s tokens_per_chunk=%s hashes_per_chunk=%s "
                    "sw_chunks=%s align=%s eagle=%s",
                    gc.group_idx, gc.tokens_per_chunk, gc.hashes_per_chunk,
                    gc.sliding_window_size_in_chunks, gc.alignment_chunk_count,
                    gc.is_eagle_group,
                )
            log.info("offloaddbg LOOKUP_GROUPS=%s", sorted(self._lookup_groups))
        except Exception as exc:
            log.info("offloaddbg GROUP log failed: %r", exc)

    OffloadingConnectorScheduler.__init__ = __init__


_rad_offload_debug_install_scheduler()
'''


def append_once(path: Path, block: str) -> None:
    text = path.read_text()
    if SENTINEL in text:
        print(f"[patch] offload-debug: {path.name} already instrumented")
        return
    path.write_text(text + block)
    print(f"[patch] offload-debug: instrumented {path}")


append_once(MANAGER, MANAGER_BLOCK)
append_once(SCHEDULER, SCHEDULER_BLOCK)
