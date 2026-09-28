#!/usr/bin/env python3
"""Fix the native KV-offload connector for hybrid (attention + Mamba/GDN) models under
speculative decoding, so dflash and CPU offload can be used together.

SYMPTOM. SPEC_METHOD=dflash + OffloadingConnector: the connector allocates its CPU region and
stores tens of GB, but vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} stays 0.0
and "External prefix cache hit rate" stays 0.0%, so a prefix evicted from the GPU pool is
recomputed in full (measured: a 130k re-prefill of ~66s instead of a ~3s host->GPU restore).
With speculative decoding off the same connector restores correctly.

ROOT CAUSE. SchedulerOffloadConfig.from_spec falls back to marking EVERY KV cache group as an
EAGLE/MTP group when speculative_config.use_eagle() is true but no group carries is_eagle_group:

    if use_eagle and not eagle_groups:
        eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

For this hybrid model that mislabels the six Mamba/GDN (SSM) groups as draft attention groups.
Consequences, both keyed off GroupOffloadConfig.is_eagle_group:
  * load path: required_window = sliding_window_size_in_chunks + 1 for eagle groups; the GDN
    groups have sliding_window_size_in_chunks == 1, so 2 chunks are demanded;
  * store path: is_store_reachable_swa_chunk uses reachable_tail = sliding_window + 1, but the
    chunks that actually survive are the segment tail, so the demanded 2 are never present.
The GDN group's _sliding_window_lookup therefore returns 0 hits, and the load loop is
all-or-nothing -- `if num_hit_chunks == 0: return 0` -- so a single failing SSM group aborts the
restore for ALL groups.

The core vLLM already treats SSM as exempt from the EAGLE trailing-chunk rule:
v1/core/kv_cache_coordinator.py:755 guards the eagle margin with
`not isinstance(spec, MambaSpec)`, and MambaManager.find_longest_cache_hit ignores
drop_eagle_block because "draft models have no mamba layers". Only the offloading connector
missed this guard, and its coarse fallback made it worse by flagging the SSM groups.

MEASURED EVIDENCE (Qwen3.8-27B MXFP4 + Qwen3.8 DFlash2-FP8, TP=1, CHUNK=4096, mamba align,
9 offload groups: 0-5 GDN (sw=1), 6-7 full attention, 8 draft SWA (sw=3)):
  * before: all 9 groups eagle=True; GDN _sliding_window_lookup window=2 -> hits=1/misses=145
    (the only stored GDN state is the tail chunk) -> 0 -> whole request aborts, CPU_to_GPU=0;
  * after: groups 0-5 eagle=False (window=1) and groups 6,7,8 eagle=True; all six GDN groups
    return the tail hit, the load converges, restore = 126720/130017 tokens, E TTFT 2.92s vs
    66.09s cold, CPU_to_GPU=4.26 GB.
  * correctness: 192 greedy tokens after an offload-restored 90k prefix are byte-identical to
    the cold-prefill continuation (MATCH=True).

NOTE. GDN state snapshots only exist at block-aligned prefill-step ends (indices
3,7,11,...,143,145 for a 147-chunk prompt), so the load's window=1 caps the restore to the
nearest stored SSM boundary -- correct, but it means the CPU tier must retain those (large)
SSM blocks. At 130k, 16 GiB was not enough (SSM keys evicted, restore aborted); 24 GiB worked.
Size cpu_bytes_to_use accordingly for long contexts.

FIX. The volatile-trailing-chunk rule is a property of draft ATTENTION KV (spec-token rejection
can rewrite the last accepted position's KV); it does not apply to SSM state groups. Restrict the
fallback to non-Mamba groups, mirroring the core's MambaSpec guard. The intended eagle handling
for real attention groups (and for a draft model's own sliding-window attention) is preserved.

Applied by default (RADIANCE_OFFLOAD_EAGLE_FIX unset or 1); RADIANCE_OFFLOAD_EAGLE_FIX=0 skips.
`RADIANCE_OFFLOAD_EAGLE_FIX=none` disables the fallback entirely (diagnostic). Idempotent; run
once pre-serve.
"""
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
SENTINEL = "_radiance_offload_eagle_fix"

ANCHOR = """        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
"""

REPLACEMENT = """        if use_eagle and not eagle_groups:
            # _radiance_offload_eagle_fix: restrict the fallback to attention groups.
            # A Mamba/SSM (GDN) group is not draft attention KV, and marking it eagle makes the
            # load path demand sliding_window+1 chunks while the store only keeps the segment
            # tail, so the group always reports 0 hits -- and because the load is all-or-nothing
            # that single zero aborts the restore for every group. MambaSpec is already imported
            # in this module. RADIANCE_OFFLOAD_EAGLE_FIX=none disables the fallback entirely
            # (diagnostic: reproduces the spec-off group flags under spec decode).
            import os as _os
            if _os.environ.get("RADIANCE_OFFLOAD_EAGLE_FIX", "").strip().lower() == "none":
                eagle_groups = set()
            else:
                eagle_groups = {
                    idx
                    for idx, g in enumerate(kv_cache_config.kv_cache_groups)
                    if not isinstance(g.kv_cache_spec, MambaSpec)
                }
"""


def main() -> None:
    text = TARGET.read_text()
    if SENTINEL in text:
        print(f"[patch] offload-eagle-fix: already applied ({TARGET.name})")
        return
    if ANCHOR not in text:
        print(f"[patch] offload-eagle-fix: ANCHOR NOT FOUND in {TARGET}; not applied")
        return
    TARGET.write_text(text.replace(ANCHOR, REPLACEMENT, 1))
    print(f"[patch] offload-eagle-fix: applied to {TARGET}")


main()
