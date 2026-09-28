#!/usr/bin/env python3
"""Keep the HEAD of each prefix in the CPU offload tier instead of the tail.

WHY. Full attention can only be restored from chunk 0 -- it needs every block of the
prefix. The offload cache evicts LRU, so under capacity pressure it drops the OLDEST
blocks first, i.e. the head. A prefix larger than the tier therefore loses its head and
restores *nothing*: partial recovery is impossible, and a cycling workload (reuse P0..Pn
with sum(prefixes) > tier) wipes out every prefix before its turn, because re-prefilling
P0 stores its tail and evicts P1/P2's heads.

FIX. Cap the tokens offloaded per request so each prefix stores only its head. The
connector already supports this: `RequestOffloadState.max_offload_tokens` is applied in
`_calc_num_offloadable_tokens` as `min(num_computed, max_offload_tokens)` -- a cap from
the START. It was only reachable per request via `kv_transfer_params`; this patch adds a
server-level default so the offload tier can never be over-committed by a long prompt.

Set the cap so that `cap * sessions * keys_per_token <= tier_blocks`, i.e. roughly
`tier_tokens / expected_concurrent_prefixes` (see ./offload-size.py). When the tier is
large enough for the whole prefix the cap is a no-op (set it to 0 / omit it).

RESULT. With a cap, each prefix retains a contiguous head of `cap` tokens; a reuse
restores that head and prefills the remainder, instead of prefilling everything. It also
removes the cycling wipeout: nothing stores a tail, so nothing evicts another prefix's
head.

Config (in kv_connector_extra_config, or the RADIANCE_OFFLOAD_MAX_TOKENS env):
    "max_offload_tokens": 40000
    "max_offload_tokens": "auto"   # fair share of the tier per concurrent request
A per-request `kv_transfer_params["max_offload_tokens"]` still wins. "auto" needs the
offload spec to have resolved num_blocks (it is a no-op, returning None, otherwise).

Idempotent source patch of the installed connector; inert when the key/env is unset.
"""
import os
import sysconfig
from pathlib import Path

SP = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"]))
TARGET = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
SENTINEL = "_radiance_offload_head_cap"

ANCHOR_FIELD = """class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    offload_prompt_only: bool
"""

REPL_FIELD = """class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    offload_prompt_only: bool
    # _radiance_offload_head_cap: server-level default for max_offload_tokens.
    # Caps offloaded tokens from the START of each request so the CPU tier keeps a
    # contiguous prefix head (see patch_offload_head_cap.py). Per-request
    # kv_transfer_params still overrides.
    default_max_offload_tokens: int | None = None
"""

ANCHOR_RETURN = """            blocks_per_chunk=spec.blocks_per_chunk,
            offload_prompt_only=spec.offload_prompt_only,
        )
"""

REPL_RETURN = """            blocks_per_chunk=spec.blocks_per_chunk,
            offload_prompt_only=spec.offload_prompt_only,
            default_max_offload_tokens=_radiance_head_cap(
                spec, kv_cache_config, vllm_config
            ),
        )
"""

ANCHOR_CAP = """        return cls(
            num_workers=vllm_config.parallel_config.world_size,"""

REPL_CAP = """        return cls(
            num_workers=vllm_config.parallel_config.world_size,"""

ANCHOR_INIT = """        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
"""

REPL_INIT = """        raw = params.get("max_offload_tokens") if params else None
        if raw is None:
            # _radiance_offload_head_cap: fall back to the server-level default.
            raw = self.config.default_max_offload_tokens
        if type(raw) is int and raw >= 0:
"""

HELPER = '''

# radiance-offload-head-cap
def _radiance_head_cap(spec, kv_cache_config=None, vllm_config=None):
    """Server-wide default max_offload_tokens (extra_config, else env). None = off.

    "auto" derives a fair-share cap so every concurrent prefix keeps a contiguous head
    (see _radiance_auto_head_cap) -- the difference between restoring a partial prefix
    and restoring NOTHING when the working set exceeds the tier.
    """
    import os

    raw = spec.extra_config.get("max_offload_tokens")
    if raw is None:
        raw = os.environ.get("RADIANCE_OFFLOAD_MAX_TOKENS")
    if raw is None:
        return None
    if str(raw).strip().lower() == "auto":
        return _radiance_auto_head_cap(spec, kv_cache_config, vllm_config)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _radiance_auto_head_cap(spec, kv_cache_config, vllm_config):
    """Per-request cap = one request's fair share of the CPU tier, chunk-aligned.

    The tier holds `spec.num_blocks` chunk-slots (each `kv_bytes_per_chunk`), but a chunk
    needs one slot per storing KV group, so it retains about `num_blocks // num_groups`
    chunks -- on this model 24 GiB measures ~360k tokens, not the 686k the raw byte count
    suggests. Splitting that across `max_num_seqs` requests gives each prefix a head that
    fits, so a returning request always restores SOMETHING and prefills the rest, instead
    of losing its head to LRU and restoring nothing. Rounded down to the store/load grid
    (chunk x Mamba stride) so the retained head lands on a lookupable boundary.
    """
    import os

    try:
        num_blocks = int(getattr(spec, "num_blocks", 0))
        # The number of KV groups as the OFFLOAD connector sees them (len(tokens_per_block)),
        # NOT kv_cache_config.kv_cache_groups: patch_kv_group_size.py subdivides the GPU
        # grouping, and using that finer count underestimates the cap several-fold.
        n_groups = len(spec.tokens_per_block)
        sched = getattr(vllm_config, "scheduler_config", None)
        max_seqs = int(getattr(sched, "max_num_seqs", 0) or 0)
        tpc = int(spec.tokens_per_block[0]) * int(spec.blocks_per_chunk)
        stride = max(1, int(os.environ.get("RADIANCE_MAMBA_STORE_STRIDE", "1")))
    except Exception:
        return None
    if num_blocks <= 0 or n_groups <= 0 or max_seqs <= 0 or tpc <= 0:
        return None
    grid = max(1, tpc * stride)
    # (num_blocks // n_groups) * tpc models one block per (chunk, storing group). That
    # under-reports the MEASURED tier by ~1.8x on this model (model 174k tokens vs ~360k
    # measured: a 39,600-token cap kept all of 8x60k resident and restored 294k). The ratio
    # is the measured allocation efficiency; tune with RADIANCE_OFFLOAD_CAP_RATIO.
    try:
        ratio = float(os.environ.get("RADIANCE_OFFLOAD_CAP_RATIO", "1.8") or 1.8)
    except ValueError:
        ratio = 1.8
    tier_tokens = int((num_blocks // n_groups) * tpc * max(1.0, ratio))
    # 5% headroom: an exact fit still evicts on the first store after the last lookup.
    cap = ((tier_tokens * 95 // 100) // max_seqs) // grid * grid
    try:
        from vllm.logger import init_logger

        init_logger("vllm.radiance.offload").info(
            "[radiance] auto head cap: num_blocks=%d groups=%d max_seqs=%d tpc=%d "
            "stride=%d tier_tokens=%d -> cap=%s",
            num_blocks,
            n_groups,
            max_seqs,
            tpc,
            stride,
            tier_tokens,
            cap if cap > 0 else None,
        )
    except Exception:
        pass
    return cap if cap > 0 else None
'''


def main() -> None:
    text = TARGET.read_text()
    if SENTINEL in text:
        print(f"[patch] offload-head-cap: already applied ({TARGET.name})")
        return
    for anchor, repl, name in (
        (ANCHOR_FIELD, REPL_FIELD, "NamedTuple field"),
        (ANCHOR_RETURN, REPL_RETURN, "from_spec return"),
        (ANCHOR_INIT, REPL_INIT, "__post_init__ fallback"),
    ):
        if anchor not in text:
            print(f"[patch] offload-head-cap: ANCHOR NOT FOUND ({name}); not applied")
            return
        text = text.replace(anchor, repl, 1)
    text = text + HELPER
    TARGET.write_text(text)
    print(f"[patch] offload-head-cap: applied to {TARGET}")


main()
