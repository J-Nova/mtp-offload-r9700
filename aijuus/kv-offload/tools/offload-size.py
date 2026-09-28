#!/usr/bin/env python3
"""Recommend a size for the vLLM CPU KV-offload tier (OffloadingConnector).

The connector keeps evicted prefixes in a CPU region (`cpu_bytes_to_use`, a `/dev/shm`
mmap). It only helps when the *sum of distinct live long prefixes* exceeds the GPU KV
pool, and it only restores a prefix if that prefix is stored **whole** -- full attention
needs every block from chunk 0, and the LRU policy evicts the oldest first, so a prefix
whose KV does not fit the tier loses its head and re-prefills entirely (no partial
recovery). Restore itself is a two-tier chain: the GPU cache serves the head and the CPU
tier serves from the GPU hit boundary.

Model (all constants measured on this host, 2026-09-23):

  GPU    ~38.4 KB per token   (production pin 8761733283 B / 227981 tokens)
  CPU    ~2.0x that per token, because a CPU block is sized for one full
         chunk-across-all-groups (`worker_kv_bytes_per_block`) while each key stores a
         single group. Net: ~14k tokens/GiB, i.e. ~1.9 GiB per 130k-token context.

  Sizing:   cpu_bytes_to_use >= sum(prefix_tokens) * 2.0 * 38.4 KB

Validated against measurements at KV 8.16 GiB: 2x130k (~19 GiB) and 4x80k (~23 GiB)
restored with a 24 GiB tier; 3x130k (~28 GiB) needed 28 GiB and failed at 24 GiB.

Examples:
  ./offload-size.py                                  # production shape, 3 sessions
  ./offload-size.py --context 130000 --sessions 3
  ./offload-size.py --context 60000 --sessions 8
  ./offload-size.py --capacity --cpu-gib 24          # what a 24 GiB tier can hold
"""
import argparse
import json
import math
import shutil
import sys

MIB = 1 << 20
GIB = 1 << 30

# Measured on this host (Qwen3.8-27B MXFP4 + DFlash2, TP=1, CHUNK=4096, kv fp8).
DEFAULT_GPU_KV_BYTES = 8761733283  # production KV_CACHE_MEMORY pin
DEFAULT_GPU_TOKENS = 227981  # KV tokens that pin holds
DEFAULT_CPU_OVERHEAD = 2.0  # CPU bytes/token vs GPU bytes/token
DEFAULT_CONTEXT = 160000  # deployment MAX_MODEL_LEN
DEFAULT_SESSIONS = 3


def gib(n: int) -> float:
    return n / GIB


def mib_ceil(n: float) -> int:
    return int(math.ceil(n / MIB)) * MIB


def shm_total_bytes() -> int | None:
    try:
        return shutil.disk_usage("/dev/shm").total
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recommend cpu_bytes_to_use for the vLLM CPU KV-offload tier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                   help="tokens per long prefix (usually MAX_MODEL_LEN)")
    p.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                   help="distinct long prefixes you want restorable at once")
    p.add_argument("--gpu-kv-bytes", type=int, default=DEFAULT_GPU_KV_BYTES,
                   help="KV_CACHE_MEMORY pin, in bytes")
    p.add_argument("--gpu-tokens", type=int, default=DEFAULT_GPU_TOKENS,
                   help="KV tokens that pin holds (from the startup log)")
    p.add_argument("--cpu-overhead", type=float, default=DEFAULT_CPU_OVERHEAD,
                   help="CPU bytes/token as a multiple of GPU bytes/token")
    p.add_argument("--safety", type=float, default=1.05,
                   help="headroom multiplier on the recommendation")
    p.add_argument("--chunk", type=int, default=880,
                   help="tokens per offload chunk (offload group tokens_per_chunk)")
    p.add_argument("--mamba-stride", type=int, default=1,
                   help="RADIANCE_MAMBA_STORE_STRIDE: store Mamba/GDN snapshots every Nth "
                        "chunk (1 = off; the CPU-tier density lever, see patch_mamba_stride.py)")
    p.add_argument("--mamba-share", type=float, default=6 / 9,
                   help="fraction of offload groups that are Mamba/GDN (6 of 9 on this model)")
    p.add_argument("--shm-gib", type=float, default=None,
                   help="available /dev/shm in GiB (default: auto-detect)")
    p.add_argument("--capacity", action="store_true",
                   help="instead, report what a given --cpu-gib tier can retain")
    p.add_argument("--cpu-gib", type=float, default=None,
                   help="tier size to evaluate with --capacity")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return p.parse_args()


def main() -> int:
    a = parse_args()

    bpt_gpu = a.gpu_kv_bytes / a.gpu_tokens
    # Stride only divides the Mamba/GDN share of what is stored per chunk (the six groups are
    # snapshots, not histories). Modelled from the group counts, not measured here: at stride 1
    # the factor is 1 (validated thresholds below), at stride 8 it is (3 + 6/8)/9 = 0.417, i.e.
    # 2.4x denser -- the sibling build's measured 0.417x. Validate before trusting it.
    stride = max(1, a.mamba_stride)
    stride_factor = (1.0 - a.mamba_share) + a.mamba_share / stride
    bpt_cpu = bpt_gpu * a.cpu_overhead * stride_factor

    shm_bytes = int(a.shm_gib * GIB) if a.shm_gib else shm_total_bytes()
    shm_note = f"{gib(shm_bytes):.2f} GiB" if shm_bytes else "unknown"

    # ---- capacity mode -------------------------------------------------------
    if a.capacity:
        if not a.cpu_gib:
            print("--capacity needs --cpu-gib", file=sys.stderr)
            return 2
        tier = int(a.cpu_gib * GIB)
        tokens = tier / bpt_cpu
        sessions = tokens / a.context if a.context else 0
        out = {
            "cpu_gib": a.cpu_gib,
            "mamba_stride": stride,
            "retainable_tokens": int(tokens),
            "sessions_at_context": round(sessions, 2),
            "gpu_pool_tokens": a.gpu_tokens,
            "gain_over_gpu_tokens": int(tokens - a.gpu_tokens),
        }
        if a.json:
            print(json.dumps(out, indent=2))
            return 0
        print(f"CPU tier {a.cpu_gib:.2f} GiB  (cpu {bpt_cpu/1024:.1f} KB/token)")
        print(f"  retainable tokens   : {int(tokens):,}")
        print(f"  sessions @ {a.context:,}  : {sessions:.2f}")
        print(f"  GPU pool            : {a.gpu_tokens:,} tokens")
        print(f"  tokens beyond GPU   : {int(tokens - a.gpu_tokens):+,}")
        verdict = ("offload adds capacity" if tokens > a.gpu_tokens
                   else "smaller than the GPU pool -- offload adds no retention")
        print(f"  verdict             : {verdict}")
        return 0

    # ---- sizing mode ---------------------------------------------------------
    working = a.sessions * a.context
    gpu_can_hold = a.gpu_tokens
    needed = mib_ceil(working * bpt_cpu * a.safety)

    # If the tier cannot hold the whole working set, a per-request head cap is required:
    # LRU would evict each prefix's head and it would restore nothing at all
    # (see patch_offload_head_cap.py).
    retainable_tokens = int(shm_bytes / bpt_cpu) if shm_bytes else None
    head_cap = 0
    if retainable_tokens is not None and needed > shm_bytes:
        # Keep one contiguous head per prefix, with the same headroom as the tier
        # itself, so the caps of `--sessions` prefixes fit the tier together.
        capped = shm_bytes / (bpt_cpu * a.safety)
        head_cap = (int(capped) // a.sessions // a.chunk) * a.chunk

    out = {
        "context_tokens": a.context,
        "sessions": a.sessions,
        "working_set_tokens": working,
        "gpu_pool_tokens": gpu_can_hold,
        "bytes_per_token_gpu": round(bpt_gpu, 1),
        "bytes_per_token_cpu": round(bpt_cpu, 1),
        "mamba_stride": stride,
        "cpu_stride_factor": round(stride_factor, 4),
        "recommended_cpu_bytes": needed,
        "recommended_cpu_gib": round(gib(needed), 2),
        "shm_total_gib": round(gib(shm_bytes), 2) if shm_bytes else None,
        "retainable_tokens": retainable_tokens,
        "head_cap_tokens": head_cap or None,
    }

    needed_min = mib_ceil(working * bpt_cpu)  # no safety headroom
    fits_gpu = working <= gpu_can_hold
    out["min_cpu_bytes"] = needed_min
    out["fits_in_gpu_pool"] = fits_gpu
    out["threshold_basis"] = "measured" if stride == 1 else "modelled"

    if a.json:
        print(json.dumps(out, indent=2))
        return 0

    print("KV CPU-offload sizing")
    print(f"  GPU pool            : {gpu_can_hold:,} tokens "
          f"({gib(a.gpu_kv_bytes):.2f} GiB, {bpt_gpu/1024:.1f} KB/token)")
    stride_note = f", mamba stride {a.mamba_stride}" if a.mamba_stride > 1 else ""
    print(f"  CPU tier density    : {bpt_cpu/1024:.1f} KB/token "
          f"({bpt_cpu and GIB/bpt_cpu:,.0f} tokens/GiB, "
          f"{a.cpu_overhead * stride_factor:.2f}x GPU{stride_note})")
    print(f"  working set         : {a.sessions} x {a.context:,} = {working:,} tokens")
    print(f"  /dev/shm            : {shm_note}")

    if fits_gpu:
        print()
        print(f"  -> {working:,} tokens fit in the {gpu_can_hold:,}-token GPU pool.")
        print("     Offload is not needed for this workload (GPU hits are ~0.75 s).")
        print("     Enable it only for headroom; a small tier (~8 GiB) costs nothing")
        print("     but also restores nothing under pressure. Recommended: disabled.")
        return 0

    print()
    print(f"  GPU holds only {gpu_can_hold:,} of {working:,} tokens; "
          f"{working - gpu_can_hold:,} must come from the CPU tier.")
    print(f"  minimum tier        : {gib(needed_min):.2f} GiB "
          f"({'measured' if stride == 1 else 'MODELLED: stride>1 is not yet measured here'})")
    print(f"  recommended tier    : {gib(needed):.2f} GiB "
          f"({needed:,} B, {a.safety:.2f}x headroom)")

    if shm_bytes:
        pct = 100.0 * needed / shm_bytes
        print(f"  of /dev/shm         : {pct:.0f}%")
        if needed > shm_bytes:
            max_tokens = shm_bytes / (bpt_cpu * a.safety)
            print()
            print("  WARNING: does not fit /dev/shm.")
            print(f"     max retainable      : {int(max_tokens):,} tokens "
                  f"(~{max_tokens / a.context:.1f} sessions @ {a.context:,})")
            print("     lower --sessions/--context, raise /dev/shm, or accept re-prefill")
            if head_cap > 0:
                print(f"     or cap each prefix : max_offload_tokens={head_cap} "
                      f"({head_cap / a.context:.0%} of context restored, rest prefilled)")
            return 1

    print()
    print("  apply:")
    print(f'    cpu_bytes_to_use = {needed}')
    print("    Coolify env on both instances:")
    print(f'      KV_CACHE_MEMORY={a.gpu_kv_bytes}')
    ec = f'"cpu_bytes_to_use":{needed}'
    if head_cap:
        ec += f',"max_offload_tokens":{head_cap}'
    print('      EXTRA=\'--kv-transfer-config {"kv_connector":"OffloadingConnector",'
          '"kv_role":"kv_both","kv_connector_extra_config":'
          f'{{{ec}}}}}\'')
    if head_cap:
        print(f'      # or RADIANCE_OFFLOAD_MAX_TOKENS={head_cap}')
    if stride > 1:
        print(f'      RADIANCE_MAMBA_STORE_STRIDE={stride}   # requires patch_mamba_stride.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
