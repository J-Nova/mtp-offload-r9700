#!/usr/bin/env python3
"""RADIANCE offload eagle/MTP group inclusion (E1, Stage 2) — gate RADIANCE_OFFLOAD_EAGLE_INCLUDE.

Stage 1 (`patch_offload_suffix_inv.py`) added rejection-driven suffix invalidation. Stage 2 removes
the two withholding mechanisms that currently exclude the volatile MTP/draft trailing chunk, so MTP
groups are stored/loaded and the per-turn one-chunk hit-rate cap is lifted:

  store side: `storable_chunks` no longer drops the trailing chunk while decoding;
  load side:  `_lookup_complete_chunks` no longer queries an extra chunk and pops it.

Correctness is provided by Stage 1: a rejected draft's stale suffix is invalidated. Failure mode if
the invalidation races an in-flight store is reduced draft acceptance (stale *draft* KV), not wrong
served tokens — the target verifies every proposal. Validate with byte-identical outputs + acceptance
parity before trusting.

Default off; requires RADIANCE_OFFLOAD_SUFFIX_INV=1 to be meaningful. Idempotent; ast.parse.
"""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_offload_eagle_include"
SP = Path(sysconfig.get_paths()["purelib"])
OSCHED = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

PAIRS = [
    (
        "store-drop",
        "        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens\n"
        "        if group_config.is_eagle_group and is_decoding:\n"
        "            num_chunks = max(0, num_chunks - 1)\n",
        "        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens\n"
        "        if (group_config.is_eagle_group and is_decoding\n"
        "                and not _EAGLE_INCLUDE):\n"
        "            num_chunks = max(0, num_chunks - 1)\n",
    ),
    (
        "load-query-extra",
        "                query_max = max_hit_size_tokens\n"
        "                if is_eagle_unverified and sliding_window_size_in_chunks is not None:\n",
        "                query_max = max_hit_size_tokens\n"
        "                if (is_eagle_unverified and not _EAGLE_INCLUDE\n"
        "                        and sliding_window_size_in_chunks is not None):\n",
    ),
    (
        "load-required-window",
        "                if required_window is not None:\n"
        "                    if is_eagle_unverified:\n"
        "                        required_window += 1\n",
        "                if required_window is not None:\n"
        "                    if is_eagle_unverified and not _EAGLE_INCLUDE:\n"
        "                        required_window += 1\n",
    ),
    (
        "load-pop-chunk",
        "                    if is_eagle_unverified:\n"
        "                        num_hit_chunks -= 1\n"
        "                        eagle_verified.add(group_idx)\n",
        "                    if is_eagle_unverified and not _EAGLE_INCLUDE:\n"
        "                        num_hit_chunks -= 1\n"
        "                        eagle_verified.add(group_idx)\n",
    ),
]

TAIL = '''

# ---- RADIANCE offload eagle/MTP group inclusion (patch_offload_eagle_include) ----------------
import os as _ei_os

_EAGLE_INCLUDE = _ei_os.environ.get("RADIANCE_OFFLOAD_EAGLE_INCLUDE", "0") == "1"
'''


def main():
    src = OSCHED.read_text()
    if MARK in src:
        print("[eagle-include] already applied")
        return
    for tag, old, new in PAIRS:
        n = src.count(old)
        if n != 1:
            print(f"[eagle-include] anchor {tag} matched {n}x, NOT applied", file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    src += TAIL
    ast.parse(src)
    OSCHED.write_text(src)
    print("[eagle-include] applied: offloading/scheduler.py")


if __name__ == "__main__":
    main()
