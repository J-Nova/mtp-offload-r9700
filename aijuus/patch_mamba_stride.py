#!/usr/bin/env python3
"""Store the Mamba/GDN groups every Nth chunk instead of every chunk (the density lever).

Ported from zzpanic/qwen3.6-vllm-gfx1201-launchers `kv-cache/patches/patch_mamba_stride.py`
(same vLLM 0.27.1 / radiance 0.9.3 image), re-anchored to this tree.

THE PROBLEM. A Mamba group holds ONE recurrent state, not a per-token history
(`get_sliding_window_size_in_chunks` returns 1 for MambaSpec), and the load path fetches
exactly one Mamba chunk per request (the last one). Yet the store path writes a fresh snapshot
of all six Mamba/GDN groups at every chunk boundary: ~N snapshots for a long conversation where
the GPU keeps 2. Measured on the sibling build: every 1,648-token chunk writes all nine groups
at 27,000,832 B each = 147,456 B/token, of which only ~34,264 B/token is ever read back -- a
4.17x amplification. The Mamba state is real (g0-g5 are 97-99% dense), so the waste is
TEMPORAL, not padding.

WHAT THIS CHANGES. Keep every Nth Mamba snapshot (N = RADIANCE_MAMBA_STORE_STRIDE, default 1 =
off), with two halves that must agree:
  STORE  -- in _build_store_jobs, a Mamba group's chunk is stored only when
            (absolute_chunk_index + 1) % N == 0.
  LOOKUP -- resolve_mamba_align_size returns N * tokens_per_chunk, so _lookup's existing
            round_down() clamps max_hit_size_tokens to a multiple of N chunks.
The absolute grid (`abs_chunk_idx`, not tail-relative) is what makes this work: the existing
is_store_reachable_swa_chunk() is relative to the current tail, so during chunked prefill every
chunk is the tail once and would be stored anyway.

WHY THE TWO HALVES ARE CONSISTENT. For a Mamba group the load materialises only the final
chunk's blocks (index num_chunks - 1, where num_chunks = max_hit_size_tokens / tokens_per_chunk).
max_hit_size_tokens is rounded down to a multiple of N * tokens_per_chunk, so num_chunks is a
multiple of N and the requested index is N-1 (mod N) -- exactly what STORE keeps.

WHAT IT BUYS. Per N=8 chunks, in units of one block: today 8 x 9 = 72; with the stride
6 Mamba x 1 + 2 attention x 8 + 1 draft x 8 = 30, i.e. 0.417x -- 2.4x more tokens resident per
byte. On the sibling build the CPU tier went 115,360 -> ~276,900 tokens, which is what moves a
hit from the disk tier (a 64 s promotion) to the CPU tier (1-2 s).

WHAT IT COSTS. A hit is truncated down to an N-chunk boundary: up to N * tokens_per_chunk of
held prefix is declined (our chunk is 880 tokens; N=4 -> 3,520, N=8 -> 7,040). Dead zone: any
prefix shorter than N * tokens_per_chunk gets ZERO external hit, because MambaSpec's window is
1 and _sliding_window_lookup finds no snapshot for a sub-N-chunk prefix. That is why this is a
capacity trade, not a free win -- but the Mamba boundary already limited the hit, so it is not
an extra cost on top of the attention groups.

PREREQUISITE. Apply `patch_offload_eagle_groups.py` (or keep the eagle fallback restriction)
first: while every group is mislabelled as a draft group, storable_chunks() drops each group's
trailing chunk during decode and the store grid stops lining up with the hit window.

GATE. RADIANCE_MAMBA_STORE_STRIDE=1 (default) disables the patch completely -- both halves
collapse to today's behaviour -- without unpatching. Larger N trades more truncated prefix for
more CPU-tier residency. Default is 1 here (opt-in); the sibling build ships 4.
"""
import os
import sys
import sysconfig
from pathlib import Path

# _patchlib.py lives at the repo root (one level up from aijuus/); insert it
# explicitly since sys.path[0] is aijuus/ when run as `python3 aijuus/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _patchlib import apply  # noqa: E402

SP = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"]))
SCHED = (
    SP
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "offloading"
    / "scheduler.py"
)

print("[radiance] mamba store cadence")

apply(
    SCHED,
    anchor=(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "import time\n"
    ),
    new=(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "import os as _radiance_os\n"
        "import time\n"
        "\n"
        "# radiance mamba store cadence: keep every Nth Mamba/GDN snapshot rather than one per\n"
        "# chunk. A Mamba group holds a single recurrent state and the load path reads exactly one\n"
        "# chunk of it, yet the store path writes a full snapshot of all six groups at every chunk\n"
        "# boundary. 1 disables the patch entirely (both halves collapse to today's behaviour).\n"
        "_RADIANCE_MAMBA_STRIDE = max(\n"
        "    1, int(_radiance_os.environ.get(\"RADIANCE_MAMBA_STORE_STRIDE\", \"1\"))\n"
        ")\n"
    ),
    sentinel="_RADIANCE_MAMBA_STRIDE = max(",
    label="1 scheduler: RADIANCE_MAMBA_STORE_STRIDE knob",
)

apply(
    SCHED,
    anchor="    is_eagle_group: bool = False\n",
    new=(
        "    is_eagle_group: bool = False\n"
        "    # radiance mamba store cadence: True for MambaSpec groups. Distinct from\n"
        "    # sliding_window_size_in_chunks == 1, which a genuinely small attention window would\n"
        "    # also produce.\n"
        "    is_mamba_group: bool = False\n"
    ),
    sentinel="is_mamba_group: bool = False",
    label="2 scheduler: GroupOffloadConfig.is_mamba_group",
)

apply(
    SCHED,
    anchor="                    is_eagle_group=idx in eagle_groups,\n",
    new=(
        "                    is_eagle_group=idx in eagle_groups,\n"
        "                    is_mamba_group=isinstance(  # radiance mamba store cadence\n"
        "                        kv_cache_config.kv_cache_groups[idx].kv_cache_spec, MambaSpec\n"
        "                    ),\n"
    ),
    sentinel="is_mamba_group=isinstance(",
    label="3 scheduler: populate is_mamba_group in from_spec",
)

apply(
    SCHED,
    anchor=("            mamba_align_size = tokens_per_chunk\n    return mamba_align_size\n"),
    new=(
        "            mamba_align_size = tokens_per_chunk\n"
        "    if mamba_align_size is not None and _RADIANCE_MAMBA_STRIDE > 1:\n"
        "        # radiance mamba store cadence: we only keep every Nth Mamba snapshot, so the hit\n"
        "        # window must land on the same grid -- otherwise _lookup asks for a state we did\n"
        "        # not store and _sliding_window_lookup walks backwards probing chunks that cannot\n"
        "        # exist. This rounding is what creates the dead zone: a prefix shorter than\n"
        "        # N * tokens_per_chunk gets zero external hit.\n"
        "        mamba_align_size *= _RADIANCE_MAMBA_STRIDE\n"
        "    return mamba_align_size\n"
    ),
    sentinel="radiance mamba store cadence: we only keep every Nth Mamba snapshot",
    label="4 scheduler: resolve_mamba_align_size honours the stride",
)

apply(
    SCHED,
    anchor="                    abs_chunk_idx = start_chunk_idx + key_idx\n",
    new=(
        "                    abs_chunk_idx = start_chunk_idx + key_idx\n"
        "                    # radiance mamba store cadence: keep only every Nth Mamba/GDN\n"
        "                    # snapshot. The grid is absolute, not tail-relative: a tail-relative\n"
        "                    # rule would store every chunk as it briefly became the newest one.\n"
        "                    if (\n"
        "                        group_config.is_mamba_group\n"
        "                        and _RADIANCE_MAMBA_STRIDE > 1\n"
        "                        and (abs_chunk_idx + 1) % _RADIANCE_MAMBA_STRIDE != 0\n"
        "                    ):\n"
        "                        continue\n"
    ),
    sentinel="radiance mamba store cadence: keep only every Nth Mamba/GDN",
    label="5 scheduler: skip off-grid Mamba chunks in _build_store_jobs",
)

print(
    "[radiance] applied -- stride "
    f"{os.environ.get('RADIANCE_MAMBA_STORE_STRIDE', '1')} (1 = off)"
)
