#!/usr/bin/env python3
"""RADIANCE mamba conv spec-scratch zeroing (port of tcclaviger dev, vLLM 0.29).

Speculative decoding allocates conv-state width `conv_width = state_len + num_spec`
(`mamba_utils.py:_orient_conv_shape(divide(conv_dim, tp), conv_kernel_size - 1 + num_spec)`),
while the conv kernel's logical window is `state_len = conv_width - 1`. The trailing
`num_spec - (something)` columns are scratch the verify reads at
`conv_state_token_offset = num_accepted_tokens - 1 > 0`.

The align state-copy (used by `--mamba-cache-mode align`, which this deployment runs, and by
the KV-offload page migration) shifts the window by `token_bias` and copies only
`conv_width - token_bias` columns, leaving the destination tail holding whatever the
destination page's previous owner wrote. A later verify then convolves those stale values.

Fix (mirrors the dev's `v1/worker/mamba_utils.py` change): zero the destination columns
`[num_dst_tokens, conv_width)` in both conv layouts, on both copy paths (the distinct/self
u64 fast path and the same-block path). Triton-only; no kernel rebuild. The conv-kernel
write side (causal_conv1d ZERO_SPARE) is a separate item and is not touched here.

Idempotent (marker patch_mamba_scratch_zero); every edited file must ast.parse.
"""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_mamba_scratch_zero"
SP = Path(sysconfig.get_paths()["purelib"])
MU = SP / "vllm/v1/worker/mamba_utils.py"

# ---- DS conv layout (CONV_STATE_DIM_FIRST): zero tail dim rows -----------------------------
DS_OLD = (
    "                        data_u8 = tl.load(src_u8, mask=mask)\n"
    "                        tl.store(dst_u8, data_u8, mask=mask)\n"
    "        return\n"
    "\n"
    "    if is_conv_state:\n"
)
DS_NEW = (
    "                        data_u8 = tl.load(src_u8, mask=mask)\n"
    "                        tl.store(dst_u8, data_u8, mask=mask)\n"
    "        # patch_mamba_scratch_zero: columns past the shifted window keep the destination\n"
    "        # block's previous owner; the spec verify reads them at accepted offset > 0.\n"
    "        for _zt in range(num_dst_tokens, conv_width):\n"
    "            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):\n"
    "                rows = row_base + offsets\n"
    "                mask = rows < dim_rows\n"
    "                dst_byte_addr = dst_addr + rows * row_stride + _zt * state_elem_size\n"
    "                if state_elem_size == 2:\n"
    "                    tl.store(dst_byte_addr.to(tl.pointer_type(tl.uint16)),\n"
    "                             tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.uint16), mask=mask)\n"
    "                elif state_elem_size == 4:\n"
    "                    tl.store(dst_byte_addr.to(tl.pointer_type(tl.uint32)),\n"
    "                             tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.uint32), mask=mask)\n"
    "                else:\n"
    "                    for byte_idx in range(0, state_elem_size):\n"
    "                        tl.store((dst_byte_addr + byte_idx).to(tl.pointer_type(tl.uint8)),\n"
    "                                 tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.uint8), mask=mask)\n"
    "        return\n"
    "\n"
    "    if is_conv_state:\n"
)

# ---- SD conv layout: distinct/self fast path ---------------------------------------------
SD1_OLD = (
    "            copy_size = num_dst_tokens.to(tl.int64) * token_bytes\n"
    "            _memcpy_u64_tiled(\n"
    "                src_addr,\n"
    "                dst_addr,\n"
    "                copy_size,\n"
    "                tile_idx,\n"
    "                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,\n"
    "                NUM_TILES=1,\n"
    "            )\n"
    "            return\n"
)
SD_TAIL_12 = (
    "            # patch_mamba_scratch_zero: zero the destination tail past the shifted window.\n"
    "            _zbytes = token_bias.to(tl.int64) * token_bytes\n"
    "            _zbase = (dst_addr + num_dst_tokens.to(tl.int64) * token_bytes).to(\n"
    "                tl.pointer_type(tl.uint8)\n"
    "            )\n"
    "            _zoff = tl.arange(0, COPY_BLOCK_SIZE)\n"
    "            for _zi in range(0, _zbytes, COPY_BLOCK_SIZE):\n"
    "                tl.store(_zbase + _zi + _zoff, tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.uint8),\n"
    "                         mask=(_zi + _zoff) < _zbytes)\n"
)
SD1_NEW = SD1_OLD.replace("            return\n", SD_TAIL_12 + "            return\n")

# ---- SD conv layout: same-block path ------------------------------------------------------
SD2_OLD = (
    "            _memcpy_u64_tiled(\n"
    "                src_token,\n"
    "                dst_token,\n"
    "                token_bytes,\n"
    "                tile_idx,\n"
    "                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,\n"
    "                NUM_TILES=1,\n"
    "            )\n"
    "        return\n"
)
SD_TAIL_8 = (
    "        # patch_mamba_scratch_zero: zero the destination tail past the shifted window.\n"
    "        _zbytes = token_bias.to(tl.int64) * token_bytes\n"
    "        _zbase = (dst_addr + num_dst_tokens.to(tl.int64) * token_bytes).to(\n"
    "            tl.pointer_type(tl.uint8)\n"
    "        )\n"
    "        _zoff = tl.arange(0, COPY_BLOCK_SIZE)\n"
    "        for _zi in range(0, _zbytes, COPY_BLOCK_SIZE):\n"
    "            tl.store(_zbase + _zi + _zoff, tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.uint8),\n"
    "                     mask=(_zi + _zoff) < _zbytes)\n"
)
SD2_NEW = SD2_OLD.replace("        return\n", SD_TAIL_8 + "        return\n")


def main():
    src = MU.read_text()
    if MARK in src:
        print("[mamba-zero] already applied")
        return
    for tag, old, new in (("ds", DS_OLD, DS_NEW), ("sd1", SD1_OLD, SD1_NEW), ("sd2", SD2_OLD, SD2_NEW)):
        n = src.count(old)
        if n != 1:
            print(f"[mamba-zero] anchor {tag} matched {n}x, NOT applied", file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    ast.parse(src)
    MU.write_text(src)
    print("[mamba-zero] applied: mamba_utils.py")


if __name__ == "__main__":
    main()
