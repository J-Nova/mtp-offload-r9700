#!/usr/bin/env python3
"""Fold calibrated KV scales into a single-file checkpoint.

`calibrate_kv.py` writes `model-kvscales.safetensors` plus entries in
`model.safetensors.index.json`, but a single-file checkpoint has no index, so it
leaves only the shard behind. This adds the index entries that make the shard
reachable.

Why no `quantization_config` change is needed: vLLM's Quark loader returns
`QuarkKVCacheMethod` for every non-excluded `Attention`
(`quantization/quark/quark.py`), and `BaseKVCacheMethod.create_weights` creates
`k_scale`/`v_scale`/`q_scale`/`prob_scale` during load. So scales named through
the calibrator's load-probe (a name it proved routes to that layer's parameter)
load as-is; `kv_cache_config` may stay `None`.

The merge is non-destructive: `model.safetensors` is never rewritten. Only the
index (temp-write + rename) and a copy of the small scales shard are written.

Usage:
    python3 merge_kv_scales.py --modeldir DIR [--scales SHARD] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

SCALE_SHARD = "model-kvscales.safetensors"
WEIGHTS = "model.safetensors"
INDEX = "model.safetensors.index.json"


def tensor_names(path: Path) -> list[str]:
    """Tensor names from a safetensors header, without loading any data."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def load_scales(path: Path) -> dict[str, float]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"ERROR: safetensors is required to read scales: {exc}")
    out: dict[str, float] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open has no __contains__
            t = f.get_tensor(key).float()
            if t.numel() != 1:
                sys.exit(
                    f"ERROR: {key} has {t.numel()} elements; vLLM's scale "
                    "parameter is scalar")
            out[key] = float(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modeldir", required=True, type=Path)
    ap.add_argument("--scales", type=Path, default=None,
                    help=f"calibrated scales shard (default: <modeldir>/{SCALE_SHARD})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir: Path = args.modeldir
    weights = model_dir / WEIGHTS
    if not weights.is_file():
        # A sharded checkpoint already has an index; calibrate_kv handled it.
        sys.exit(f"ERROR: {weights} not found; this merge is for single-file "
                 "checkpoints (sharded checkpoints are updated by calibrate_kv itself)")
    scales_path = args.scales or (model_dir / SCALE_SHARD)
    if not scales_path.is_file():
        sys.exit(f"ERROR: scales shard not found: {scales_path} "
                 "(run tools/kv_calib/calibrate_kv.py first)")

    scales = load_scales(scales_path)
    if not scales:
        sys.exit(f"ERROR: {scales_path} has no tensors")
    bad = [k for k in scales if not k.endswith(("k_scale", "v_scale", "q_scale",
                                                "prob_scale", "output_scale"))]
    if bad:
        sys.exit(f"ERROR: unexpected scale names (not *_scale): {bad[:5]}")
    print(f"scales: {len(scales)} tensors over "
          f"{len({k.rsplit('.', 2)[0] for k in scales})} attention layers")

    existing = set(tensor_names(weights))
    collide = sorted(set(scales) & existing)
    if collide:
        sys.exit(f"ERROR: scale name already present in {WEIGHTS}: {collide[:5]}")

    # Keep the shard local to the model dir so the index references one place.
    shard_name = SCALE_SHARD
    dest_shard = model_dir / shard_name
    if scales_path.resolve() != dest_shard.resolve():
        if not args.dry_run:
            tmp = dest_shard.with_suffix(".tmp")
            shutil.copy2(scales_path, tmp)
            os.replace(tmp, dest_shard)
        print(f"copied scales -> {dest_shard}")

    index_path = model_dir / INDEX
    if index_path.is_file():
        index = json.loads(index_path.read_text())
    else:
        index = {"metadata": {"format": "pt"}, "weight_map": {}}
    wmap = index.setdefault("weight_map", {})

    # A single-file checkpoint: map every existing tensor to model.safetensors.
    for name in existing:
        wmap.setdefault(name, WEIGHTS)
    # Drop entries that pointed at the shard but are no longer calibrated.
    stale = [n for n, s in list(wmap.items()) if s == shard_name and n not in scales]
    for n in stale:
        del wmap[n]
    new = [n for n in scales if wmap.get(n) != shard_name]
    for name in scales:
        wmap[name] = shard_name

    try:
        total = weights.stat().st_size + dest_shard.stat().st_size
        index.setdefault("metadata", {})["total_size"] = total
    except OSError:
        pass

    if args.dry_run:
        print(f"[dry-run] would write {INDEX}: +{len(new)} new, "
              f"-{len(stale)} stale, {len(wmap)} total")
        return 0

    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    os.replace(tmp, index_path)
    print(f"wrote {INDEX}: +{len(new)} new, -{len(stale)} stale, "
          f"{len(wmap)} total")
    print("verify with: python3 ../verify_kv_scales.py " + str(model_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
