#!/usr/bin/env python3
"""Verify that a Quark MXFP4 checkpoint carries calibrated fp8 KV-cache scales.

Serving runs `--kv-cache-dtype fp8`. If the checkpoint has no k/v scales, vLLM
quantizes the KV cache at scale 1.0 and warns. This checks both halves of the
contract produced by `convert-blend-mxfp4.sh`:

  1. config.json: `export.kv_cache_group` is non-empty and each entry is matched
     by a `layer_quant_config` entry carrying `output_tensors` (the fp8 scheme).
     vLLM's QuarkConfig.from_config raises if this is inconsistent.
  2. safetensors: one `<layer>.self_attn.{k,v}_proj.output_scale` scalar per
     full-attention layer. QuarkConfig.get_cache_scale_mapper maps these to
     `attn.k_scale` / `attn.v_scale`; vLLM then loads them via
     KVCacheScaleParameter (scalar only).

Exit code is non-zero when the checkpoint would serve fp8 KV at scale 1.0, so it
can gate the conversion pipeline.

Usage:
    python3 verify_kv_scales.py [MODEL_DIR]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import json
import sys
from pathlib import Path

# The runtime does NOT divide by the fp8 max (448). vLLM's own calc_kv_scales
# divides q/k/v by Q_/K_/V_SCALE_CONSTANT (200/200/100 in vllm/envs.py), so a
# calibrated checkpoint stores amax/constant and the derived amax is
# scale*constant. These are the defaults; override with the same env vars the
# runtime reads if the serve image changes them.
K_SCALE_CONSTANT = float(os.environ.get("K_SCALE_CONSTANT", "200"))
V_SCALE_CONSTANT = float(os.environ.get("V_SCALE_CONSTANT", "100"))


def load_config(root: Path) -> dict:
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        sys.exit(f"ERROR: {cfg_path} not found")
    return json.loads(cfg_path.read_text())


def check_config(root: Path, cfg: dict) -> list[str]:
    problems: list[str] = []
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "quark":
        problems.append(f"quant_method is {qc.get('quant_method')!r}, expected 'quark'")

    export = qc.get("export") or {}
    kv_group = export.get("kv_cache_group") or []
    print(f"export.kv_cache_group        = {kv_group}")
    print(f"kv_cache_post_rope           = {qc.get('kv_cache_post_rope')}")
    kv_quant_cfg = qc.get("kv_cache_quant_config") or {}
    print(f"kv_cache_quant_config entries= {len(kv_quant_cfg)}")

    if not kv_group:
        # A tensors-only merge (tools/kv_calib/merge_kv_scales.py) is valid:
        # vLLM returns QuarkKVCacheMethod for every non-excluded Attention and
        # create_weights makes the scale params, so the tensors load without a
        # kv_cache_group. The presence of scale tensors below is the real gate.
        print("note: export.kv_cache_group is empty (tensors-only merge); scales "
              "still load through QuarkKVCacheMethod.create_weights")
        return problems

    layer_qc = qc.get("layer_quant_config") or {}

    def _matches_key(kv_pattern: str, key: str) -> bool:
        # Mirror vLLM's loader rule for layer_quant_config keys exactly: a key
        # with `*` is matched with fnmatch against the pattern, a key without `*`
        # matches only when the pattern is a substring of it.
        if "*" in key:
            return fnmatch.fnmatchcase(kv_pattern, key)
        return kv_pattern in key

    matches = {
        pattern: [name for name in layer_qc if _matches_key(pattern, name)]
        for pattern in kv_group
    }
    for pattern, names in matches.items():
        if not names:
            problems.append(
                f"kv_cache_group pattern {pattern!r} has no matching "
                "layer_quant_config entry (vLLM QuarkConfig.from_config will "
                "refuse the checkpoint; fp8_mtp.py must merge, not replace, "
                "layer_quant_config)"
            )
            continue
        out = layer_qc.get(names[0], {}).get("output_tensors")
        if not out:
            problems.append(
                f"layer_quant_config[{names[0]!r}].output_tensors is empty, "
                "expected the fp8_e4m3 per-tensor scheme"
            )
        else:
            dtype = (out.get("dtype") or "").lower()
            qscheme = (out.get("qscheme") or "").lower()
            print(f"  {names[0]:<12} output_tensors dtype={dtype} qscheme={qscheme}")
            if dtype not in ("fp8_e4m3", "fp8"):
                problems.append(f"{names[0]!r} output dtype {dtype!r} is not fp8_e4m3")

    if kv_quant_cfg and not any(kv_quant_cfg):
        problems.append("kv_cache_quant_config is present but all entries are empty")
    return problems


def _confined_shard(root: Path, name: str) -> Path:
    """Resolve a shard name under `root`, rejecting absolute or `..` escapes."""
    if Path(name).is_absolute():
        sys.exit(f"ERROR: {root}: shard name {name!r} is absolute")
    candidate = root / name
    root_resolved = root.resolve()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        sys.exit(f"ERROR: {root}: cannot resolve shard {name!r}: {exc}")
    if not resolved.is_relative_to(root_resolved):
        sys.exit(f"ERROR: {root}: shard {name!r} escapes the model directory")
    return candidate


def iter_safetensors(root: Path):
    # Prefer the index: a merged single-file checkpoint has model.safetensors
    # plus an index that also points at model-kvscales.safetensors, so returning
    # on the single file would miss the scales.
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            yield _confined_shard(root, shard)
        return
    single = root / "model.safetensors"
    if single.is_file():
        yield single
        return
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        sys.exit(f"ERROR: no .safetensors found under {root}")
    yield from shards


def check_scales(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"ERROR: safetensors is required to read scale values: {exc}")

    found: dict[str, float] = {}
    for shard in iter_safetensors(root):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                low = key.lower()
                if not low.endswith(("output_scale", "k_scale", "v_scale")):
                    continue
                if not (".k_proj." in low or ".v_proj." in low
                        or low.endswith(("attn.k_scale", "attn.v_scale"))):
                    continue
                t = f.get_tensor(key).float()
                if t.numel() != 1:
                    problems.append(
                        f"{key}: scale has {t.numel()} elements; vLLM "
                        "KVCacheScaleParameter accepts a scalar only"
                    )
                    continue
                found[key] = float(t)

    print(f"\nfound {len(found)} k/v scale tensor(s)")
    if not found:
        problems.append(
            "no k/v scale tensors in the safetensors: vLLM will serve fp8 KV at 1.0"
        )
        return problems, 0

    by_layer: dict[str, dict[str, float]] = {}
    for key, val in sorted(found.items()):
        layer = key.rsplit(".", 2)[0]
        kind = "k" if ".k_proj." in key or key.endswith("attn.k_scale") else "v"
        by_layer.setdefault(layer, {})[kind] = val

    print(f"{'attention layer':60s} {'k_scale':>12s} {'v_scale':>12s}  {'k_amax':>9s} {'v_amax':>9s}")
    for layer, scales in sorted(by_layer.items()):
        k, v = scales.get("k"), scales.get("v")
        fmt = lambda x: f"{x:.6g}" if x is not None else "-"
        print(f"{layer:60s} {fmt(k):>12s} {fmt(v):>12s}  "
              f"{fmt(k * K_SCALE_CONSTANT if k is not None else None):>9s} "
              f"{fmt(v * V_SCALE_CONSTANT if v is not None else None):>9s}")
        for kind, s in scales.items():
            if s is None:
                continue
            if s <= 0:
                problems.append(f"{layer}: {kind}_scale={s} is not positive")
            if abs(s - 1.0) < 1e-9:
                problems.append(
                    f"{layer}: {kind}_scale is exactly 1.0 -- observer did not run "
                    "(check the [CACHE INTEGRATION] log lines)"
                )
    if len(by_layer) != 16:
        problems.append(
            f"found scales for {len(by_layer)} attention layers, expected 16 "
            "(full_attention indices 3,7,...,63)"
        )
    return problems, len(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="/home/juup/models/Qwen3.8-3.6-27B-blend-MXFP4-mtpfp8",
    )
    args = parser.parse_args()
    root = Path(args.model_dir)
    print(f"checking {root}\n")

    problems = check_config(root, load_config(root))
    scale_problems, _ = check_scales(root)
    problems += scale_problems

    print()
    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: k/v scales present and consistent with the Quark config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
