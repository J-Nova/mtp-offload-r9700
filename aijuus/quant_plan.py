#!/usr/bin/env python3
"""Quantization plan: a single declarative file that drives the Blender Quark
MXFP4 conversion, including the ability to clone another checkpoint's layer
selection ("reference import").

Why this exists
---------------
Layer selection in Quark/vLLM is already data-driven: `global_quant_config`,
`layer_quant_config`, `layer_type_quant_config`, `exclude`, the KV scheme and
`version` all live in `config.json` and are resolved at load time. The converter
hardcoded all of them, so there was no way to say "quantize like checkpoint X" or
"leave MTP bf16" without editing the script.

This tool resolves a small plan into the concrete values the converter passes to
Quark, and the values the MTP rewrite consumes. It is deliberately stdlib-only
(no vLLM import) so it can run in the host venv and in the container.

Plan schema (all fields optional; defaults reproduce the existing blend output)
-------------------------------------------------------------------------------
{
  "reference_model": "/path/to/other/checkpoint",   // import scheme/exclude/version
  "global": {"scheme": "mxfp4", "algo": "qronos", "algo_config_file": "algo-configs/qronos-qwen38.json"},
  "exclude": {"mode": "append", "entries": []},     // append|replace|reference
  "allow_missing_vision_head": false,               // required to let replace drop them
  "mtp": {"mode": "fp8", "module_list": null, "keep_in_exclude": true},
  "kv": {"scheme": "fp8", "post_rope": true, "min_scale": 1e-3},
  "attention": null,                                 // null | "fp8"
  "layers": [],                                      // [{"pattern": "*down_proj", "scheme": "fp8"}]
  "layer_type": {"Linear": "mxfp4"},                 // per-module-type scheme overrides
  "version": "0.13+unknown",
  "retain_algo_config": false
}

`global.algo` accepts one name, a comma string ("gptq,qronos"), or a list; Quark
applies them in order. Qronos already embeds the GPTQ error-diffusion loop, so
`qronos` alone is GPTQ+Qronos. `algo_config_file` is a path (single algo) or an
`{algo: path}` mapping (multi-algo); every non-awq algorithm needs an entry.

`reference_model` imports the global scheme (reconstructed from the serialized
weight dtype, so it may be rejected as ambiguous), the exclude list, and the
version tag. `global.scheme` overrides the imported one.

`mtp.mode`:
  "fp8"  -> leave MTP unquantized in the Quark pass (excluded), then let
            fp8_mtp.py rewrite it to fp8 and drop MTP from the final exclude
            (the current blend behavior).
  "bf16" -> leave MTP unquantized and do not rewrite it; MTP stays bf16 and
            stays excluded in the final config.

Usage
-----
  quant_plan.py resolve --plan P.json [--json-out R.json] [--shell-out R.sh]
  quant_plan.py report  --model DIR
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_defaults import (  # noqa: E402
    DEFAULT_QUANT_EXCLUDE,
    DEFAULT_MTP_EXCLUDE,
    DEFAULT_MTP_KEEP_IN_EXCLUDE,
)

# Quark's _SUPPORTED_SCHEMES (a superset; Quark validates the real list and the
# per-model capability matrix at conversion time). Kept here only so an obvious
# typo fails before a multi-hour run.
KNOWN_SCHEMES = {
    "fp8", "ptpc_fp8", "int8", "int4_wo_32", "int4_wo_64", "int4_wo_128",
    "int4_wo_per_channel", "uint4_wo_32", "uint4_wo_64", "uint4_wo_128",
    "uint4_wo_per_channel", "mxfp4", "mxfp6_e3m2", "mxfp6_e2m3", "mx6",
    "bfp16", "int4_fp8", "nvfp4", "fp4_block16_scale_e4m3",
}
# Quark's ALGORITHM_CONFIG_MAPS names (get_supported_algorithm_types()).
KNOWN_ALGOS = {"none", "awq", "gptq", "gptaq", "qronos", "smoothquant",
               "autosmoothquant", "rotation"}
# A non-awq algorithm has no built-in config for qwen3_5 in Quark, so an explicit
# algo_config_file is required. awq/none work with the converter's default recipe.
ALGOS_NEEDING_CONFIG = KNOWN_ALGOS - {"none", "awq"}
# Quark's layer_type_config is keyed by nn.Module subclasses; the plan carries
# names. Only these survive Quark's export/import round-trip (config.py restricts
# from_dict to Conv2d/Linear/ConvTranspose2d).
LAYER_TYPE_NAMES = {"Linear", "Conv2d", "ConvTranspose2d"}
MTP_MODES = {"fp8", "bf16"}


def _read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _quant_config(model_dir: str | Path) -> dict:
    cfg = _read_json(Path(model_dir) / "config.json")
    return cfg.get("quantization_config") or {}


def _scheme_from_serialized(weight: dict) -> str | None:
    """Map a serialized weight spec back to a Quark scheme name.

    Quark exports the resolved Dtype enum (e.g. "fp4"), not the scheme name, so a
    reference import must reconstruct the scheme. Ambiguous specs return None and
    the caller requires an explicit global.scheme.
    """
    dtype = str(weight.get("dtype") or "").lower()
    qscheme = str(weight.get("qscheme") or "").lower()
    group = weight.get("group_size")
    scale_format = str(weight.get("scale_format") or "").lower()
    if dtype in ("fp8_e4m3", "fp8"):
        return "fp8"
    if dtype == "int8":
        return "int8"
    if dtype in ("int4", "uint4"):
        if qscheme == "per_channel":
            return f"{dtype}_wo_per_channel"
        if qscheme == "per_group" and group in (32, 64, 128):
            return f"{dtype}_wo_{group}"
        return None
    if dtype == "fp4":
        if scale_format == "e8m0" and group == 32:
            return "mxfp4"
        if qscheme == "per_group" and group == 16:
            return "nvfp4"
        return None
    return None


def load_reference(model_dir: str | Path) -> dict:
    """Pull the layer-selection fields out of an existing checkpoint.

    Only the fields ``resolve`` consumes are returned: the global scheme (mapped
    from the serialized dtype), the exclude list and the version tag.
    """
    qc = _quant_config(model_dir)
    if qc.get("quant_method") != "quark":
        raise SystemExit(
            f"ERROR: reference {model_dir} quant_method={qc.get('quant_method')!r}, "
            "only quark checkpoints can be imported")
    weight = (qc.get("global_quant_config") or {}).get("weight") or {}
    return {
        "version": qc.get("version"),
        "global_scheme": _scheme_from_serialized(weight),
        "exclude": list(qc.get("exclude") or []),
    }


def _merge_exclude(plan: dict, reference: dict | None) -> list[str]:
    spec = plan.get("exclude") or {}
    mode = spec.get("mode", "append")
    entries = list(spec.get("entries") or [])
    if mode == "replace":
        base: list[str] = []
    elif mode == "reference":
        if reference is None:
            raise SystemExit("ERROR: exclude.mode=reference requires reference_model")
        base = list(reference["exclude"])
    else:  # append
        base = list(DEFAULT_QUANT_EXCLUDE)
    out = base + [e for e in entries if e not in base]
    return out


def _apply_mtp(exclude: list[str], mtp: dict, reference: dict | None,
               explicit_entries: list[str]) -> tuple[list[str], list[str], bool]:
    """Return (quant_time_exclude, final_mtp_entries, rewrite_mtp).

    MTP is always pulled out of the body quantization so Quark leaves it bf16;
    whether it is then rewritten to fp8 (and whether the final config keeps a
    Standard-style MTP set in exclude) is the MTP policy.

    `final_mtp_entries` is only used when mtp.keep_in_exclude is true. With a
    reference it is the reference's own (weight-scoped, load-inert) entries; with
    no reference it is the canonical Standard set, so the default final config
    matches AMD's checkpoint.
    """
    mode = mtp.get("mode", "fp8")
    if mode not in MTP_MODES:
        raise SystemExit(f"ERROR: mtp.mode={mode!r}, expected one of {sorted(MTP_MODES)}")
    module_list = mtp.get("module_list")
    out = [e for e in exclude if "mtp" not in e]
    for p in (module_list or ["mtp.*"]):
        if p not in out:
            out.append(p)
    keep = bool(mtp.get("keep_in_exclude", DEFAULT_MTP_KEEP_IN_EXCLUDE))
    entries: list[str] = []
    if keep:
        if reference is not None:
            entries = [e for e in exclude if "mtp" in e]
        else:
            entries = list(DEFAULT_MTP_EXCLUDE)
        for e in explicit_entries:
            if "mtp" in e and e not in entries:
                entries.append(e)
    return out, entries, mode == "fp8"


# vLLM fuses these at load time and requires every shard of a fused module to use
# the same scheme (quark/utils.py deep_compare). A plan that assigns q_proj/k_proj/
# v_proj (-> qkv_proj) or gate_proj/up_proj (-> gate_up_proj) different schemes in
# the same scope would raise at load, so reject it before the run.
_FUSED_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("gate_proj", "up_proj"),
)


def _fused_conflicts(layers: list[list[str]]) -> list[str]:
    problems = []
    for group in _FUSED_GROUPS:
        scope_scheme: dict[tuple, str] = {}
        for pattern, scheme in layers:
            for member in group:
                if pattern.endswith(member):
                    scope = pattern[: -len(member)]
                    key = (group, scope)
                    prev = scope_scheme.get(key)
                    if prev is not None and prev != scheme:
                        problems.append(
                            f"fused module under {scope!r}: {group} resolve to "
                            f"different schemes ({prev!r} vs {scheme!r}); vLLM "
                            "requires every shard of a fused module to match")
                    else:
                        scope_scheme[key] = scheme
    return problems


def resolve(plan: dict) -> tuple[dict, list[str]]:
    problems: list[str] = []
    reference = None
    ref_path = plan.get("reference_model")
    if ref_path:
        if not (Path(ref_path) / "config.json").is_file():
            raise SystemExit(f"ERROR: reference_model {ref_path} has no config.json")
        reference = load_reference(ref_path)

    global_spec = plan.get("global") or {}
    ref_scheme = (reference or {}).get("global_scheme")
    scheme = global_spec.get("scheme") or ref_scheme
    if scheme is None:
        if reference is not None and not global_spec.get("scheme"):
            problems.append(
                "cannot infer global.scheme from reference_model: its serialized "
                "weight dtype is ambiguous; set global.scheme explicitly")
        scheme = "mxfp4"
    # global.algo accepts one name, a comma string, or a list. Quark applies a
    # comma-separated list sequentially (algorithm/api.py "apply algorithms
    # sequentially"). Note Qronos ALREADY embeds the GPTQ error-diffusion loop
    # (qronos.py "GPTQ loop ... error diffused W"), so `qronos` alone is
    # GPTQ+Qronos; only use an explicit "gptq,qronos" if you want a separate GPTQ
    # pass first instead of/in addition to Qronos's own.
    raw_algo = global_spec.get("algo", "awq")
    if isinstance(raw_algo, str):
        algos = [a.strip() for a in raw_algo.split(",") if a.strip()]
    elif isinstance(raw_algo, list):
        algos = [str(a).strip() for a in raw_algo if str(a).strip()]
    else:
        raise SystemExit(
            f"ERROR: global.algo must be a string or list, got {type(raw_algo).__name__}")
    if not algos:
        algos = ["awq"]
    raw_cfg = global_spec.get("algo_config_file")
    if isinstance(raw_cfg, dict):
        algo_files = {str(k).lower(): v for k, v in raw_cfg.items()}
    elif isinstance(raw_cfg, str):
        algo_files = {algos[0].lower(): raw_cfg}
    elif raw_cfg is None:
        algo_files = {}
    else:
        raise SystemExit(
            "ERROR: global.algo_config_file must be a path or an {algo: path} mapping")

    exclude_spec = plan.get("exclude") or {}
    exclude = _merge_exclude(plan, reference)
    mtp = plan.get("mtp") or {}
    explicit_exclude = list(exclude_spec.get("entries") or [])
    quant_exclude, final_mtp_entries, rewrite_mtp = _apply_mtp(
        exclude, mtp, reference, explicit_exclude)

    # exclude.mode=replace bypasses the built-in vision/lm_head guard. The vision
    # tower is CPU-placed and never calibrated, and lm_head must stay unquantized;
    # quantizing either silently produces a bad checkpoint after a multi-hour run.
    if exclude_spec.get("mode") == "replace" and not plan.get("allow_missing_vision_head"):
        missing_guards = [p for p in ("model.visual", "lm_head")
                          if not any(p in e for e in quant_exclude)]
        if missing_guards:
            problems.append(
                "exclude.mode=replace drops the required "
                f"{missing_guards} exclusion(s); add them to exclude.entries or set "
                "allow_missing_vision_head=true to proceed")

    version = plan.get("version") or (reference or {}).get("version")
    kv = plan.get("kv") or {}
    kv_scheme = kv.get("scheme", "fp8")
    attention = plan.get("attention")

    # Per-name scheme overrides -> Quark --layer_quant_scheme PATTERN SCHEME.
    layer_schemes: list[list[str]] = []
    raw_layers = plan.get("layers") or []
    if not isinstance(raw_layers, list):
        raise SystemExit("ERROR: plan.layers must be a list of {pattern, scheme}")
    for ent in raw_layers:
        if not isinstance(ent, dict) or "pattern" not in ent or "scheme" not in ent:
            problems.append(f"layers entry must be {{pattern, scheme}}: {ent!r}")
            continue
        pat, sch = str(ent["pattern"]), str(ent["scheme"])
        if sch not in KNOWN_SCHEMES:
            problems.append(f"layers[{pat!r}].scheme {sch!r} is not a known Quark scheme")
        layer_schemes.append([pat, sch])
    if layer_schemes:
        problems += _fused_conflicts(layer_schemes)

    # Per-module-type overrides -> LLMTemplate._set_layer_type_config. quantize_quark
    # never forwards layer_type_config, so the wrapper injects it from
    # PLAN_LAYER_TYPE_SCHEMES after get_config returns.
    layer_type_schemes: list[list[str]] = []
    raw_types = plan.get("layer_type") or {}
    if not isinstance(raw_types, dict):
        problems.append("plan.layer_type must be an object mapping module type -> scheme")
    else:
        for tname, sch in raw_types.items():
            tname, sch = str(tname), str(sch)
            if tname not in LAYER_TYPE_NAMES:
                problems.append(
                    f"layer_type {tname!r} is not one of {sorted(LAYER_TYPE_NAMES)} "
                    "(Quark's export/import only round-trips these)")
            if sch not in KNOWN_SCHEMES:
                problems.append(f"layer_type[{tname!r}].scheme {sch!r} is not a known Quark scheme")
            layer_type_schemes.append([tname, sch])

    # Runtime-ignore rewrite (the image's --overrideLayers analogue): a layer
    # targeted by `layers` must not stay in the exclude list, or vLLM would load
    # it unquantized (and fail on packed weights). Drop exact exclude entries the
    # layer patterns cover; a regex exclude that could also cover one is a problem.
    exclude_rewrites: list[str] = []
    if layer_schemes:
        kept = []
        for e in quant_exclude:
            if e.startswith("re:"):
                if any(re.search(e[3:], pat) for pat, _ in layer_schemes):
                    problems.append(
                        f"exclude {e!r} is a regex that may also cover a layer targeted "
                        "by plan.layers; narrow it")
                kept.append(e)
                continue
            hit = any(
                fnmatch.fnmatchcase(e, pat) or ("*" not in pat and pat in e)
                for pat, _ in layer_schemes
            )
            if hit:
                exclude_rewrites.append(e)
            else:
                kept.append(e)
        quant_exclude = kept
        if any("mtp" in pat for pat, _ in layer_schemes) and any("mtp" in e for e in exclude):
            if mtp.get("mode", "fp8") == "fp8":
                problems.append("plan.layers targets mtp.* while mtp.mode=fp8; "
                                "use mtp.mode to control MTP instead")

    if scheme not in KNOWN_SCHEMES:
        problems.append(f"global.scheme {scheme!r} is not a known Quark scheme")
    for a in algos:
        if a not in KNOWN_ALGOS:
            problems.append(f"global.algo {a!r} is not one of {sorted(KNOWN_ALGOS)}")
        elif a in ALGOS_NEEDING_CONFIG and not algo_files.get(a):
            problems.append(
                f"global.algo {a!r} needs an algo_config_file entry: Quark has no "
                "built-in config for qwen3_5 (see algo-configs/)")
        if a == "none" and algo_files.get(a):
            problems.append("global.algo_config_file is set but algo=none")
        if a == "gptq" and scheme == "mxfp4" and "qronos" not in algos:
            problems.append(
                "global.algo gptq with scheme mxfp4: Quark documents GPTQ as "
                "int4/uint4 per-group. Use qronos (it already embeds the GPTQ "
                "error-diffusion loop and supports mxfp4), or add it as "
                "\"gptq,qronos\" so the qronos pass handles mxfp4")
        if a == "qronos" and scheme not in {"mxfp4", "int4_wo_32", "int4_wo_64",
                                            "int4_wo_128", "uint4_wo_32",
                                            "uint4_wo_64", "uint4_wo_128"}:
            problems.append(
                f"global.algo qronos with scheme {scheme!r}: Qronos supports "
                "int3/int4/uint4/mxfp4 per-group only")
    if kv_scheme != "fp8":
        problems.append(
            f"kv.scheme {kv_scheme!r}: vLLM's Quark KV path implements only "
            "fp8_e4m3 per_tensor (quark.py:776-788)")
    if attention not in (None, "fp8"):
        problems.append(f"attention {attention!r}: Quark's template supports only 'fp8'")

    resolved = {
        "plan_scheme": scheme,
        "plan_algo": ",".join(algos),
        "plan_algo_config_files": [
            [a, algo_files[a]] for a in algos if a != "none" and algo_files.get(a)
        ],
        "plan_exclude": quant_exclude,
        "plan_layer_schemes": layer_schemes,
        "plan_layer_type_schemes": layer_type_schemes,
        "plan_exclude_rewrites": exclude_rewrites,
        "plan_version": version,
        "plan_kv_scheme": kv_scheme,
        "plan_kv_post_rope": bool(kv.get("post_rope", True)),
        "plan_kv_min_scale": kv.get("min_scale", 1e-3),
        "plan_attention": attention,
        "plan_mtp_mode": mtp.get("mode", "fp8"),
        "plan_mtp_rewrite": rewrite_mtp,
        "plan_mtp_module_list": mtp.get("module_list"),
        "plan_mtp_keep_in_exclude": bool(mtp.get("keep_in_exclude", DEFAULT_MTP_KEEP_IN_EXCLUDE)),
        "plan_final_mtp_entries": final_mtp_entries,
        "plan_retain_algo_config": bool(plan.get("retain_algo_config", False)),
    }
    return resolved, problems


def _shell_assignments(resolved: dict, json_out: str) -> str:
    lines = [f"PLAN_QUANT_PLAN_RESOLVED={shlex.quote(json_out)}"]
    for key, val in resolved.items():
        env_key = key.upper()
        if val is None:
            quoted = "''"
        elif isinstance(val, list):
            quoted = shlex.quote(json.dumps(val))
        else:
            quoted = shlex.quote(str(val))
        lines.append(f"{env_key}={quoted}")
    return "\n".join(lines) + "\n"


def cmd_resolve(args: argparse.Namespace) -> int:
    plan = _read_json(Path(args.plan))
    resolved, problems = resolve(plan)
    if problems:
        print("Plan validation failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out_json = args.json_out or (str(Path(args.plan).with_suffix(".resolved.json")))
    Path(out_json).write_text(json.dumps({"plan": plan, "resolved": resolved}, indent=2))
    shell_out = args.shell_out or (out_json + ".sh")
    Path(shell_out).write_text(_shell_assignments(resolved, out_json))

    print(f"Resolved plan -> {out_json}")
    print(f"  scheme={resolved['plan_scheme']} algo={resolved['plan_algo']} "
          f"version={resolved['plan_version']}")
    print(f"  quant exclude ({len(resolved['plan_exclude'])}): "
          f"{resolved['plan_exclude'][:6]}{' ...' if len(resolved['plan_exclude']) > 6 else ''}")
    print(f"  mtp mode={resolved['plan_mtp_mode']} rewrite={resolved['plan_mtp_rewrite']} "
          f"keep_in_exclude={resolved['plan_mtp_keep_in_exclude']}")
    print(f"  shell env -> {shell_out}")
    return 0


# --- report -----------------------------------------------------------------
# Reimplements vLLM's Quark selection rules without importing vLLM:
#   exclude: exact equality or `re:` prefix (quark/utils.py:97-123)
#   layer_quant_config: fnmatch if the pattern has '*', else substring match
#     (quark.py:604-611)
#   fallback: global_quant_config (quark.py:620-623)
def _excluded(name: str, exclude: list[str]) -> bool:
    for t in exclude:
        if t.startswith("re:"):
            if re.match(t[3:], name):
                return True
        elif t == name:
            return True
    return False


def _matches_config(name: str, patterns: dict) -> str | None:
    # vLLM's rule (quark.py `_matches_pattern`): a pattern with `*` is matched
    # against the full layer name with fnmatch; a pattern without `*` matches only
    # when the layer name is a substring OF the pattern (`layer_name in pattern`).
    for pat, cfg in patterns.items():
        if "*" in pat:
            if fnmatch.fnmatchcase(name, pat):
                return pat
        elif name in pat:
            return pat
    return None


def _confined_shard(root: Path, name: str) -> Path:
    """Resolve a shard name under `root`, rejecting absolute or `..` escapes.

    Shard names come from `model.safetensors.index.json`, which is untrusted
    checkpoint metadata; joining without confinement would let a crafted index
    point at any file on disk.
    """
    if Path(name).is_absolute():
        raise SystemExit(f"ERROR: {root}: shard name {name!r} is absolute")
    candidate = root / name
    root_resolved = root.resolve()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise SystemExit(f"ERROR: {root}: cannot resolve shard {name!r}: {exc}")
    if not resolved.is_relative_to(root_resolved):
        raise SystemExit(f"ERROR: {root}: shard {name!r} escapes the model directory")
    return candidate


def _enum_layers(model_dir: Path) -> list[str]:
    """Module prefixes that carry a quantizable 2-D weight, from the headers."""
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    # The index wins when present: a merged checkpoint keeps the single file and
    # adds model-kvscales.safetensors through the index.
    shards = []
    if index.is_file():
        wm = json.loads(index.read_text())["weight_map"]
        shards = [_confined_shard(model_dir, s) for s in sorted(set(wm.values()))]
    elif single.is_file():
        shards = [single]
    if not shards:
        shards = sorted(model_dir.glob("*.safetensors"))
    names: set[str] = set()
    for shard in shards:
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for key, meta in header.items():
            # Packed MXFP4 scales are named `<...>.weight_scale` and are dropped
            # by this filter; only real 2-D weight tensors are enumerated.
            if not key.endswith(".weight"):
                continue
            if len(meta.get("shape", [])) != 2:
                continue
            names.add(key[: -len(".weight")])
    return sorted(names)


def cmd_report(args: argparse.Namespace) -> int:
    model = Path(args.model)
    qc = _quant_config(model)
    exclude = list(qc.get("exclude") or [])
    layer_qc = qc.get("layer_quant_config") or {}
    global_dtype = (qc.get("global_quant_config") or {}).get("weight", {}).get("dtype")
    print(f"{model}")
    print(f"  global weight dtype: {global_dtype}; exclude entries: {len(exclude)}")
    print(f"  {'layer':62s} {'effective':>28s}")
    counts: dict[str, int] = {}
    for name in _enum_layers(model):
        if _excluded(name, exclude):
            eff = "EXCLUDED"
        else:
            pat = _matches_config(name, layer_qc)
            if pat is None:
                eff = f"global:{global_dtype}"
            else:
                cfg = layer_qc[pat] or {}
                w = (cfg.get("weight") or {})
                eff = f"{pat} ({w.get('dtype')}/{w.get('qscheme')})"
        counts[eff] = counts.get(eff, 0) + 1
        if args.verbose:
            print(f"  {name:62s} {eff:>28s}")
    print("  summary:")
    for eff, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {eff}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_res = sub.add_parser("resolve", help="resolve a plan into converter values")
    p_res.add_argument("--plan", required=True)
    p_res.add_argument("--json-out")
    p_res.add_argument("--shell-out")
    p_res.set_defaults(func=cmd_resolve)

    p_rep = sub.add_parser("report", help="print a checkpoint's effective selection")
    p_rep.add_argument("--model", required=True)
    p_rep.add_argument("--verbose", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
