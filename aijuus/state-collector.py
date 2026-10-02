#!/usr/bin/env python3
"""state-collector: expose the live model state as a Prometheus textfile metric.

Reads (read-only):
  /model-state/state.json      instance -> {model, ready, endpoint, rank}
  /model-state/usage.json      model    -> last-used unix ts (LRU)
  /model-registry.json         model    -> per-model serving config

and writes /textfiles/model_state.prom (atomically) into the shared serve-shape
volume, which node-exporter's textfile collector scrapes. This makes "which model
is loaded on which instance + is it ready + its spec config" visible in Grafana
without any change to node-exporter.

Pure stdlib. Safe to run forever; only rewrites the file when content changes.
"""
import json
import os
import time

STATE_PATH = os.environ.get("STATE_PATH", "/model-state/state.json")
USAGE_PATH = os.environ.get("USAGE_PATH", "/model-state/usage.json")
REGISTRY_PATH = os.environ.get("REGISTRY_PATH", "/model-registry.json")
OUT_PATH = os.environ.get("OUT_PATH", "/textfiles/model_state.prom")
REFRESH_SECONDS = float(os.environ.get("REFRESH_SECONDS", "30"))


def esc(v):
    """Escape a Prometheus label value."""
    s = str(v)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def to_num(v):
    """Coerce registry numeric fields (int/str) to float, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def build():
    state = load_json(STATE_PATH) or {}
    usage = load_json(USAGE_PATH) or {}
    registry = load_json(REGISTRY_PATH) or {}
    defaults = registry.get("_defaults", {}) if isinstance(registry.get("_defaults", {}), dict) else {}
    reg_models = {}
    for k, v in registry.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        merged = dict(defaults)
        merged.update(v)
        reg_models[k] = merged

    lines = []
    lines.append("# HELP radiance_model_state_ready 1 if the instance is ready to serve its loaded model")
    lines.append("# TYPE radiance_model_state_ready gauge")
    for inst, info in state.items():
        if not isinstance(info, dict):
            continue
        model = info.get("model", "unknown")
        ready = 1 if info.get("ready") else 0
        rank = info.get("rank", "")
        endpoint = info.get("endpoint", "")
        lines.append(
            'radiance_model_state_ready{instance="%s",model="%s",rank="%s",endpoint="%s"} %d'
            % (esc(inst), esc(model), esc(rank), esc(endpoint), ready)
        )

    lines.append("# HELP radiance_model_last_used_seconds unix time the model was last used (LRU recency)")
    lines.append("# TYPE radiance_model_last_used_seconds gauge")
    for model, ts in usage.items():
        n = to_num(ts)
        if n is not None:
            lines.append('radiance_model_last_used_seconds{model="%s"} %s' % (esc(model), repr(n)))

    lines.append("# HELP radiance_model_registry_spec_tokens speculative depth configured for the model")
    lines.append("# TYPE radiance_model_registry_spec_tokens gauge")
    lines.append("# HELP radiance_model_registry_max_model_len max context length (tokens)")
    lines.append("# TYPE radiance_model_registry_max_model_len gauge")
    lines.append("# HELP radiance_model_registry_kv_cache_bytes calibrated KV pin (bytes)")
    lines.append("# TYPE radiance_model_registry_kv_cache_bytes gauge")
    lines.append("# HELP radiance_model_registry_verify_head 1 if the int2 verify-head GEMM is enabled")
    lines.append("# TYPE radiance_model_registry_verify_head gauge")
    lines.append("# HELP radiance_model_registry_spec_method one-hot by spec method (dflash|mtp|none)")
    lines.append("# TYPE radiance_model_registry_spec_method gauge")
    for model, cfg in reg_models.items():
        spec = to_num(cfg.get("spec_tokens"))
        if spec is not None:
            lines.append('radiance_model_registry_spec_tokens{model="%s"} %s' % (esc(model), repr(spec)))
        maxlen = to_num(cfg.get("max_model_len"))
        if maxlen is not None:
            lines.append('radiance_model_registry_max_model_len{model="%s"} %s' % (esc(model), repr(maxlen)))
        kv = to_num(cfg.get("kv_cache_memory"))
        if kv is not None:
            lines.append('radiance_model_registry_kv_cache_bytes{model="%s"} %s' % (esc(model), repr(kv)))
        vh = cfg.get("verify_head")
        if isinstance(vh, (int, float, str)) and not isinstance(vh, bool):
            lines.append('radiance_model_registry_verify_head{model="%s"} %s' % (esc(model), 1 if str(vh) == "1" else 0))
        method = cfg.get("spec_method")
        if method:
            lines.append('radiance_model_registry_spec_method{model="%s",method="%s"} 1' % (esc(model), esc(method)))

    lines.append("")
    return "\n".join(lines)


def write_atomic(content):
    d = os.path.dirname(OUT_PATH) or "."
    tmp = os.path.join(d, ".model_state.prom.tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, OUT_PATH)


def main():
    last = None
    while True:
        try:
            content = build()
            if content != last:
                write_atomic(content)
                last = content
        except Exception as e:
            print("state-collector: %r" % (e,), flush=True)
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
