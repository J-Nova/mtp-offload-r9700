#!/usr/bin/env python3
"""Topology-aware engine supervisor for the single dual-card radiance service.

The compose service runs ONE container that owns both GPUs. entrypoint.sh does
the one-time model selection and runtime overlay/patch/prep, then execs this
supervisor. The supervisor reads the selected model's `topology` from
model-registry.json and launches one or two vLLM engines inside this container:

  topology="dp"  -> two TP=1 engines, one per card (ports 8000/8001). This is
                    the MTP/data-parallel shape the 2-GPU deployment shipped.
  topology="tp2" -> one TP=2 engine across both cards (port 8000). This is the
                    int5-paro / large-model shape.

Each engine is an entrypoint.sh invocation in RANK-CHILD mode: it skips model
selection and prep (the parent did both) and only resolves its own serving
shape and execs vLLM. Per-child cache roots keep concurrent cold compiles from
racing (the reason the old per-instance compose used separate cache dirs).

If any engine dies the supervisor tears the rest down and exits non-zero, so the
compose healthcheck fails and the container is restarted; the restart re-reads
state.json and brings up whatever topology the newly selected model wants.
"""
import json
import os
import signal
import subprocess
import sys
import time

ENTRYPOINT = os.environ.get("RADIANCE_ENTRYPOINT",
                            "/patches/aijuus/kv-offload/ops/entrypoint.sh")
# NOTE: do NOT key these off MODEL_REGISTRY_FILE / MODEL_STATE_HOST_DIR -- those
# are HOST paths used by docker-compose bind specs, and Coolify injects them into
# this container too, where they do not exist. Inside the container the repo is
# bind-mounted at /patches and model-state at /model-state (read-only).
REGISTRY = os.environ.get("RADIANCE_REGISTRY_PATH",
                          "/patches/aijuus/model-registry.json")
STATE = os.environ.get("RADIANCE_STATE_PATH", "/model-state/state.json")

# Per-engine compile caches, suffixed per rank/topology under the single /cache
# bind so two cold compiles in one container cannot race on the same output tree.
# AITER_ROOT_DIR is deliberately NOT suffixed: module_aiter_core is compiled into
# the container's site-packages (not /cache), so suffixing it buys nothing and
# could diverge aiter asset resolution between engines.
CACHE_ROOTS = {
    "VLLM_CACHE_ROOT": "/cache/vllm",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
    "TRITON_CACHE_DIR": "/cache/triton",
}


def log(msg):
    sys.stderr.write("[supervisor] %s\n" % msg)
    sys.stderr.flush()


def load_registry():
    with open(REGISTRY) as f:
        full = json.load(f)
    return {k: v for k, v in full.items() if not k.startswith("_")}


def selected_model(reg):
    """Same selection as entrypoint.sh: state.json for this container, then
    MODEL_DEFAULT, then the first registry key."""
    name = os.environ.get("MODEL_DEFAULT")
    try:
        with open(STATE) as f:
            st = json.load(f)
        for inst in ("vllm-0", "vllm-1"):
            e = st.get(inst)
            if e and e.get("model") in reg:
                name = e["model"]
                break
    except Exception:
        pass
    if not name or name not in reg:
        name = sorted(reg)[0]
    return name


def plan(topology):
    """The engine process set for a topology."""
    if topology == "tp2":
        return [{"rank": 0, "port": 8000, "tp": 2, "rocr": "0,1", "hip": "0,1"}]
    # data-parallel: one TP=1 engine per card, one HTTP port each
    return [
        {"rank": 0, "port": 8000, "tp": 1, "rocr": "0", "hip": "0"},
        {"rank": 1, "port": 8001, "tp": 1, "rocr": "1", "hip": "0"},
    ]


def child_env(proc, topology):
    env = dict(os.environ)
    suffix = "-tp2" if topology == "tp2" else "-r%d" % proc["rank"]
    env.update({
        "RADIANCE_RANK_CHILD": "1",
        "SERVE_RANK": str(proc["rank"]),
        "SERVE_PORT": str(proc["port"]),
        "TP": str(proc["tp"]),
        "RADIANCE_TOPOLOGY": topology,
        "ROCR_VISIBLE_DEVICES": proc["rocr"],
        "HIP_VISIBLE_DEVICES": proc["hip"],
    })
    for var, base in CACHE_ROOTS.items():
        env[var] = base + suffix
    return env


def _describe(procs):
    return ", ".join("rank%d:port%d/tp%d" % (p["rank"], p["port"], p["tp"])
                     for p in procs)


def main():
    reg = load_registry()
    model = selected_model(reg)
    topology = (os.environ.get("RADIANCE_TOPOLOGY")
                or (reg.get(model) or {}).get("topology", "dp"))
    procs = plan(topology)
    log("model=%s topology=%s -> %d engine(s): %s"
        % (model, topology, len(procs), _describe(procs)))

    children = []
    for p in procs:
        log("starting engine rank%d on port %d (ROCR=%s HIP=%s TP=%d)"
            % (p["rank"], p["port"], p["rocr"], p["hip"], p["tp"]))
        children.append((p, subprocess.Popen(["bash", ENTRYPOINT],
                                             env=child_env(p, topology))))

    stop = {"flag": False}

    def _term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    rc = 0
    while not stop["flag"]:
        for p, c in children:
            code = c.poll()
            if code is not None:
                log("engine rank%d (port %d) exited rc=%s -- tearing down all engines"
                    % (p["rank"], p["port"], code))
                rc = code or 1
                stop["flag"] = True
                break
        if not stop["flag"]:
            time.sleep(1)

    for _, c in children:
        if c.poll() is None:
            try:
                c.terminate()
            except Exception:
                pass
    deadline = time.time() + 60
    for _, c in children:
        while c.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if c.poll() is None:
            try:
                c.kill()
            except Exception:
                pass
    log("supervisor exiting rc=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
