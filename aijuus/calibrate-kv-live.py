#!/usr/bin/env python3
"""calibrate-kv-live.py -- measure a KV-cache pin for THIS deployment.

Upstream `calibrate-kv.sh` drives `serve-mxfp4.sh`, which is pinned to the stock
`stilldeadcode/vllm-radiance:0.9.3` image lineage. Our deployment diverged: the
stock image's `mamba/abstract.py` lacks the lazy-GDN anchor, and our
`juupp/vllm-radiance:0.9.3-collect-tokens` image no longer ships
`radiance_allreduce.py` (so patch_ar_maxbytes hard-fails under that launcher's
`set -e`). Neither image can be calibrated through it.

This harness drives the REAL stack instead: it edits `kv_cache_memory` in
aijuus/model-registry.json, restarts the target instance through the
model-controller's POST /reload (so the router stops routing to it while it
churns), polls /health, and runs the same PASS test upstream uses -- one
CHUNK-sized prefill plus a short decode. The chosen pin is written back.

IMPORTANT DIFFERENCES FROM calibrate-kv.sh
  * It does NOT profile. vLLM's profiler charges the transient/cudagraph peak,
    so on this model it sizes ~4.05 GiB, below the ~5.45 GiB needed for
    max_model_len=160000, and the engine refuses to boot. A pin must start from a
    value known to serve (the current registry pin, or --start).
  * It FAILS FAST and RESCUES. A pin that cannot boot makes the container
    crash-loop; the harness detects the engine-start failure in the logs and you
    should not be left waiting on the controller's 3600 s timeout. On any
    failure it restores the last known-good pin and restarts, so the instance is
    never left down.

A reload RESTARTS the target instance -- a deployment op: run it yourself.

  ./aijuus/calibrate-kv-live.py                 raise from the current pin
  ./aijuus/calibrate-kv-live.py --quick         verify the current pin only
  ./aijuus/calibrate-kv-live.py --dry-run       print the plan, change nothing
  ./aijuus/calibrate-kv-live.py --no-reload     probe the current server only
  ./aijuus/calibrate-kv-live.py --pins 6.5e9,6.8e9,7.0e9   explicit pins
  ./aijuus/calibrate-kv-live.py --start 6.0e9   start from this pin

Env: MODEL_KEY, INSTANCE, CONTROLLER, CONTROLLER_SERVICE, RELOAD_PORT,
VLLM_API_KEY, VLLM_BASE, MODEL_REGISTRY_FILE, CHUNK, START, STEP, MAX_STEPS,
BACKOFF_STEPS, BOOT_TIMEOUT.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_KEY = os.environ.get("MODEL_KEY", "mtp-27B-MXFP4-blend")
INSTANCE = os.environ.get("INSTANCE", "vllm-0")
CONTROLLER = os.environ.get("CONTROLLER", "")
CONTROLLER_SERVICE = os.environ.get("CONTROLLER_SERVICE", "model-controller")
RELOAD_PORT = int(os.environ.get("RELOAD_PORT", "8101"))
API_KEY = os.environ.get("VLLM_API_KEY", "juup-123")
REGISTRY = Path(os.environ.get("MODEL_REGISTRY_FILE", str(HERE / "model-registry.json")))
INSTANCE_PORT = {"vllm-0": 8000, "vllm-1": 8001}
STEP = float(os.environ.get("STEP", "0.02"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))
BACKOFF_STEPS = int(os.environ.get("BACKOFF_STEPS", "1"))
BOOT_TIMEOUT = int(os.environ.get("BOOT_TIMEOUT", "900"))     # per-attempt boot wait
RELOAD_TIMEOUT = int(os.environ.get("RELOAD_TIMEOUT", "1800"))  # final apply wait
MODEL_DEFAULT_MAXLEN = 160000
GIB = 1 << 30

# A boot that contains any of these never becomes a serving pin -- fail fast.
FATAL = re.compile(
    r"Engine core initialization failed|larger than the available KV cache memory"
    r"|OutOfMemoryError|out of memory|hipErrorOutOfMemory|EngineCore failed to start"
    r"|estimated maximum model length is", re.I)


def say(msg):
    print("[calibrate-kv-live] %s" % msg, flush=True)


def die(msg, *hints):
    print("[calibrate-kv-live] ERROR: %s" % msg, file=sys.stderr)
    for h in hints:
        print("  %s" % h, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- http
def _req(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if API_KEY:
        req.add_header("Authorization", "Bearer " + API_KEY)
    return urllib.request.urlopen(req, timeout=timeout)


def http_status(method, url, body=None, timeout=30):
    try:
        with _req(method, url, body, timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 0, b""


def get_json(method, url, body=None, timeout=30):
    st, raw = http_status(method, url, body, timeout)
    try:
        return st, json.loads(raw or b"{}")
    except Exception:
        return st, {}


# --------------------------------------------------------------------------- docker
def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


def container_id(instance):
    r = sh("docker", "ps", "-a", "--filter",
           "label=com.docker.compose.service=%s" % instance, "--format", "{{.ID}}")
    return next((l for l in r.stdout.split() if l.strip()), None)


def container_networks(cid):
    if not cid:
        return {}
    r = sh("docker", "inspect", "-f",
           "{{range $n,$v := .NetworkSettings.Networks}}{{$n}}={{$v.IPAddress}} {{end}}", cid)
    out = {}
    for tok in r.stdout.split():
        n, _, a = tok.partition("=")
        if a:
            out[n] = a
    return out


def started_at(instance):
    """Docker's .RestartCount does NOT move on a manual `docker restart`; the
    container's StartedAt does, so it is the reliable 'has it rebooted?' signal."""
    cid = container_id(instance)
    if not cid:
        return ""
    return sh("docker", "inspect", "-f", "{{.State.StartedAt}}", cid).stdout.strip()


def instance_base(instance):
    override = os.environ.get("VLLM_BASE")
    if override:
        return override.rstrip("/")
    cid = container_id(instance)
    if not cid:
        die("no container for service %r (is the compose up?)" % instance)
    ip = next((a for a in container_networks(cid).values() if a), None)
    if not ip:
        die("could not resolve an IP for %s" % instance)
    return "http://%s:%d" % (ip, INSTANCE_PORT.get(instance, 8000))


def controller_base():
    if CONTROLLER:
        return CONTROLLER.rstrip("/")
    cid = container_id(CONTROLLER_SERVICE)
    if not cid:
        die("no container for controller service %r" % CONTROLLER_SERVICE)
    nets = container_networks(cid)
    inst_nets = container_networks(container_id(INSTANCE))
    ip = next((a for n, a in nets.items() if n in inst_nets), None) or \
        next((a for a in nets.values() if a), None)
    if not ip:
        die("could not resolve an IP for the controller")
    return "http://%s:%d" % (ip, RELOAD_PORT)


def logs_since(instance, since):
    cid = container_id(instance)
    if not cid:
        return ""
    r = subprocess.run(["docker", "logs", "--since", since, "--tail", "800", cid],
                       capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def fatal_seen(instance, since):
    return bool(FATAL.search(logs_since(instance, since)))


def read_boot_facts(instance, since):
    t = logs_since(instance, since)
    toks = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", t)
    avail = re.findall(r"Available KV cache memory:\s*([\d.]+)\s*GiB", t)
    reserved = re.findall(r"reserved\s*([\d.]+)\s*GiB memory for KV", t)
    return {
        "tokens": int(toks[-1].replace(",", "")) if toks else None,
        "available_gib": float(avail[-1]) if avail else None,
        "reserved_gib": float(reserved[-1]) if reserved else None,
        "oom": bool(re.search(r"out of memory|hipErrorOutOfMemory", t, re.I)),
    }


def docker_restart(instance):
    cid = container_id(instance)
    if not cid:
        return False
    return sh("docker", "restart", cid).returncode == 0


# --------------------------------------------------------------------------- registry
def load_reg():
    with open(REGISTRY) as f:
        return json.load(f)


def save_reg(reg):
    tmp = str(REGISTRY) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=1)
        f.write("\n")
    os.replace(tmp, REGISTRY)


def set_pin(reg, value):
    reg[MODEL_KEY]["kv_cache_memory"] = value


def current_pin(reg):
    v = reg[MODEL_KEY].get("kv_cache_memory")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- reload
def controller_reload_async(controller):
    """POST /reload off-thread: the controller blocks until healthy (or its own
    SWAP_TIMEOUT), and we do not want to block on that when a pin cannot boot."""
    def _go():
        try:
            get_json("POST", controller + "/reload",
                     {"model": MODEL_KEY, "instance": INSTANCE}, timeout=3600)
        except Exception:
            pass
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


def wait_healthy(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_status("GET", base + "/health", timeout=5)[0] == 200:
            return True
        time.sleep(4)
    return False


def rescue(good_pin, reg, base):
    """Restore the last known-good pin and bring the instance back, so a failed
    attempt never leaves the server crash-looping."""
    say("  rescuing: restoring %d bytes and restarting %s" % (good_pin, INSTANCE))
    set_pin(reg, int(good_pin))
    save_reg(reg)
    docker_restart(INSTANCE)
    ok = wait_healthy(base, BOOT_TIMEOUT)
    say("  rescue %s" % ("OK -- server healthy again" if ok else "FAILED to recover"))
    return ok


def probe(base, model, chunk, timeout=600):
    reps = max(16, int(chunk * 1.05 / 11))  # ~11 tokens per 9-word phrase
    prompt = "the quick brown fox jumps over the lazy dog. " * reps
    body = {"model": model, "prompt": prompt, "max_tokens": 32, "temperature": 0}
    return http_status("POST", base + "/v1/completions", body, timeout=timeout)[0]


def served_name(reg):
    return reg[MODEL_KEY].get("served_name") or MODEL_KEY


def attempt(pin, chunk, reg, controller, base, good_pin):
    """Set pin, reload+restart, wait for the NEW boot, probe. Returns a result dict.

    Fast-fails when the new boot logs an engine-start failure, and rescues
    (restores good_pin) so the instance is left serving.
    """
    set_pin(reg, "" if pin is None else int(pin))
    save_reg(reg)
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    st0 = started_at(INSTANCE)
    if controller:
        controller_reload_async(controller)
    else:
        docker_restart(INSTANCE)

    # Phase 1: wait for the instance to actually restart (the controller drains
    # first, so the OLD pin keeps answering /health for a while -- don't trust it).
    deadline = time.time() + BOOT_TIMEOUT
    restarted = False
    while time.time() < deadline:
        if fatal_seen(INSTANCE, since):
            break
        if started_at(INSTANCE) != st0:
            restarted = True
            break
        time.sleep(2)

    if fatal_seen(INSTANCE, since):
        return {"pin": pin, "pass": False, "reason": "engine failed to start",
                "facts": read_boot_facts(INSTANCE, since)}
    if not restarted:
        # no controller restart observed -- force one so we can assess the pin
        docker_restart(INSTANCE)
        if not wait_healthy(base, BOOT_TIMEOUT):
            return {"pin": pin, "pass": False, "reason": "never restarted / unhealthy",
                    "facts": read_boot_facts(INSTANCE, since)}

    # Phase 2: assess the new boot (fresh log window from the restart).
    since2 = time.strftime("%Y-%m-%dT%H:%M:%S")
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        if fatal_seen(INSTANCE, since2):
            return {"pin": pin, "pass": False, "reason": "engine failed to start",
                    "facts": read_boot_facts(INSTANCE, since2)}
        if http_status("GET", base + "/health", timeout=5)[0] == 200:
            code = probe(base, served_name(reg), chunk)
            facts = read_boot_facts(INSTANCE, since2)
            if code != 200:
                return {"pin": pin, "pass": False, "reason": "prefill probe HTTP %s" % code,
                        "facts": facts}
            return {"pin": pin, "pass": True, "facts": facts}
        time.sleep(3)
    return {"pin": pin, "pass": False, "reason": "boot timeout",
            "facts": read_boot_facts(INSTANCE, since2)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="measure a KV pin for the live deployment")
    ap.add_argument("--quick", action="store_true", help="verify the start pin only, do not raise")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ap.add_argument("--no-reload", action="store_true", help="probe the running server only")
    ap.add_argument("--pins", default="", help="explicit comma-separated pins in bytes")
    ap.add_argument("--start", default=os.environ.get("START", ""), help="starting pin in bytes")
    args = ap.parse_args()

    if not REGISTRY.exists():
        die("registry not found: %s" % REGISTRY)
    reg = load_reg()
    if MODEL_KEY not in reg:
        die("model %r not in %s" % (MODEL_KEY, REGISTRY))
    entry = reg[MODEL_KEY]
    chunk = int(os.environ.get("CHUNK") or entry.get("max_num_batched_tokens") or 4096)
    orig_text = REGISTRY.read_text()
    good0 = current_pin(reg)

    say("registry:  %s" % REGISTRY)
    say("model:     %s (served as %r)" % (MODEL_KEY, served_name(reg)))
    say("instance:  %s @ %s" % (INSTANCE, instance_base(INSTANCE) if container_id(INSTANCE) else "(not running)"))
    say("controller:%s" % (CONTROLLER or "(auto-discover on the compose net)"))
    say("shape:     chunk=%d maxlen=%d spec=%s" % (chunk, int(entry.get("max_model_len") or MODEL_DEFAULT_MAXLEN),
                                                   entry.get("spec_tokens", "?")))
    say("current:   kv_cache_memory=%s" % entry.get("kv_cache_memory"))
    say("step:      %+.0f%% x %d, backoff %d, boot-timeout %ds  (reload restarts %s)"
        % (STEP * 100, MAX_STEPS, BACKOFF_STEPS, BOOT_TIMEOUT, INSTANCE))

    if args.dry_run:
        start = args.start or (str(good0) if good0 else "<none set>")
        say("start:     %s" % start)
        say("pass 1: verify the start pin serves a CHUNK-sized prefill")
        say("pass 2: raise by %+.0f%% up to %dx, stop at first failure, back off %d step(s)"
            % (STEP * 100, MAX_STEPS, BACKOFF_STEPS))
        return

    explicit = [int(float(x)) for x in args.pins.split(",") if x.strip()]
    if args.start:
        explicit = [int(float(args.start))] + explicit

    base = instance_base(INSTANCE) if container_id(INSTANCE) else die("instance not running")

    if args.no_reload:
        code = probe(base, served_name(reg), chunk)
        facts = read_boot_facts(INSTANCE, "1970-01-01T00:00:00")
        say("probe HTTP %s; KV tokens=%s oom=%s" % (code, facts["tokens"], facts["oom"]))
        return

    controller = controller_base()
    results = []
    final = None
    good = explicit[0] if explicit else good0
    if not good:
        die("no known-good starting pin", "set --start <bytes> or put a kv_cache_memory in the registry")

    try:
        if explicit:
            for pin in explicit:
                say("trying %d bytes (%.2f GiB)" % (pin, pin / GIB))
                r = attempt(pin, chunk, reg, controller, base, good)
                results.append(r)
                say("  %s%s" % ("PASS" if r["pass"] else "FAIL",
                                "" if r["pass"] else " -- " + r.get("reason", "")))
                if r["pass"]:
                    good = pin  # advance the known-good pin
                else:
                    rescue(good, reg, base)
            passed = [r for r in results if r["pass"]]
            final = max((r["pin"] for r in passed), default=None)
            if final is None:
                die("no explicit pin passed")
        else:
            say("pass 1: verifying the start pin %d bytes (%.2f GiB)" % (good, good / GIB))
            r1 = attempt(good, chunk, reg, controller, base, good)
            results.append(r1)
            if not r1["pass"]:
                say("  FAIL -- %s" % r1.get("reason", ""))
                rescue(good, reg, base)
                die("the starting pin does not serve; pick another --start")
            say("  PASS, %s tokens" % r1["facts"]["tokens"])
            final = good

            if not args.quick:
                say("pass 2: raising the pin until it stops serving")
                best = good
                for step in range(1, MAX_STEPS + 1):
                    try_pin = int(good * (1 + STEP * step))
                    say("  +%.0f%%: %d bytes (%.2f GiB)" % (STEP * step * 100, try_pin, try_pin / GIB))
                    r = attempt(try_pin, chunk, reg, controller, base, good)
                    results.append(r)
                    if r["pass"]:
                        say("    served, %s tokens" % r["facts"]["tokens"])
                        best = try_pin
                        good = try_pin
                    else:
                        say("    %s -- stopping" % r.get("reason", "failed"))
                        rescue(good, reg, base)
                        break
                if best != good0 and BACKOFF_STEPS > 0:
                    backed = int(best / (1 + STEP * BACKOFF_STEPS))
                    if backed > good0:
                        say("backing off %d step(s) for margin -> %d bytes (%.2f GiB)"
                            % (BACKOFF_STEPS, backed, backed / GIB))
                        best = backed
                final = best

        say("final pin: %d bytes (%.2f GiB)" % (final, final / GIB))
        set_pin(reg, int(final))
        save_reg(reg)
        ra = attempt(final, chunk, reg, controller, base, good)
        results.append(ra)
        say("applied: %s (%s tokens)" % ("PASS" if ra["pass"] else "FAIL: " + ra.get("reason", ""),
                                         (ra.get("facts") or {}).get("tokens")))
        if not ra["pass"]:
            rescue(good, reg, base)
            die("final pin failed to apply")
    finally:
        if final is None:
            REGISTRY.write_text(orig_text)
            say("restored original registry")
            if good:
                rescue(good, reg, base)
        say("results:")
        for r in results:
            f = r.get("facts") or {}
            say("  pin=%-12s %s  tokens=%s  %s"
                % (r["pin"], "PASS" if r["pass"] else "FAIL", f.get("tokens"), r.get("reason", "")))


if __name__ == "__main__":
    main()
