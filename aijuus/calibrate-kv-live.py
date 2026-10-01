#!/usr/bin/env python3
"""calibrate-kv-live.py -- measure a KV-cache pin for THIS deployment.

Upstream `calibrate-kv.sh` drives `serve-mxfp4.sh`, which is pinned to the stock
`stilldeadcode/vllm-radiance:0.9.3` image lineage. Our deployment diverged: the
stock image's `mamba/abstract.py` lacks the lazy-GDN anchor, and our
`juupp/vllm-radiance:0.9.3-collect-tokens` image no longer ships
`radiance_allreduce.py` (so patch_ar_maxbytes hard-fails under that launcher's
`set -e`). Neither image can be calibrated through it.

This harness instead drives the REAL stack, exactly as it serves:

  1. set `kv_cache_memory` in aijuus/model-registry.json (the controller bind-
     mounts that file at /model-registry.json),
  2. POST /reload to the model-controller -> it restarts the target instance so
     its entrypoint re-reads the registry with the fresh pin,
  3. poll /health, then run the same PASS test upstream uses: one CHUNK-sized
     prefill plus a short decode. A pin only passes if that step completes --
     reaching "GPU KV cache size" is not enough, because cudagraph capture and
     the first real activation happen after it.
  4. write the chosen pin back to the registry.

A reload RESTARTS the target instance, so this is a deployment operation: run it
yourself, it is not something an agent should do on your behalf.

  ./aijuus/calibrate-kv-live.py                 full sweep (profile, then raise)
  ./aijuus/calibrate-kv-live.py --quick         profile only; keep that pin
  ./aijuus/calibrate-kv-live.py --dry-run       print the plan, change nothing
  ./aijuus/calibrate-kv-live.py --no-reload     probe the current server only
  ./aijuus/calibrate-kv-live.py --pins 6.0e9,6.5e9   explicit pins (bytes)

Env overrides: MODEL_KEY, INSTANCE, CONTROLLER, VLLM_BASE, VLLM_API_KEY,
MODEL_REGISTRY_FILE, CHUNK, STEP, MAX_STEPS, BACKOFF_STEPS, RELOAD_TIMEOUT.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

MODEL_KEY = os.environ.get("MODEL_KEY", "mtp-27B-MXFP4-blend")
INSTANCE = os.environ.get("INSTANCE", "vllm-0")
CONTROLLER = os.environ.get("CONTROLLER", "http://172.18.0.5:8101")
API_KEY = os.environ.get("VLLM_API_KEY", "juup-123")
REGISTRY = Path(os.environ.get("MODEL_REGISTRY_FILE", str(HERE / "model-registry.json")))
INSTANCE_PORT = {"vllm-0": 8000, "vllm-1": 8001}
STEP = float(os.environ.get("STEP", "0.02"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "6"))
BACKOFF_STEPS = int(os.environ.get("BACKOFF_STEPS", "1"))
RELOAD_TIMEOUT = int(os.environ.get("RELOAD_TIMEOUT", "1200"))
MODEL_DEFAULT_MAXLEN = 160000

GIB = 1 << 30


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
           "label=com.docker.compose.service=%s" % instance,
           "--format", "{{.ID}}")
    ids = [l for l in r.stdout.split() if l.strip()]
    return ids[0] if ids else None


def instance_base(instance):
    override = os.environ.get("VLLM_BASE")
    if override:
        return override.rstrip("/")
    cid = container_id(instance)
    if not cid:
        die("no container for service %r (is the compose up?)" % instance)
    r = sh("docker", "inspect", "-f",
           "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", cid)
    ip = next((t for t in r.stdout.split() if t), None)
    if not ip:
        die("could not resolve an IP for %s" % instance)
    return "http://%s:%d" % (ip, INSTANCE_PORT.get(instance, 8000))


def read_logs(instance, tail=3000):
    cid = container_id(instance)
    if not cid:
        return ""
    return sh("docker", "logs", "--tail", str(tail), cid).stdout + \
        sh("docker", "logs", "--tail", str(tail), cid).stderr


def last(pattern, text, cast=str):
    m = re.findall(pattern, text)
    return cast(m[-1]) if m else None


def read_boot_facts(instance):
    """Parse the last boot's KV figures from the instance's container logs."""
    t = read_logs(instance)
    toks = last(r"GPU KV cache size:\s*([\d,]+)\s*tokens", t)
    return {
        "tokens": int(toks.replace(",", "")) if toks else None,
        "available_gib": last(r"Available KV cache memory:\s*([\d.]+)\s*GiB", t, float),
        "reserved_gib": last(r"reserved\s*([\d.]+)\s*GiB memory for KV", t, float),
        "free_gib": last(r"Initial free memory\s*([\d.]+)\s*GiB", t, float),
        "oom": bool(re.search(r"out of memory|hipErrorOutOfMemory", t, re.I)),
    }


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


# --------------------------------------------------------------------------- steps
def wait_health(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_status("GET", base + "/health", timeout=5)[0] == 200:
            return True
        time.sleep(5)
    return False


def reload_instance():
    st, body = get_json("POST", CONTROLLER.rstrip("/") + "/reload",
                        {"model": MODEL_KEY, "instance": INSTANCE}, timeout=RELOAD_TIMEOUT)
    return st, body


def probe(base, model, chunk, timeout=600):
    """One CHUNK-sized prefill + a short decode -- upstream's PASS test."""
    n_words = max(64, int(chunk * 1.05 / 11))  # ~11 tokens per phrase
    prompt = "the quick brown fox jumps over the lazy dog. " * (n_words // 9 + 1)
    body = {"model": model, "prompt": prompt, "max_tokens": 32, "temperature": 0}
    return http_status("POST", base + "/v1/completions", body, timeout=timeout)[0]


def served_name(reg):
    return reg[MODEL_KEY].get("served_name") or MODEL_KEY


def attempt(pin, chunk, reg, do_reload):
    """Set pin (or profile if None), (optionally) reload, probe. Returns a dict."""
    set_pin(reg, "" if pin is None else int(pin))
    save_reg(reg)
    if do_reload:
        st, body = reload_instance()
        if st != 200 or not body.get("ok"):
            return {"pin": pin, "pass": False, "reason": "reload failed: %s %s" % (st, body)}
    else:
        say("  --no-reload: probing the currently-served config")
    base = instance_base(INSTANCE)
    if not wait_health(base, RELOAD_TIMEOUT):
        facts = read_boot_facts(INSTANCE)
        return {"pin": pin, "pass": False,
                "reason": "did not become healthy" + (" (OOM)" if facts["oom"] else ""),
                "facts": facts}
    code = probe(base, served_name(reg), chunk)
    facts = read_boot_facts(INSTANCE)
    if code != 200:
        return {"pin": pin, "pass": False, "reason": "prefill probe HTTP %s" % code,
                "facts": facts}
    return {"pin": pin, "pass": True, "facts": facts}


def main():
    ap = argparse.ArgumentParser(description="measure a KV pin for the live deployment")
    ap.add_argument("--quick", action="store_true", help="profile only; keep that pin")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ap.add_argument("--no-reload", action="store_true", help="probe current server only")
    ap.add_argument("--pins", default="", help="explicit comma-separated pins in bytes")
    args = ap.parse_args()

    if not REGISTRY.exists():
        die("registry not found: %s" % REGISTRY)
    reg = load_reg()
    if MODEL_KEY not in reg:
        die("model %r not in %s" % (MODEL_KEY, REGISTRY), "known: %s" % ", ".join(sorted(k for k in reg if not k.startswith("_"))))
    entry = reg[MODEL_KEY]
    chunk = int(os.environ.get("CHUNK") or entry.get("max_num_batched_tokens") or 4096)
    maxlen = int(entry.get("max_model_len") or MODEL_DEFAULT_MAXLEN)
    orig_text = REGISTRY.read_text()

    say("registry:  %s" % REGISTRY)
    say("model:     %s (served as %r)" % (MODEL_KEY, served_name(reg)))
    say("instance:  %s @ %s" % (INSTANCE, instance_base(INSTANCE) if container_id(INSTANCE) else "(not running)"))
    say("controller:%s" % CONTROLLER)
    say("shape:     chunk=%d maxlen=%d spec=%s maxseqs=%s" % (
        chunk, maxlen, entry.get("spec_tokens", "?"),
        entry.get("server_env", {}).get("MAX_NUM_SEQS", "8")))
    say("current:   kv_cache_memory=%s" % entry.get("kv_cache_memory"))
    say("step:      %+.0f%% x %d, backoff %d  (reload restarts %s)"
        % (STEP * 100, MAX_STEPS, BACKOFF_STEPS, INSTANCE))

    if args.dry_run:
        base = entry.get("kv_cache_memory") or 6 * GIB
        say("pass 1: profile (kv_cache_memory=\"\")")
        say("pass 2: raise by %+.0f%% up to %dx from the profiled figure" % (STEP * 100, MAX_STEPS))
        say("        then back off %d step(s); would write the best pin to the registry" % BACKOFF_STEPS)
        return

    explicit = [int(float(x)) for x in args.pins.split(",") if x.strip()]
    results = []
    final = None
    try:
        if args.no_reload:
            base = instance_base(INSTANCE)
            code = probe(base, served_name(reg), chunk)
            facts = read_boot_facts(INSTANCE)
            say("probe HTTP %s; KV tokens=%s free=%sGiB oom=%s"
                % (code, facts["tokens"], facts["free_gib"], facts["oom"]))
            return

        if explicit:
            for pin in explicit:
                say("trying %d bytes (%.2f GiB)" % (pin, pin / GIB))
                r = attempt(pin, chunk, reg, True)
                results.append(r)
                say("  %s%s" % ("PASS" if r["pass"] else "FAIL",
                                "" if r["pass"] else " -- " + r.get("reason", "")))
            passed = [r for r in results if r["pass"]]
            if passed:
                final = max(r["pin"] for r in passed)
            else:
                die("no explicit pin passed", "try smaller pins")
        else:
            say("pass 1/2: profiling run (kv_cache_memory=\"\")")
            r1 = attempt(None, chunk, reg, True)
            results.append(r1)
            if not r1["pass"]:
                die("the profiling run itself failed: %s" % r1.get("reason", ""),
                    "the config does not serve at this shape -- fix that before calibrating")
            avail = r1["facts"]["available_gib"]
            if not avail:
                die("profiled run produced no 'Available KV cache memory' figure",
                    "cannot compute a starting pin; use --pins explicitly")
            base = int(avail * GIB)
            say("pass 1: profiled %.2f GiB, %s tokens" % (avail, r1["facts"]["tokens"]))
            final = base

            if not args.quick:
                say("pass 2/2: raising the pin until it stops serving")
                best = base
                for step in range(1, MAX_STEPS + 1):
                    try_pin = int(base * (1 + STEP * step))
                    say("  +%.0f%%: %d bytes (%.2f GiB)" % (STEP * step * 100, try_pin, try_pin / GIB))
                    r = attempt(try_pin, chunk, reg, True)
                    results.append(r)
                    if r["pass"]:
                        say("    served, %s tokens" % r["facts"]["tokens"])
                        best = try_pin
                    else:
                        say("    %s -- stopping" % r.get("reason", "failed"))
                        break
                if best != base and BACKOFF_STEPS > 0:
                    backed = int(best / (1 + STEP * BACKOFF_STEPS))
                    if backed > base:
                        say("backing off %d step(s) for margin -> %d bytes (%.2f GiB)"
                            % (BACKOFF_STEPS, backed, backed / GIB))
                        best = backed
                final = best

        say("final pin: %d bytes (%.2f GiB)" % (final, final / GIB))
        set_pin(reg, int(final))
        save_reg(reg)
        say("written to %s" % REGISTRY)
        say("NOTE: restart %s (or POST /reload) so the live server picks it up." % INSTANCE)
    finally:
        if final is None:
            # abort -- leave the registry exactly as we found it
            REGISTRY.write_text(orig_text)
            say("restored original registry (no pin selected)")
        say("results:")
        for r in results:
            f = r.get("facts") or {}
            say("  pin=%-12s %s  tokens=%s free=%sGiB  %s"
                % (r["pin"], "PASS" if r["pass"] else "FAIL",
                   f.get("tokens"), f.get("free_gib"), r.get("reason", "")))


if __name__ == "__main__":
    main()
