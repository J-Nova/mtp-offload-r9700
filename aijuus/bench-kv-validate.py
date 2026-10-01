#!/usr/bin/env python3
"""bench-kv-validate.py -- quick "does this KV pin OOM?" stress test (~2-4 min).

BetterBench is the thorough reference, but for validating a `kv_cache_memory` pin
the two things that matter are (a) a cold LONG prefill (the transient activation
peak, amplified by the chunk) and (b) concurrency (KV pressure + a mixed batch).
This fires those directly against a running compose instance and prints a verdict.

It does NOT measure quality or produce charts -- it answers "did the engine
survive, and how fast". On the first hard failure it reports which case died
(OOM / 500 / connection refused) instead of pretending the rest is valid.

  python3 aijuus/bench-kv-validate.py
  python3 aijuus/bench-kv-validate.py --depths 16000,64000,160000 --conc 1,4,8
  python3 aijuus/bench-kv-validate.py --base http://172.18.0.2:8000

Env: BENCH_API_KEY / VLLM_API_KEY, INSTANCE, VLLM_BASE, MODEL_KEY.
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

INSTANCE = os.environ.get("INSTANCE", "vllm-0")
MODEL = os.environ.get("MODEL_KEY", "mtp-27B-MXFP4-blend")
API_KEY = os.environ.get("BENCH_API_KEY") or os.environ.get("VLLM_API_KEY") or "juup-123"
INSTANCE_PORT = {"vllm-0": 8000, "vllm-1": 8001}

# repeat-light, varied text so n-gram / prompt-lookup cannot shortcut the prefill
UNIT = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. ")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError|hipErrorOutOfMemory|EngineCore encountered", re.I)


def _req(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        r.add_header("Content-Type", "application/json")
    if API_KEY:
        r.add_header("Authorization", "Bearer " + API_KEY)
    return r


def http(url, body=None, timeout=900):
    t0 = time.time()
    try:
        with urllib.request.urlopen(_req(url, body), timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, raw, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0
    except Exception as e:
        return 0, str(e).encode(), time.time() - t0


def discover_base():
    if os.environ.get("VLLM_BASE"):
        return os.environ["VLLM_BASE"].rstrip("/")
    cid = subprocess.run(["docker", "ps", "-a", "--filter",
                          "label=com.docker.compose.service=%s" % INSTANCE,
                          "--format", "{{.ID}}"], capture_output=True, text=True).stdout.split()
    if not cid:
        raise SystemExit("no container for %s; pass --base" % INSTANCE)
    ips = subprocess.run(["docker", "inspect", "-f",
                          "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", cid[0]],
                         capture_output=True, text=True).stdout.split()
    ip = next((i for i in ips if i), None)
    return "http://%s:%d" % (ip, INSTANCE_PORT.get(INSTANCE, 8000))


def build_prompt(n_tokens, nonce):
    n = max(1, (n_tokens * 4) // len(UNIT) + 1)   # ~4 chars/token
    return ("[%s]\n" % nonce) + UNIT * n


def health(base, timeout=8):
    try:
        with urllib.request.urlopen(_req(base + "/health"), timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def prefill_case(base, n_tokens, gen=1):
    nonce = uuid.uuid4().hex
    body = {"model": MODEL, "messages": [{"role": "user", "content": build_prompt(n_tokens, nonce)}],
            "max_tokens": gen, "temperature": 0.0}
    status, raw, dt = http(base + "/v1/chat/completions", body)
    ptok = ctok = None
    if status == 200:
        try:
            u = json.loads(raw).get("usage", {})
            ptok, ctok = u.get("prompt_tokens"), u.get("completion_tokens")
        except Exception:
            pass
    return {"target": n_tokens, "status": status, "latency": dt, "ptok": ptok, "ctok": ctok,
            "err": raw[:200].decode(errors="replace") if status != 200 else "",
            "oom": bool(OOM_RE.search(raw[:2000].decode(errors="replace")))}


def conc_case(base, level, prompt_tokens, gen):
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(level)

    def one(i):
        nonce = uuid.uuid4().hex
        body = {"model": MODEL, "messages": [{"role": "user", "content": build_prompt(prompt_tokens, nonce)}],
                "max_tokens": gen, "temperature": 0.7}
        barrier.wait()
        status, raw, dt = http(base + "/v1/chat/completions", body)
        u = {}
        if status == 200:
            try:
                u = json.loads(raw).get("usage", {})
            except Exception:
                pass
        with lock:
            results.append({"status": status, "dt": dt, "ctok": u.get("completion_tokens"),
                            "oom": bool(OOM_RE.search(raw[:2000].decode(errors="replace")))})

    wall0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(level)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - wall0
    ok = [r for r in results if r["status"] == 200]
    tot = sum((r["ctok"] or 0) for r in ok)
    return {"level": level, "ok": len(ok), "n": level, "wall": wall,
            "agg_tps": (tot / wall) if wall > 0 else 0.0,
            "per_req_tps": (sum((( r["ctok"] or 0) / r["dt"]) for r in ok) / len(ok)) if ok else 0.0,
            "any_oom": any(r["oom"] for r in results),
            "statuses": sorted({r["status"] for r in results if r["status"] != 200})}


def main():
    ap = argparse.ArgumentParser(description="quick KV-pin OOM stress validator")
    ap.add_argument("--base", default="")
    ap.add_argument("--depths", default="8000,16000,32000,64000,96000,128000,160000",
                    help="cold long-prefill depths (approx tokens)")
    ap.add_argument("--conc", default="1,2,4,8", help="concurrency levels")
    ap.add_argument("--conc-prompt", type=int, default=4096, help="tokens per concurrent prompt")
    ap.add_argument("--conc-gen", type=int, default=128, help="tokens to generate per concurrent req")
    ap.add_argument("--quick", action="store_true", help="depths 16000,64000,160000; conc 8 only")
    args = ap.parse_args()

    base = (args.base or discover_base()).rstrip("/")
    depths = [int(x) for x in (("16000,64000,160000" if args.quick else args.depths)).split(",")]
    levels = [int(x) for x in ("8" if args.quick else args.conc).split(",")]

    print("[kv-validate] base=%s model=%s instance=%s" % (base, MODEL, INSTANCE), flush=True)
    if not health(base):
        raise SystemExit("[kv-validate] server not healthy at %s -- is the pin loaded?" % base)

    failures = []
    print("\n== cold long-prefill (max_tokens=1) ==")
    print("  target   actual   TTFT/s   PP t/s   status")
    for d in depths:
        r = prefill_case(base, d)
        pp = (r["ptok"] / r["latency"]) if (r["ptok"] and r["latency"]) else 0.0
        flag = "OK" if r["status"] == 200 else ("OOM" if r["oom"] else "HTTP %s" % r["status"])
        print("  %7d  %7s  %7.2fs  %8.0f   %s" %
              (d, r["ptok"] or "-", r["latency"], pp, flag), flush=True)
        if r["status"] != 200:
            failures.append(("prefill", d, flag, r["err"][:120]))
            if not health(base):
                print("  !! engine is down after %d -- stopping" % d, flush=True)
                break

    if health(base):
        print("\n== concurrency (prompt=%d tok, gen=%d) ==" % (args.conc_prompt, args.conc_gen))
        print("  level   ok     wall    agg t/s  per-req t/s")
        for lv in levels:
            r = conc_case(base, lv, args.conc_prompt, args.conc_gen)
            print("  %5d  %d/%d  %6.1fs  %8.1f  %10.1f" %
                  (lv, r["ok"], r["n"], r["wall"], r["agg_tps"], r["per_req_tps"]), flush=True)
            if r["ok"] != r["n"]:
                failures.append(("conc", lv, "ok %d/%d oom=%s statuses=%s"
                                 % (r["ok"], r["n"], r["any_oom"], r["statuses"]), ""))
                if not health(base):
                    print("  !! engine is down after conc=%d -- stopping" % lv, flush=True)
                    break

    print()
    if not failures:
        print("[kv-validate] VERDICT: PASS -- survived prefill to %d tok and conc to %d, no OOM"
              % (max(depths), max(levels)))
    else:
        print("[kv-validate] VERDICT: FAIL")
        for kind, d, flag, err in failures:
            print("  %s %s: %s %s" % (kind, d, flag, err))
        print("  health now: %s" % ("up" if health(base) else "DOWN (engine died)"))


if __name__ == "__main__":
    main()
