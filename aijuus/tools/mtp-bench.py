#!/usr/bin/env python3
"""Reproducible decode benchmark for the MTP A/B battery.

Drives the OpenAI-compatible endpoint and reports greedy + sampled decode throughput and spec-decode
acceptance, so every battery arm (B1 SPEC depth, B2 W4 head, B3 drafter quant, ...) is measured the
same way. Dependency-free (urllib).

Run it where the endpoint is reachable, e.g. inside a vLLM container:
    python3 /patches/aijuus/tools/mtp-bench.py --url http://localhost:8000
or against the router from wherever :8100 is published:
    python3 aijuus/tools/mtp-bench.py --url http://localhost:8100

Env: VLLM_API_KEY (default "").
Options: --model, --reps, --max-tokens, --warmup, --temp, --metrics.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

PROMPTS = {
    "greedy-code": "Write a Python function that merges two sorted lists, with a docstring and tests.",
    "greedy-prose": "Summarize the history of the Roman Republic in detail, with dates and key figures.",
    "sampled-prose": "Explain how TCP congestion control works, in detail.",
    "sampled-json": "Respond ONLY with JSON: a list of 20 objects with fields id, name, email, role, active.",
}


def post(url, key, model, prompt, mx, temp):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as f:
        d = json.load(f)
    return d["usage"]["completion_tokens"], time.time() - t0


def fetch_metrics(url, key):
    req = urllib.request.Request(url.rstrip("/") + "/metrics",
                                 headers={"Authorization": "Bearer " + key})
    out = {}
    try:
        with urllib.request.urlopen(req, timeout=30) as f:
            for line in f.read().decode().splitlines():
                for k in ("accepted", "drafted"):
                    name = "vllm:spec_decode_num_%s_tokens" % k
                    if line.startswith(name + " "):
                        out[k] = float(line.split()[1])
    except Exception as e:
        out["error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--metrics", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")

    post(a.url, key, a.model, "hi", a.warmup, 0.0)  # warm

    results = {}
    for name, prompt in PROMPTS.items():
        greedy = name.startswith("greedy")
        tps = []
        for _ in range(a.reps):
            ct, dt = post(a.url, key, a.model, prompt, a.max_tokens, 0.0 if greedy else a.temp)
            tps.append(ct / dt)
        results[name] = tps
        print(f"{name:14} tps={statistics.mean(tps):6.1f}  "
              f"(min {min(tps):.1f} max {max(tps):.1f})")

    g = [x for k, v in results.items() if k.startswith("greedy") for x in v]
    s = [x for k, v in results.items() if k.startswith("sampled") for x in v]
    print(f"greedy mean={statistics.mean(g):6.1f}  sampled mean={statistics.mean(s):6.1f}")
    if a.metrics:
        m = fetch_metrics(a.url, key)
        if "accepted" in m and "drafted" in m:
            acc = m["accepted"] / m["drafted"] * 100 if m["drafted"] else 0.0
            print(f"acceptance: accepted={m['accepted']:.0f} drafted={m['drafted']:.0f} "
                  f"({acc:.1f}%)")
        else:
            print("metrics:", m)


if __name__ == "__main__":
    main()
