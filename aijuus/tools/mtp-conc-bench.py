#!/usr/bin/env python3
"""Concurrency sweep bench for the MTP dynamic-SD A/B battery.

`mtp-bench.py` is single-stream, and at batch=1 the dynamic schedule resolves to the
static depth ([[1,1,8],...] -> nspec=8), so it cannot show a dynamic-SD benefit. This
drives C concurrent requests per point (C in 1/2/4/8) and reports aggregate decode
throughput plus spec-decode acceptance from `/metrics` deltas.

Dependency-free (urllib + threads). Run where the endpoint is reachable:
    python3 /patches/aijuus/tools/mtp-conc-bench.py --url http://localhost:8000 --conc 1,2,4,8
Env: VLLM_API_KEY. Options: --model, --reps, --max-tokens, --warmup.
"""
import argparse
import json
import os
import random
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    "Write a detailed technical explanation of how transformer attention scales, with complexity analysis.",
    "Summarize the causes and consequences of the fall of the Western Roman Empire.",
    "Explain how TCP congestion control works, covering slow start, congestion avoidance and fast recovery.",
    "Write a Python module implementing a red-black tree with insert, delete and verification, plus tests.",
    "Describe how virtual memory, paging and TLB shootdown work in a modern OS kernel.",
    "Explain public-key cryptography: RSA, elliptic curves, and why padding matters, with examples.",
    "Write a long essay on the history of compiler optimization, from peepholes to SSA and auto-vectorization.",
    "Explain distributed consensus: Paxos, Raft, and how they handle partitions and leader failure.",
]


def post(url, key, model, prompt, mx, unique=False):
    if unique:
        prompt = prompt + " (ref %d)" % random.randrange(1 << 30)
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "temperature": 0.0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as f:
        d = json.load(f)
    return d["usage"]["completion_tokens"], time.time() - t0


def read_metrics(url, key):
    out = {}
    req = urllib.request.Request(url.rstrip("/") + "/metrics",
                                 headers={"Authorization": "Bearer " + key})
    wanted = {"vllm:spec_decode_num_draft_tokens_total": "drafted",
              "vllm:spec_decode_num_accepted_tokens_total": "accepted"}
    with urllib.request.urlopen(req, timeout=30) as f:
        for line in f.read().decode().splitlines():
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in wanted:
                out[wanted[name]] = float(line.split()[1])
    return out


def read_token_counts(url, key):
    """Total prompt tokens served by one instance (for split reporting).

    Returns None when the endpoint has no /metrics (e.g. a load balancer).
    """
    if not url:
        return None
    try:
        req = urllib.request.Request(url.rstrip("/") + "/metrics",
                                     headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=30) as f:
            data = f.read().decode()
    except Exception:
        return None
    total, found = 0.0, False
    for line in data.splitlines():
        if line.startswith("#"):
            continue
        if line.split("{", 1)[0].split(" ", 1)[0] == "vllm:prompt_tokens_total":
            try:
                total += float(line.rsplit(" ", 1)[1])
                found = True
            except Exception:
                pass
    return total if found else None


def run_point(url, key, model, conc, mx, reps, metrics=True, instances=None,
              unique=False):
    tps, accs = [], []
    before = [read_token_counts(i, key) for i in (instances or [])]
    for r in range(reps):
        res = [None] * conc
        m0 = read_metrics(url, key) if metrics else {}

        def worker(i):
            try:
                res[i] = post(url, key, model, PROMPTS[i % len(PROMPTS)], mx, unique)
            except Exception:
                res[i] = None
        th = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        toks = sum(x[0] for x in res if x)
        errs = sum(1 for x in res if not x)
        if errs:
            print("  [warn] %d/%d requests failed at conc=%d" % (errs, conc, conc))
        if toks:
            tps.append(toks / wall)
        if metrics:
            m1 = read_metrics(url, key)
            dd = m1["drafted"] - m0["drafted"]
            da = m1["accepted"] - m0["accepted"]
            accs.append(100.0 * da / dd if dd else 0.0)
    split = None
    if instances:
        after = [read_token_counts(i, key) for i in instances]
        deltas = [(b - a) if (a is not None and b is not None) else 0.0
                  for a, b in zip(before, after)]
        tot = sum(deltas)
        split = [(100.0 * d / tot if tot else 0.0) for d in deltas]
    if not tps:
        return 0.0, None, split
    return statistics.mean(tps), (statistics.mean(accs) if accs else None), split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip /metrics (e.g. behind a load balancer that does not proxy it)")
    ap.add_argument("--instances", default="",
                    help="comma-separated instance base URLs to report the LB split")
    ap.add_argument("--unique", action="store_true",
                    help="append a nonce to each prompt (defeats prefix-cache/fs-tier reuse)")
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")
    concs = [int(x) for x in a.conc.split(",") if x.strip()]
    instances = [x.strip() for x in a.instances.split(",") if x.strip()]

    post(a.url, key, a.model, "hi", a.warmup)
    metrics = not a.no_metrics
    print(f"{'conc':>4}  {'tok/s':>8}  {'accept%':>7}" + ("  split" if instances else ""))
    for c in concs:
        t, acc, split = run_point(a.url, key, a.model, c, a.max_tokens, a.reps,
                                  metrics, instances, a.unique)
        accs = "n/a" if acc is None else f"{acc:.1f}"
        row = f"{c:>4}  {t:>8.1f}  {accs:>7}"
        if split:
            row += "  " + " ".join("%s=%.0f%%" % (i.rsplit("/", 1)[-1], s)
                                   for i, s in zip(instances, split))
        print(row)


if __name__ == "__main__":
    main()
