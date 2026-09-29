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


def post(url, key, model, prompt, mx):
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


def run_point(url, key, model, conc, mx, reps):
    tps, accs = [], []
    for r in range(reps):
        res = [None] * conc
        m0 = read_metrics(url, key)

        def worker(i):
            res[i] = post(url, key, model, PROMPTS[i % len(PROMPTS)], mx)
        th = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        toks = sum(x[0] for x in res)
        m1 = read_metrics(url, key)
        dd = m1["drafted"] - m0["drafted"]
        da = m1["accepted"] - m0["accepted"]
        tps.append(toks / wall)
        accs.append(100.0 * da / dd if dd else 0.0)
    return statistics.mean(tps), statistics.mean(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=8)
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")
    concs = [int(x) for x in a.conc.split(",") if x.strip()]

    post(a.url, key, a.model, "hi", a.warmup)
    print(f"{'conc':>4}  {'tok/s':>8}  {'accept%':>7}")
    for c in concs:
        t, acc = run_point(a.url, key, a.model, c, a.max_tokens, a.reps)
        print(f"{c:>4}  {t:>8.1f}  {acc:>7.1f}")


if __name__ == "__main__":
    main()
