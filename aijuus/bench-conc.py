#!/usr/bin/env python3
"""Quick concurrency decode probe (~1 min) against an already-running server.

Fires C concurrent streaming chat completions of a fixed token budget and reports
aggregate decode t/s, per-request t/s, and the spec-decode acceptance delta from
/metrics. Companion to bench-quick.py (single-stream). Tuned for the compose
instances (vllm-0/vllm-1), which require Bearer auth.

  python3 bench-conc.py --base http://172.18.0.4:8000 --model mtp-27B-MXFP4-blend \
      --conc 8 --gen 256 --out /tmp/kilo/ab/conc.json
"""
import argparse
import json
import os
import threading
import time
import urllib.request

API_KEY = os.environ.get("BENCH_API_KEY") or os.environ.get("VLLM_API_KEY") or ""

PROMPT = os.environ.get("BENCH_PROMPT") or (
    "Write a detailed technical explanation of how a B-tree index works, why node splits "
    "happen, and how the fanout affects lookup cost. Be specific and do not repeat yourself.")


def _req(url, body=None, headers=None):
    h = dict(headers or {})
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return urllib.request.Request(url, body, h)


def metrics(base):
    try:
        raw = urllib.request.urlopen(_req(base + "/metrics"), timeout=10).read().decode()
    except Exception:
        return None
    want = {"n": "vllm:spec_decode_num_drafts_total",
            "d": "vllm:spec_decode_num_draft_tokens_total",
            "a": "vllm:spec_decode_num_accepted_tokens_total"}
    out = {}
    pos = {}
    for line in raw.splitlines():
        for k, m in want.items():
            if line.startswith(m):
                out[k] = float(line.rsplit(" ", 1)[1])
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            try:
                p = int(line.split('position="', 1)[1].split('"', 1)[0])
                pos[p] = float(line.rsplit(" ", 1)[1])
            except Exception:
                pass
    if len(out) != 3:
        return None
    out["pos"] = pos
    return out


def one(base, model, prompt, gen, seed, timeout, res, idx):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": f"[s{seed}]\n{prompt}"}],
                       "max_tokens": gen, "temperature": 0.7, "seed": seed, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "stream_options": {"include_usage": True}}).encode()
    try:
        r = urllib.request.urlopen(_req(base + "/v1/chat/completions", body,
                                        {"Content-Type": "application/json"}), timeout=timeout)
        t0 = time.time()
        tf = last = None
        usage = None
        for line in r:
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("choices"):
                delta = d["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    now = time.time()
                    if tf is None:
                        tf = now
                    last = now
            if d.get("usage"):
                usage = d["usage"]
        ct = usage["completion_tokens"] if usage else 0
        dec = max((last - tf) if tf else 0.0, 1e-9)
        res[idx] = {"ct": ct, "ttft": (tf - t0) if tf else None, "tp": ct / dec if tf else None}
    except Exception as e:
        res[idx] = {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://172.18.0.4:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "mtp-27B-MXFP4-blend"))
    ap.add_argument("--conc", type=int, default=int(os.environ.get("BENCH_CONC", "8")))
    ap.add_argument("--gen", type=int, default=int(os.environ.get("BENCH_GEN", "256")))
    ap.add_argument("--reps", type=int, default=int(os.environ.get("BENCH_REPS", "2")))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY") or os.environ.get("VLLM_API_KEY") or "")
    a = ap.parse_args()
    global API_KEY
    API_KEY = a.api_key

    # warmup
    res = {}
    for _ in range(1):
        one(a.base, a.model, "hi", 8, 1, a.timeout, res, 0)
    rows = []
    for rep in range(a.reps):
        time.sleep(0.5)
        before = metrics(a.base)
        t0 = time.time()
        res = {}
        ths = [threading.Thread(target=one, args=(a.base, a.model, PROMPT, a.gen, 1000 + rep * 100 + i,
                                                   a.timeout, res, i)) for i in range(a.conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        after = metrics(a.base)
        good = [r for r in res.values() if r.get("tp")]
        ct = sum(r["ct"] for r in good)
        agg = ct / max(wall, 1e-9)
        per = (sum(r["tp"] for r in good) / len(good)) if good else float("nan")
        ttfts = [r["ttft"] for r in good if r.get("ttft")]
        mean_ttft = (sum(ttfts) / len(ttfts) * 1000) if ttfts else float("nan")
        accd = maxd = None
        posrate = None
        if before and after:
            dn, dd, da = (after[k] - before[k] for k in ("n", "d", "a"))
            accd = da / dn if dn else None
            maxd = da / max(dd, 1) if dd else None
            # per-position conditional acceptance: accepted[pos] / accepted[pos-1]
            bpos, apos = before.get("pos", {}), after.get("pos", {})
            dp = {p: apos.get(p, 0.0) - bpos.get(p, 0.0) for p in set(bpos) | set(apos)}
            posrate = []
            prev = None
            for p in sorted(dp):
                if prev is None:
                    posrate.append(dp[p] / dn if dn else 0.0)  # P(accept pos0 | draft)
                else:
                    posrate.append(dp[p] / prev if prev else 0.0)  # P(accept pos p | accept p-1)
                prev = dp[p]
        rows.append({"rep": rep, "conc": a.conc, "gen": a.gen, "wall_s": wall,
                     "tokens": ct, "agg_tps": agg, "per_req_tps": per, "ttft_ms": mean_ttft,
                     "acc_per_draft": accd, "accept_rate": maxd, "pos_rate": posrate})
        print(f"  conc {a.conc:>2} r{rep} | {ct:6d} tok | {wall:5.2f} s | agg {agg:7.1f} tok/s | "
              f"per-req {per:6.1f} | ttft {mean_ttft:6.0f} ms | acc/draft {accd if accd is None else round(accd,3)} | "
              f"rate {None if maxd is None else round(100*maxd,2)}%", flush=True)
        if posrate:
            print("       acc by pos (cond): " + " ".join(f"p{i}={r:.2f}" for i, r in enumerate(posrate)),
                  flush=True)
    agg = sum(r["agg_tps"] for r in rows) / len(rows)
    print(f"--- conc {a.conc}: mean aggregate {agg:.1f} tok/s over {len(rows)} reps ---")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"base": a.base, "model": a.model, "conc": a.conc, "gen": a.gen,
                   "mean_agg_tps": agg, "runs": rows}, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
