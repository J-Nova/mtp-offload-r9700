#!/usr/bin/env python3
"""MT1/MT2 discriminating oracle: is the cached KV writer faithful? (full-hit vs cold)

turnbench --exact compares a CACHED run (whose suffix is prefilled on a hit-anchored, block-aligned
chunk grid) against a COLD twin (full prefill from 0). Under chunked prefill + fp8 KV + a chunk-local
GDN scan those grids differ, so bit-identity is not expected even when the stored bytes are correct.

This isolates the WRITER: request the SAME prompt twice under the SAME cache_salt (second is a full
prefix hit -> no prefill chunk at all) and once under a FRESH salt (cold recompute), all temp=0 with
logprobs, and compare bit-exactly.

  full-hit == cold      -> the stored KV bytes are cold-reproducible; MT1 is inherent resume-phase
                           numerics (a), not a KV bug. (MT2 then inherits through the same mechanism.)
  full-hit != cold      -> writer-side bug (b); escalate to the byte-level statecmp.

Also reports the partial-hit case (a prompt that shares a prefix but appends new tokens) so the
signature is visible in one run.

Usage: python3 hit_oracle.py --base http://172.18.0.4:8000 --model mtp-27B-MXFP4-blend --tokens 12000
Reads VLLM_API_KEY for auth.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

KEY = os.environ.get("VLLM_API_KEY") or os.environ.get("TIERBENCH_API_KEY") or ""


def hdrs():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def ask(base, model, prompt, salt, max_tokens=12):
    return ask_msgs(base, model, [{"role": "user", "content": prompt}], salt, max_tokens)


def ask_msgs(base, model, messages, salt, max_tokens=12):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True,
                       "top_logprobs": 5, "cache_salt": salt,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", body, hdrs()), timeout=600))
    ch = r["choices"][0]
    toks = ch["logprobs"]["content"] if ch.get("logprobs") else []
    ids = [t.get("token") for t in toks]
    lps = [t.get("logprob") for t in toks]
    return {"ids": ids, "lps": lps, "text": ch["message"]["content"]}


def cmp(a, b):
    n = min(len(a["ids"]), len(b["ids"]))
    tid = a["ids"][:n] == b["ids"][:n]
    dlp = 0.0
    for i in range(n):
        if a["lps"][i] is not None and b["lps"][i] is not None:
            dlp = max(dlp, abs(a["lps"][i] - b["lps"][i]))
    first = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
    return {"tok_identical": tid, "max_dlp": dlp, "first_div": first, "n": n}


def build(n_tokens):
    unit = ("def f(x): return x + 1\n# a comment line about accumulation order and rounding\n"
            "for i in range(10): print(i, i*i, 'done')\n")
    rep = max(1, (n_tokens * 4) // len(unit))
    return "You are reviewing this code. Summarise it.\n" + unit * rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://172.18.0.4:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "mtp-27B-MXFP4-blend"))
    ap.add_argument("--tokens", type=int, default=12000)
    a = ap.parse_args()
    p = build(a.tokens)
    s1 = "mtsalt-%x" % (int(time.time()) & 0xFFFFFF)
    s2 = s1                       # same salt -> full prefix hit on the 2nd request
    s3 = s1 + "-cold"             # fresh salt -> cold recompute
    cold_writer = ask(a.base, a.model, p, s1)          # first request: cold writer, populates cache
    full_hit = ask(a.base, a.model, p, s2)             # full hit (no prefill chunk)
    cold_twin = ask(a.base, a.model, p, s3)            # cold recompute
    print("prompt tokens (approx):", a.tokens)
    print("writer(cold) vs cold_twin :", cmp(cold_writer, cold_twin))
    print("full_hit     vs cold_twin :", cmp(full_hit, cold_twin))
    print("full_hit     vs writer    :", cmp(full_hit, cold_writer))
    # partial hit: same long prefix + a small new suffix (forces a hit-anchored suffix prefill)
    pp = p + "\nNow add a second function g that multiplies two numbers."
    part = ask(a.base, a.model, pp, s1)                # partial hit (prefix cached under s1)
    coldp = ask(a.base, a.model, pp, s3)               # cold recompute of the longer prompt
    print("partial_hit  vs cold_twin :", cmp(part, coldp))

    # multi-turn: reuse a cached prefix that CONTAINS an assistant reply written by the MTP decode
    # path (the turnbench structure; reply blocks were written by M=1 split-K decode, while the cold
    # twin recomputes them with a prefill GEMM -> different accumulation, inherent).
    msgs1 = [{"role": "user", "content": p}]
    r1c = ask_msgs(a.base, a.model, msgs1, s1, max_tokens=160)
    msgs2 = msgs1 + [{"role": "assistant", "content": r1c["text"]},
                     {"role": "user", "content": "Now list three edge cases as bullet points."}]
    turn2_hit = ask_msgs(a.base, a.model, msgs2, s1, max_tokens=160)   # cached (reply reuse)
    turn2_cold = ask_msgs(a.base, a.model, msgs2, s3, max_tokens=160)  # cold recompute
    print("turn2_hit(reply reuse) vs cold :", cmp(turn2_hit, turn2_cold))


if __name__ == "__main__":
    main()
