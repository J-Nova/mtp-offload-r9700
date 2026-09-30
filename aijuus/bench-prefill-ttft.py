#!/usr/bin/env python3
"""Long-prompt prefill / TTFT probe against a running server (~1 min).

Companion to bench-conc.py (short-prompt decode) and bench-quick.py. Sends one prompt of a
chosen token budget and reports TTFT, prompt-eval t/s, and decode t/s. Built for PF1/P2/P3
(chunk-size and prefill-geometry measurements), where the short probe prompts are useless.

  python3 bench-prefill.py --base http://172.18.0.4:8000 --model mtp-27B-MXFP4-blend \
      --prompt-tokens 8192 --gen 16 --reps 3 --out /tmp/kilo/ab/prefill_8k.json

Prompt tokens are approximated at ~4 chars/token; pass --chars-per-token to tune.
"""
import argparse
import json
import os
import time
import urllib.request

API_KEY = os.environ.get("BENCH_API_KEY") or os.environ.get("VLLM_API_KEY") or ""


def _req(url, body=None, headers=None):
    h = dict(headers or {})
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return urllib.request.Request(url, body, h)


def build_prompt(n_tokens, cpt):
    # Varied, repeat-light text so n-gram/prompt-lookup can't shortcut the prefill.
    unit = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor "
            "jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. ")
    n = max(1, int((n_tokens * cpt) // len(unit)) + 1)
    return ("[prefill probe]\n" + unit * n).strip()


def one(base, model, prompt, gen, seed, timeout):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": gen, "temperature": 0.0, "seed": seed, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "stream_options": {"include_usage": True}}).encode()
    t0 = time.time()
    tf = last = None
    usage = None
    r = urllib.request.urlopen(_req(base + "/v1/chat/completions", body,
                                    {"Content-Type": "application/json"}), timeout=timeout)
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
    ttft = (tf - t0) if tf else None
    pt = usage["prompt_tokens"] if usage else 0
    ct = usage["completion_tokens"] if usage else 0
    dec = max((last - tf) if (last and tf) else 0.0, 1e-9)
    return {"ttft_ms": ttft * 1000 if ttft else None,
            "prompt_tokens": pt,
            "prompt_tps": (pt / ttft) if (ttft and pt) else None,
            "decode_tps": (ct / dec) if tf else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://172.18.0.4:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "mtp-27B-MXFP4-blend"))
    ap.add_argument("--prompt-tokens", type=int, default=int(os.environ.get("BENCH_PT", "8192")))
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY") or os.environ.get("VLLM_API_KEY") or "")
    a = ap.parse_args()
    global API_KEY
    API_KEY = a.api_key

    prompt = build_prompt(a.prompt_tokens, a.chars_per_token)
    one(a.base, a.model, "warm", 4, 0, a.timeout)  # warmup
    rows = []
    for rep in range(a.reps):
        time.sleep(0.5)
        r = one(a.base, a.model, prompt, a.gen, 1000 + rep, a.timeout)
        rows.append(r)
        print(f"  rep {rep} | prompt {r['prompt_tokens']:>6} tok | ttft {r['ttft_ms']:8.1f} ms | "
              f"prompt {r['prompt_tps']:8.0f} tok/s | decode {r['decode_tps']:7.1f} tok/s", flush=True)
    ok = [r for r in rows if r["ttft_ms"]]
    if ok:
        print(f"--- prompt≈{a.prompt_tokens} tok: mean ttft {sum(r['ttft_ms'] for r in ok)/len(ok):.0f} ms, "
              f"prompt {sum(r['prompt_tps'] for r in ok)/len(ok):.0f} tok/s ---")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"base": a.base, "model": a.model, "prompt_tokens": a.prompt_tokens,
                   "gen": a.gen, "runs": rows}, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
