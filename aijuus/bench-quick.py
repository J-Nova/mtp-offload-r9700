#!/usr/bin/env python3
"""Fast decode A/B probe for the RADIANCE MTP + n-gram controller (~2-3 min of generation).

Complements the 25-30 min BetterBench blend with a short, single-stream, decode-dominated probe
built to move when the n-gram / draft-depth knobs move. It hits an ALREADY-RUNNING server (use
bench-quick.sh to launch one), fires a few fixed prompts, and reports per-request:

  TTFT, decode tok/s, ms/step, acc/draft, dup-8gram  -- plus the vllm:spec_decode_* deltas.

Prompts are chosen so the categories that should benefit from verbatim n-gram tails (repetition,
code, JSON) sit next to prose that should not, so a change that helps code without hurting prose is
visible at a glance.

  python3 bench-quick.py --base http://localhost:6564 --out bench/quick.json

Env: BENCH_GEN, BENCH_REPS, BENCH_TEMP, BENCH_SEED. Exit 0 always (the numbers are the output).
"""
import argparse
import json
import os
import random
import string
import sys
import time
import urllib.request

# (category, prompt). Small prompts -> prefill is negligible and decode dominates.
PROMPTS = [
    ("echo",
     "Repeat the following line 40 times, numbered 1..40, one per line, and output nothing else:\n"
     "the quick brown fox jumps over the lazy dog\n"),
    ("code",
     "Write a Python module defining 20 dataclasses, each with fields id: int, name: str, "
     "score: float, active: bool, plus a to_dict method. Output only the code.\n"),
    ("json",
     "Output a JSON array of 32 objects, each with keys id, name, email, role, active, created_at. "
     "Use the exact same schema for every object. Output only the JSON.\n"),
    ("prose",
     "Explain how a write-ahead log works in a database and why fsync ordering matters. Write in "
     "your own words, be specific, and do not repeat yourself.\n"),
]


def metrics(base):
    """(drafts, draft_tokens, accepted) cumulative, or None. Only deltas are meaningful."""
    try:
        raw = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception:
        return None
    want = {"n": "vllm:spec_decode_num_drafts_total",
            "d": "vllm:spec_decode_num_draft_tokens_total",
            "a": "vllm:spec_decode_num_accepted_tokens_total"}
    out = {}
    for line in raw.splitlines():
        for k, m in want.items():
            if line.startswith(m):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out if len(out) == 3 else None


def one(base, model, prompt, gen, temp, seed, timeout, thinking=False):
    tag = "".join(random.Random(time.time_ns()).choices(string.ascii_lowercase, k=16))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"[session {tag}]\n{prompt}"}],
                       "max_tokens": gen, "temperature": temp, "seed": seed, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": bool(thinking)},
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    time.sleep(0.5)  # let the previous request's counters settle before snapshotting
    before = metrics(base)
    t0 = time.time()
    tf = None
    last = t0
    usage = None
    text = []
    for line in urllib.request.urlopen(req, timeout=timeout):
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
            # qwen3 with thinking on streams the reasoning in `reasoning` and the answer in
            # `content`; both are real generation, so time on whichever arrives (this is why a
            # content-only parser reported absurd tok/s: usage counted 160 tokens but the clock only
            # saw the short final answer). vLLM 0.27.1 emits `reasoning` (not `reasoning_content`).
            piece = delta.get("content") or delta.get("reasoning")
            if piece:
                now = time.time()
                if tf is None:
                    tf = now
                last = now
                text.append(piece)
        if d.get("usage"):
            usage = d["usage"]
    after = metrics(base)
    ct = usage["completion_tokens"] if usage else 0
    pt = usage["prompt_tokens"] if usage else 0
    toks = "".join(text).split()
    # A sample is valid only if we saw a first streamed chunk AND generated real output. Without a
    # chunk the decode clock is meaningless (previously produced ~1e11 tok/s), so mark it invalid.
    valid = tf is not None and ct >= 16 and last > tf
    dec = max((last - tf) if tf is not None else 0.0, 1e-9)
    grams = [" ".join(toks[i:i + 8]) for i in range(max(0, len(toks) - 7))]
    dup = 1 - (len(set(grams)) / max(1, len(grams)))
    rec = {"prompt_tokens": pt, "completion_tokens": ct, "valid": valid,
           "ttft_ms": (tf - t0) * 1000 if tf else None,
           "decode_tps": (ct / dec) if valid else None, "dup8": dup,
           "acc_per_draft": None, "ms_step": None, "accept_rate": None}
    if before and after and valid:
        dn, dd, da = (after[k] - before[k] for k in ("n", "d", "a"))
        if dn > 0:
            rec["acc_per_draft"] = da / dn
            rec["accept_rate"] = da / max(dd, 1)
            rec["ms_step"] = 1000 * (da / dn + 1) / (ct / dec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://localhost:6564"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL"))
    ap.add_argument("--gen", type=int, default=int(os.environ.get("BENCH_GEN", "256")))
    ap.add_argument("--reps", type=int, default=int(os.environ.get("BENCH_REPS", "2")))
    ap.add_argument("--temp", type=float, default=float(os.environ.get("BENCH_TEMP", "0.7")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("BENCH_SEED", "1")))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", "1")))
    ap.add_argument("--thinking", type=int, default=int(os.environ.get("BENCH_THINKING", "0")),
                    help="1 = enable qwen3 thinking (mirrors production chat); 0 = off (repeatable)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.model:
        try:
            a.model = json.load(urllib.request.urlopen(a.base + "/v1/models", timeout=10))["data"][0]["id"]
        except Exception as e:
            sys.exit(f"could not discover a model at {a.base}: {e}")
    # A tiny request first: it forces one engine step so a live knob-file change (benched A/B arms on
    # one server) is reloaded before the measured requests. Non-streaming + ignored.
    for _ in range(a.warmup):
        try:
            body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 8, "temperature": a.temp, "seed": a.seed,
                               "chat_template_kwargs": {"enable_thinking": bool(a.thinking)}}).encode()
            urllib.request.urlopen(
                urllib.request.Request(a.base + "/v1/chat/completions", body, {"Content-Type": "application/json"}),
                timeout=a.timeout).read()
        except Exception:
            pass
    print(f"=== quick decode probe ({a.model}, gen {a.gen}, {a.reps} rep) {a.base} ===", flush=True)
    results = {}
    for cat, prompt in PROMPTS:
        for r in range(a.reps):
            # vary the seed per rep: a fixed seed makes every rep the same generation, so reps would
            # just repeat one content sample instead of measuring content variance.
            rec = one(a.base, a.model, prompt, a.gen, a.temp, a.seed + 7919 * r, a.timeout, a.thinking)
            results.setdefault(cat, []).append(rec)
            extra = ""
            if rec["acc_per_draft"] is not None:
                extra = (f" | acc/draft {rec['acc_per_draft']:5.3f} | {rec['ms_step']:6.2f} ms/step"
                         f" | rate {100*rec['accept_rate']:5.2f}%")
            tt = f"{rec['ttft_ms']:7.0f}" if rec["ttft_ms"] is not None else "    n/a"
            dd = f"{rec['decode_tps']:7.1f}" if rec["decode_tps"] is not None else "    n/a"
            flag = "" if rec["valid"] else " [invalid: no streamed chunk]"
            print(f"  {cat:>6} r{r} | prompt {rec['prompt_tokens']:>5} | gen {rec['completion_tokens']:4d} "
                  f"| TTFT {tt} ms | decode {dd} tok/s | dup-8gram {rec['dup8']*100:4.1f}%{extra}{flag}",
                  flush=True)

    def avg(cat, key):
        vals = [x[key] for x in results[cat] if x["valid"] and x[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    agg = {c: {"decode_tps": avg(c, "decode_tps"), "acc_per_draft": avg(c, "acc_per_draft"),
               "ms_step": avg(c, "ms_step"), "dup8": avg(c, "dup8")} for c in results}
    # only valid samples (a streamed chunk was seen and real output was produced) enter the combined.
    good = [x for xs in results.values() for x in xs if x["valid"] and x["decode_tps"]]
    tot_ct = sum(x["completion_tokens"] for x in good)
    tot_dec = sum(x["completion_tokens"] / max(x["decode_tps"], 1e-9) for x in good)
    print("--- summary (per-category mean) ---")
    for c in results:
        print(f"  {c:>6} | decode {agg[c]['decode_tps']:7.1f} tok/s | acc/draft {agg[c]['acc_per_draft']:5.3f}"
              f" | {agg[c]['ms_step']:6.2f} ms/step | dup8 {agg[c]['dup8']*100:4.1f}%")
    combined = tot_ct / max(tot_dec, 1e-9)
    print(f"  COMBINED decode {combined:7.1f} tok/s over {tot_ct} tokens")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"base": a.base, "model": a.model, "gen": a.gen, "reps": a.reps,
                   "combined_decode_tps": combined, "categories": agg, "raw": results},
                  open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
