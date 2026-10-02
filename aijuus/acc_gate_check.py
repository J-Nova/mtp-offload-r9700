#!/usr/bin/env python3
"""One-shot MTP acceptance discriminator + acceptance-gate verifier.

WHY. A live stream can show near-zero per-position acceptance (p0 ~0.03) for two very
different reasons:
  (a) HIGH-ENTROPY output -- the target itself is near-uniform, so a greedy MTP draft
      cannot match. Expected; the only fix is to stop drafting (RADIANCE_ACC_GATE).
  (b) DRAFT/STATE DIVERGENCE -- the target is confident but the drafter points elsewhere
      (stale hidden state, slot-mapping bug, corrupted GDN state). A correctness bug the
      gate would only hide.
This tells them apart without a rebuild: a deterministic greedy (temperature 0) run over
PREDICTABLE text must accept near the top of the range. If greedy acceptance is high, the
drafter is healthy and the live collapse is (a). If greedy is ALSO near zero, it is (b).

It also reads the per-position counters so the acceptance-gate effect is visible: after the
gate engages on a low-EMA lone stream, `drafts`/step should fall (K 5 -> 2) while `accepted`
stays similar, i.e. accepted/draft rises sharply.

Usage:
  ./acc_gate_check.py --base http://172.18.0.20:8000 [--model mtp-27B-MXFP4-blend]
  ./acc_gate_check.py --base ... --regime greedy|sampled|both --max-tokens 512
"""
import argparse
import json
import sys
import time
import urllib.request

POS = [f"vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine=\"0\",model_name=\"{0}\",position=\"{i}\"}}"
       for i in range(8)]


def _get(url, key, timeout=10):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def read_metrics(base, key):
    txt = _get(base.rstrip("/") + "/metrics", key)
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft_tokens"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.rsplit(" ", 1)[1])
        elif "num_accepted_tokens_per_pos_total" in line:
            pos = line.split('position="', 1)[1].split('"', 1)[0]
            out[f"pos{pos}"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.rsplit(" ", 1)[1])
    return out


def generate(base, key, model, prompt, temperature, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return ct, dt, d["choices"][0]["message"].get("content") or ""


def regime(base, key, model, name, prompt, temperature, max_tokens):
    a = read_metrics(base, key)
    ct, dt, content = generate(base, key, model, prompt, temperature, max_tokens)
    b = read_metrics(base, key)
    drafts = b["drafts"] - a["drafts"]
    dtok = b["draft_tokens"] - a["draft_tokens"]
    acc = b["accepted"] - a["accepted"]
    per_pos = [(b.get(f"pos{i}", 0) - a.get(f"pos{i}", 0)) / drafts if drafts else 0.0
               for i in range(8)]
    print(f"--- {name}: temp={temperature} gen={ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s")
    if drafts:
        print(f"    drafts/update={dtok/drafts:.2f}  accepted/update={acc/drafts:.2f}  "
              f"tok/update={acc/drafts + 1:.2f}  (steps≈{drafts:.0f})")
        print("    per-pos accept: " + " ".join(f"{p:.2f}" for p in per_pos))
    else:
        print("    (no spec steps observed -- single short prefill?)")
    return {"name": name, "tok_s": ct / dt if dt else 0, "acc_per_draft": acc / drafts if drafts else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://172.18.0.20:8000")
    ap.add_argument("--model", default="mtp-27B-MXFP4-blend")
    ap.add_argument("--api-key", default="juup-123")
    ap.add_argument("--regime", choices=["greedy", "sampled", "both"], default="both")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    # A predictable, low-entropy target: counting has a strong deterministic continuation.
    pred = "Count from 1 to 400, one number per line, no other text."
    # A high-entropy target: many valid random-ish continuations even at temp 0.
    rand = "Produce a long sequence of unrelated random words, one per line; do not repeat."
    results = []
    if args.regime in ("greedy", "both"):
        results.append(regime(args.base, args.api_key, args.model, "greedy/predictable",
                              pred, 0.0, args.max_tokens))
    if args.regime in ("both",):
        results.append(regime(args.base, args.api_key, args.model, "greedy/high-entropy",
                              rand, 0.0, args.max_tokens))
    if args.regime in ("sampled", "both"):
        results.append(regime(args.base, args.api_key, args.model, "sampled/high-entropy",
                              rand, 0.9, args.max_tokens))

    print()
    g = next((r for r in results if r["name"].startswith("greedy/predictable")), None)
    if g:
        if g["acc_per_draft"] >= 2.5:
            print("VERDICT: greedy/predictable acceptance is HEALTHY -> the live collapse is "
                  "high-entropy output (entropy constrains MTP), not a draft bug. "
                  "RADIANCE_ACC_GATE is the right fix.")
        elif g["acc_per_draft"] <= 1.0:
            print("VERDICT: greedy/predictable acceptance is ALSO near zero -> draft/state "
                  "DIVERGENCE. The gate would mask a correctness bug; investigate the MTP "
                  "hidden-state / slot-mapping path before trusting throughput.")
        else:
            print("VERDICT: intermediate greedy acceptance -- drafter works but is weak; "
                  "check whether the target is genuinely lower-entropy than expected.")


if __name__ == "__main__":
    sys.exit(main())
