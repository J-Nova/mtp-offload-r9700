#!/usr/bin/env python3
"""Ceiling probe for prompt-lookup on this workload (runs inside the container).

For fixed prompts, generate output, tokenize prompt+output, then at every decode position find the
longest suffix match within the preceding context and measure:
  * match availability: how often a match of length >= STRONG exists;
  * oracle hit: does the matched continuation's first token equal what the model actually generated;
  * oracle accepted length: how many suffix tokens (up to cap) would have been accepted if proposed.
If the oracle accepted length on match positions is ~<= MTP's own accepted length (~3-4), prompt-lookup
has no headroom on this workload.

  docker exec -i <c> sh -c 'cd /patches && python3 aijuus/ngram_ceiling_probe.py'
"""
import json
import os
import sys
import urllib.request

import numpy as np

BASE = "http://localhost:8000"
MODEL = "mtp-27B-MXFP4-blend"
KEY = "juup-123"
TEMP = float(os.environ.get("TEMP", "0.7"))
MAXL = 32
STRONG = 8
CAP = 8
MODEL_DIR = "/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ"

REPEAT = ("Write a single Python file inventory.py with exactly 60 functions named report_<category>, "
          "each with a one-line docstring, the typed signature def report_<category>(items: list[dict]) "
          "-> dict:, and a body that sums item['qty']*item['price'] into total and returns "
          "{'category': '<category>', 'total': total}. Categories in order: "
          + ", ".join(f"cat{i}" for i in range(60)) + ". Output only the code.")
AGENT = ("Continue this JSON tool-schema array in the SAME format, adding one entry per tool named "
         + ", ".join(f"tool_{i}" for i in range(40)) + ": "
         "[{\"name\":\"tool_0\",\"parameters\":{\"$schema\":\"https://json-schema.org/draft/2020-12/"
         "schema\",\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
         "\"required\":[\"path\"]}}]")
NOVEL = ("Write an original 1200-word short story about a lighthouse keeper who finds a message in a "
         "bottle that predicts tomorrow's weather.")

PROMPTS = {"repeat": REPEAT, "agent": AGENT, "novel": NOVEL}


def gen(prompt):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": 800,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning") or "")


def best_match(tokens, t):
    """Longest suffix of tokens[:t] that appeared earlier ending at some q<t-1; return (L, q)."""
    best, best_q = 0, -1
    lim = min(MAXL, t)
    for L in range(1, lim + 1):
        s = tokens[t - L:t]
        found = -1
        for q in range(L - 1, t - 1):
            if tokens[q - L + 1:q + 1] == s:
                found = q
        if found >= 0:
            best, best_q = L, found
    return best, best_q


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    for name, prompt in PROMPTS.items():
        text = gen(prompt)
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        oids = tok(text, add_special_tokens=False)["input_ids"]
        seq = list(pids) + list(oids)
        P = len(pids)
        n_match = n_hit = 0
        acc_lens = []
        for t in range(P + 1, len(seq)):
            L, q = best_match(seq, t)
            if L >= STRONG and 0 <= q + 1 < t:
                n_match += 1
                cont = seq[q + 1:q + 1 + CAP]
                if cont and cont[0] == seq[t]:
                    n_hit += 1
                # oracle accepted length = common prefix of proposed tail and actual future tokens
                al = 0
                for j in range(min(CAP, len(cont), len(seq) - t)):
                    if cont[j] == seq[t + j]:
                        al += 1
                    else:
                        break
                acc_lens.append(al)
        dec_steps = len(seq) - P - 1
        mr = n_match / max(1, dec_steps)
        mean_al = float(np.mean(acc_lens)) if acc_lens else 0.0
        print(f"{name:7s} gen_tok={len(oids):5d} match>=8 on {mr:5.1%} of steps; "
              f"of those: first-token hit {n_hit / max(1, n_match):5.1%}, "
              f"mean oracle accepted len {mean_al:4.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
