#!/usr/bin/env python3
"""Same-prompt A/B for the n-gram tail: decode throughput per prompt type, one instance, direct.

Run the SAME prompts under two server configs (e.g. RADIANCE_DRAFT_NGRAM=0 vs 1) and compare tok/s.
Hits the instance directly (bypasses the router) so the result is not mixed across replicas.

  BASE=http://172.18.0.20:8000 python3 aijuus/ngram_ab_probe.py
"""
import json
import os
import time
import urllib.request

BASE = os.environ.get("BASE", "http://172.18.0.20:8000")
MODEL = os.environ.get("MODEL", "mtp-27B-MXFP4-Thinkingcap")
KEY = os.environ.get("KEY", "juup-123")
REPS = int(os.environ.get("REPS", "3"))
MAXT = int(os.environ.get("MAXT", "1024"))

REPEAT = ("Write a single Python file inventory.py with exactly 60 functions named report_<category>, "
          "each with a one-line docstring, the typed signature def report_<category>(items: list[dict]) "
          "-> dict:, and a body that sums item['qty']*item['price'] into total and returns "
          "{'category': '<category>', 'total': total}. Categories in order: "
          + ", ".join(f"cat{i}" for i in range(60)) + ". Output only the code.")
NOVEL = ("Write an original 1200-word short story about a lighthouse keeper who finds a message in a "
         "bottle that predicts tomorrow's weather. Be atmospheric and specific; avoid repeated phrases.")
AGENT = ("Continue this JSON tool-schema array in the SAME format, adding one entry per tool named "
         + ", ".join(f"tool_{i}" for i in range(40)) + ": "
         "[{\"name\":\"tool_0\",\"parameters\":{\"$schema\":\"https://json-schema.org/draft/2020-12/"
         "schema\",\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
         "\"required\":[\"path\"]}}]")

PROMPTS = {"repeat": REPEAT, "novel": NOVEL, "agent": AGENT}


def run(name, prompt):
    toks = 0
    t0 = time.time()
    fins = {}
    for _ in range(REPS):
        body = json.dumps({"model": MODEL, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
                           "max_tokens": MAXT, "messages": [{"role": "user", "content": prompt}]}).encode()
        r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json",
                                            "Authorization": "Bearer " + KEY})
        d = json.load(urllib.request.urlopen(r, timeout=900))
        toks += d["usage"]["completion_tokens"]
        fr = d["choices"][0]["finish_reason"]
        fins[fr] = fins.get(fr, 0) + 1
    dt = time.time() - t0
    print(f"{name:7s} reps={REPS} tok={toks:6d} {dt:7.1f}s  {toks / dt:6.1f} tok/s  finish={fins}",
          flush=True)


if __name__ == "__main__":
    print(f"base={BASE} model={MODEL} max_tokens={MAXT}", flush=True)
    for n, p in PROMPTS.items():
        run(n, p)
