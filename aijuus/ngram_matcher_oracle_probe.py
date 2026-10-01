#!/usr/bin/env python3
"""Decisive diagnostic: does OUR Triton matcher propose the same continuation as the true
longest-suffix oracle? (runs in the container; GPU + local API)

Generate a sequence, then for many decode positions compute the oracle continuation on CPU and the
matcher's top-1 continuation via gpu.match_gpu, and compare each to the actually-generated token.
If gpu_cont != oracle_cont, our matcher (not the policy) is the bug.

  docker exec -i <c> sh -c 'cd /patches && PYTHONPATH=/patches python3 aijuus/ngram_matcher_oracle_probe.py'
"""
import json
import os
import sys
import urllib.request

import numpy as np
import torch

import radiance_draft_gpu as gpu

BASE = "http://localhost:8000"
MODEL = "mtp-27B-MXFP4-blend"
KEY = "juup-123"
TEMP = float(os.environ.get("TEMP", "0.7"))
MAXL = gpu._MAXL
STRONG = 8
CAP = 8
MODEL_DIR = "/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ"
PROMPT = os.environ.get("PROMPT", "agent")

PROMPTS = {
    "agent": ("Continue this JSON tool-schema array in the SAME format, adding one entry per tool named "
              + ", ".join(f"tool_{i}" for i in range(40)) + ": "
              "[{\"name\":\"tool_0\",\"parameters\":{\"$schema\":\"https://json-schema.org/draft/2020-12/"
              "schema\",\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
              "\"required\":[\"path\"]}}]"),
    "repeat": ("Write a single Python file inventory.py with exactly 60 functions named report_<category>, "
               "each with a one-line docstring. Categories in order: " + ", ".join(f"cat{i}" for i in range(60))
               + ". Output only the code."),
}


def gen(prompt):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning") or "")


def oracle(tokens, t):
    best, best_q = 0, -1
    for L in range(1, min(MAXL, t) + 1):
        s = tokens[t - L:t]
        found = -1
        for q in range(L - 1, t - 1):
            if tokens[q - L + 1:q + 1] == s:
                found = q
        if found >= 0:
            best, best_q = L, found
    if best >= STRONG and 0 <= best_q + 1 < t:
        return best, tokens[best_q + 1]
    return best, None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text = gen(PROMPTS[PROMPT])
    pids = tok(PROMPTS[PROMPT], add_special_tokens=False)["input_ids"]
    seq = pids + tok(text, add_special_tokens=False)["input_ids"]
    P = len(pids)
    ts = list(range(P + 1, len(seq)))
    # batched: all rows are the same sequence; row i answers the position ts[i]
    R = len(ts)
    ML = len(seq)
    ctx = np.tile(np.asarray(seq, dtype=np.int32), (R, 1))
    n_arr = np.asarray(ts, dtype=np.int32)
    base = np.zeros(R, dtype=np.int32)
    nc = 2 * max(gpu._nblk(int(n), 0) for n in ts)
    ctx_t = torch.from_numpy(ctx).cuda()
    n_t = torch.from_numpy(n_arr).cuda()
    base_t = torch.from_numpy(base).cuda()
    buf = gpu.make_match_buffers(R, CAP, nc, "cuda")
    gpu.match_gpu(ctx_t, n_t, CAP, 0, base_t, nc, False, buf, MAXL)
    torch.cuda.synchronize()
    pack = buf["pack"].cpu().numpy()
    gpu_cont = pack[:, 0]  # first continuation token of the top-1 match

    n_ora = n_same = n_gpu_hit = n_ora_hit = 0
    diffs = 0
    for i, t in enumerate(ts):
        L, oc = oracle(seq, t)
        actual = seq[t]
        if oc is None:
            continue
        n_ora += 1
        gc = int(gpu_cont[i])
        if gc == oc:
            n_same += 1
        else:
            diffs += 1
            if diffs <= 8:
                print(f"  t={t} mlen={L} oracle_cont={oc} gpu_cont={gc} actual={actual}")
        if gc == actual:
            n_gpu_hit += 1
        if oc == actual:
            n_ora_hit += 1
    print(f"PROMPT={PROMPT} temp={TEMP} positions_with_match={n_ora}")
    print(f"  matcher==oracle: {n_same}/{n_ora} ({n_same / max(1, n_ora):.1%})")
    print(f"  oracle_cont==actual: {n_ora_hit}/{n_ora} ({n_ora_hit / max(1, n_ora):.1%})")
    print(f"  gpu_cont==actual:    {n_gpu_hit}/{n_ora} ({n_gpu_hit / max(1, n_ora):.1%})")


if __name__ == "__main__":
    sys.exit(main())
