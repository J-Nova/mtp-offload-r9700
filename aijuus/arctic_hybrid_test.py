#!/usr/bin/env python3
"""Offline test for the Arctic SuffixDecoding backend in patch_dynamic_depth.py (no GPU, no engine).

Extracts + execs the injected RUNNER_TAIL with RADIANCE_DRAFT_NGRAM_BACKEND=arctic, builds a fake
InputBatch / runner matching the vLLM 0.29 runtime shape, and checks:
  * a context whose tail repeats earlier -> the Arctic draft is taken (score >= tau);
  * a novel context -> MTP draft kept;
  * tau above the score -> MTP kept.

  cd /patches && PYTHONPATH=/patches python3 aijuus/arctic_hybrid_test.py
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "patch_dynamic_depth.py"


def load_tail():
    for node in ast.parse(SRC.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "RUNNER_TAIL" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("RUNNER_TAIL not found")


def build(env):
    old = dict(os.environ)
    os.environ.update(env)
    ns = {"__name__": "arctic_tail_probe"}
    try:
        exec(load_tail(), ns)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return ns["_radiance_arctic_extend"]


class IB:
    """Mimics vLLM 0.29 v1/worker/gpu/input_batch.py InputBatch (batch-row-indexed arrays)."""

    def __init__(self, toks, plen):
        n = len(toks)
        self.req_ids = ["r1"]
        self.idx_mapping_np = np.asarray([0], dtype=np.intp)
        self.num_computed_tokens_np = np.asarray([n], dtype=np.int32)
        self.prefill_len_np = np.asarray([plen], dtype=np.int32)


class _Atok:
    def __init__(self, arr):
        self.gpu = torch.tensor(arr, dtype=torch.int64)  # [num_req_states, max_len]


class _RS:
    def __init__(self, arr):
        self.all_token_ids = _Atok(arr)


class Runner:
    num_speculative_steps = 8

    def __init__(self, arr):
        self.req_states = _RS(arr)


fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")


# context: [1..8, 42..49, 1..8]; the tail (1..8) first appeared at 0 followed by 42..49
REPEAT = np.asarray([list(range(1, 9)) + list(range(42, 50)) + list(range(1, 9))], dtype=np.int64).reshape(1, -1)
NOVEL = np.asarray([list(range(100, 130))], dtype=np.int64).reshape(1, -1)
MTP = torch.tensor([[9, 9, 9, 9, 9]])


def vals(t):
    return [x for x in t.cpu().numpy().tolist()[0] if x != -1]


ext = build({"RADIANCE_DRAFT_NGRAM_BACKEND": "arctic", "RADIANCE_DRAFT_NGRAM_TAU": "1.0"})
check("repeat -> arctic draft taken", vals(ext(Runner(REPEAT), IB(REPEAT[0], len(REPEAT[0])), MTP.clone()))[:4],
      [42, 43, 44, 45])
check("novel -> MTP kept", vals(ext(Runner(NOVEL), IB(NOVEL[0], len(NOVEL[0])), MTP.clone())), [9, 9, 9, 9, 9])

ext = build({"RADIANCE_DRAFT_NGRAM_BACKEND": "arctic", "RADIANCE_DRAFT_NGRAM_TAU": "100.0"})
check("tau too high -> MTP kept", vals(ext(Runner(REPEAT), IB(REPEAT[0], len(REPEAT[0])), MTP.clone())),
      [9, 9, 9, 9, 9])

print("ARCTIC HYBRID TEST:", "PASS" if fails == 0 else "FAIL")
sys.exit(1 if fails else 0)
