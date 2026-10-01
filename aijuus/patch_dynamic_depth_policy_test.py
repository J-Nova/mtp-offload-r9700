#!/usr/bin/env python3
"""Offline unit test for the n-gram trust gate in patch_dynamic_depth.py (no model, no GPU).

`_radiance_ngram_ok` lives inside the RUNNER_TAIL string that the patch injects into
vllm's model_runner.py. This test extracts that string from the patch source, execs it (which also
proves the injected code parses), and checks the gate against:
  * the boilerplate false-positive (long match, continuation disagrees with MTP) -> must decline,
  * the echo true-positive (continuation agrees, top-2 agree) -> must accept,
for AGREE/DET on and off.

  python3 aijuus/patch_dynamic_depth_policy_test.py
"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "patch_dynamic_depth.py"


def load_tail():
    tree = ast.parse(SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RUNNER_TAIL" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("RUNNER_TAIL not found in patch_dynamic_depth.py")


def build(env):
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    ns = {"__name__": "radiance_tail_probe"}
    try:
        exec(load_tail(), ns)          # defines _radiance_ngram_ok + _radiance_ngram_extend
    finally:
        os.environ.clear()
        os.environ.update(old)
    return ns["_radiance_ngram_ok"]


fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")


# --- defaults: AGREE + DET on ---
ok = build({})
check("boilerplate declined (AGREE: cont!=mtp)", ok(32, 8, 776, 32, 8, 900, 12), False)
check("echo accepted (AGREE + DET)", ok(32, 8, 12, 32, 8, 12, 12), True)
check("short match declined", ok(4, 4, 12, 4, 4, 12, 12), False)
check("empty continuation declined", ok(32, 0, -1, 32, 0, -1, 12), False)
check("short 2nd match does not veto a long match", ok(32, 8, 12, 4, 4, 900, 12), True)

# --- pre-fix behaviour: both gates off -> long match always wins ---
ok = build({"RADIANCE_DRAFT_NGRAM_AGREE": "0", "RADIANCE_DRAFT_NGRAM_DET": "0"})
check("pre-fix: boilerplate taken", ok(32, 8, 776, 32, 8, 900, 12), True)

# --- DET only ---
ok = build({"RADIANCE_DRAFT_NGRAM_AGREE": "0", "RADIANCE_DRAFT_NGRAM_DET": "1"})
check("DET rejects disagreement (strong 2nd)", ok(32, 8, 776, 32, 8, 900, 12), False)
check("DET accepts agreement", ok(32, 8, 12, 32, 8, 12, 12), True)
check("DET ignores a short 2nd match", ok(32, 8, 12, 4, 4, 900, 12), True)

# --- AGREE only ---
ok = build({"RADIANCE_DRAFT_NGRAM_AGREE": "1", "RADIANCE_DRAFT_NGRAM_DET": "0"})
check("AGREE rejects disagreement", ok(32, 8, 776, 9, 8, 776, 12), False)
check("AGREE accepts agreement", ok(32, 8, 12, 9, 8, 776, 12), True)

# --- STRONG=0 keeps its kill-switch meaning ---
ok = build({"RADIANCE_DRAFT_NGRAM_STRONG": "0"})
check("STRONG=0 disables the tail", ok(32, 8, 12, 32, 8, 12, 12), False)

# --- F6 frequency override (agree_other, occ_other) ---
ok = build({})
check("freq: repeated continuation -> taken", ok(32, 8, 776, 32, 8, 900, 12, 3, 3), True)
check("freq: boilerplate (many occ, none agree) -> declined", ok(32, 8, 776, 32, 8, 900, 12, 0, 50), False)
check("freq: below min_freq -> declined", ok(32, 8, 776, 32, 8, 900, 12, 0, 5), False)
ok = build({"RADIANCE_DRAFT_NGRAM_FREQ": "0"})
check("freq off: no override", ok(32, 8, 776, 32, 8, 900, 12, 3, 3), False)
check("freq off: fast path still taken", ok(32, 8, 12, 32, 8, 12, 12), True)

# --- F7 prefix-preserving extension ---
_ns = {}
exec(load_tail(), _ns)
ext = _ns["_radiance_ngram_extend_row"]
check("ext appends on full-prefix agreement", ext([9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 42, 43], 7, 8, 5, 8), [9, 9, 9, 9, 9, 42, 43])
check("ext declines on disagreement", ext([9, 9, 9, 9, 9], [9, 9, 9, 1, 9, 42, 43], 7, 8, 5, 8), [9, 9, 9, 9, 9])
check("ext declines when clen<=K", ext([9, 9, 9, 9, 9], [9, 9, 9, 9, 9], 5, 8, 5, 8), [9, 9, 9, 9, 9])
check("ext declines when mlen<STRONG", ext([9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 42, 43], 7, 4, 5, 8), [9, 9, 9, 9, 9])

print("POLICY TEST:", "PASS" if fails == 0 else "FAIL")
raise SystemExit(1 if fails else 0)
