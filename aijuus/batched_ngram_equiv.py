#!/usr/bin/env python3
"""Batched n-gram matcher equivalence + HSA-safety test (standalone; no model, no server).

Runs the REAL Triton match_gpu two ways over the same random rows:
  * per-row B=1 (the proven path currently in service), and
  * one batched B=N launch with a single max-block grid (the C1/C2 path),
and asserts the [B, 2*nspec+_META] packs are bit-identical, for both full-context and windowed
search. Repeated across many random B/n/window draws to exercise the masked extra blocks and to try
to trip the gfx1201 HSA fault at B>=2.

  python3 batched_ngram_equiv.py
"""
import numpy as np
import torch

import radiance_draft_gpu as gpu

cap = 8
dev = "cuda"
torch.manual_seed(1234)
rng = np.random.default_rng(1234)
fails = 0
it_n = 40

for it in range(it_n):
    B = int(rng.integers(1, 9))
    ML = 3200
    ctx = np.zeros((B, ML), dtype=np.int32)
    lens = []
    wins = []
    for i in range(B):
        n = int(rng.integers(20, 3000))
        row = rng.integers(0, 50, size=n).astype(np.int32)  # small vocab -> many real matches
        if n > 12:
            k = int(rng.integers(0, n - 8))
            row[k : k + 8] = row[n - 8 : n]  # seed a long suffix repeat
        ctx[i, :n] = row
        lens.append(n)
        wins.append(int(rng.choice([0, 256, 1024])) if n > 1024 else 0)
    ctx_t = torch.from_numpy(ctx).to(dev)
    n_t = torch.tensor(lens, dtype=torch.int32, device=dev)

    nblks = [gpu._nblk(lens[i], wins[i]) for i in range(B)]

    # per-row reference (B=1)
    ref = np.zeros((B, 2 * cap + gpu._META), dtype=np.int64)
    b1 = gpu.make_match_buffers(1, cap, 2 * max(nblks), dev)
    for i in range(B):
        base = np.maximum(0, lens[i] - wins[i]) if wins[i] else 0
        b1["base"].fill_(int(base))
        pk = gpu.match_gpu(
            ctx_t[i : i + 1], n_t[i : i + 1], cap, 0, b1["base"], 2 * nblks[i], False, b1, gpu._MAXL
        )
        ref[i] = pk.cpu().numpy()[0]

    # batched (B=N, max-block grid)
    nc = 2 * max(nblks)
    bN = gpu.make_match_buffers(B, cap, nc, dev)
    base_s = torch.tensor(
        [np.maximum(0, lens[i] - wins[i]) if wins[i] else 0 for i in range(B)],
        dtype=torch.int32, device=dev,
    )
    pkb = gpu.match_gpu(ctx_t, n_t, cap, 0, base_s, nc, False, bN, gpu._MAXL).cpu().numpy()

    if not np.array_equal(ref, pkb):
        fails += 1
        bad = [i for i in range(B) if not np.array_equal(ref[i], pkb[i])]
        print(f"MISMATCH it={it} B={B} lens={lens} wins={wins} bad={bad}")
        for i in bad[:2]:
            print("  ref", ref[i], "\n  bat", pkb[i])
        break
    torch.cuda.synchronize()

print(f"BATCHED NGRAM EQUIV ({it_n} iters, B 1..8): {'PASS' if fails == 0 else 'FAIL'}")
raise SystemExit(1 if fails else 0)
