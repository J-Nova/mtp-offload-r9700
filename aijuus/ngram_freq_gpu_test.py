#!/usr/bin/env python3
"""GPU correctness test for match_count (F6 frequency kernel), vs a CPU reference.

Runs the REAL Triton `match_gpu` (to fill key1) then `match_count` over random rows, and diffs the
(occ, agree) counts to a CPU longest-suffix reference. Mirrors batched_ngram_equiv.py's structure.

  cd /patches && python3 aijuus/ngram_freq_gpu_test.py
"""
import numpy as np
import torch

import radiance_draft_gpu as gpu

cap = 8
dev = "cuda"
torch.manual_seed(1234)
rng = np.random.default_rng(1234)
MAXL = gpu._MAXL
MIN = gpu._MIN


def cpu_ref(ctx, lens, bases):
    B, _ = ctx.shape
    occ = np.zeros(B, dtype=np.int64)
    agree = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n, b = int(lens[i]), int(bases[i])
        Lq = {}
        for q in range(b, n - 1):
            L = 0
            for k in range(MAXL):
                sk = n - 1 - k
                qi = q - k
                if sk < 0 or qi < b or ctx[i, qi] != ctx[i, sk]:
                    break
                L += 1
            Lq[q] = L
        if not Lq:
            continue
        Lmax = max(Lq.values())
        if Lmax < MIN:
            continue
        qref = max(q for q, L in Lq.items() if L == Lmax)
        cref = int(ctx[i, qref + 1])
        o = a = 0
        for q, L in Lq.items():
            if q != qref and L >= Lmax:
                o += 1
                if int(ctx[i, q + 1]) == cref:
                    a += 1
        occ[i], agree[i] = o, a
    return occ, agree


fails = 0
it_n = 40
for it in range(it_n):
    B = int(rng.integers(1, 9))
    ML = 3200
    ctx = np.zeros((B, ML), dtype=np.int32)
    lens, wins = [], []
    for i in range(B):
        n = int(rng.integers(20, 3000))
        row = rng.integers(0, 50, size=n).astype(np.int32)
        if n > 12:
            k = int(rng.integers(0, n - 8))
            row[k : k + 8] = row[n - 8 : n]
        ctx[i, :n] = row
        lens.append(n)
        wins.append(int(rng.choice([0, 256, 1024])) if n > 1024 else 0)
    bases = [np.maximum(0, lens[i] - wins[i]) if wins[i] else 0 for i in range(B)]

    ctx_t = torch.from_numpy(ctx).to(dev)
    n_t = torch.tensor(lens, dtype=torch.int32, device=dev)
    base_t = torch.tensor(bases, dtype=torch.int32, device=dev)
    nc = 2 * max(gpu._nblk(lens[i], wins[i]) for i in range(B))

    buf = gpu.make_match_buffers(B, cap, nc, dev)
    gpu.match_gpu(ctx_t, n_t, cap, 0, base_t, nc, False, buf, MAXL)
    buf["occ"].zero_()
    buf["agree"].zero_()
    gpu.match_count(ctx_t, n_t, buf["key1"], base_t, buf["occ"], buf["agree"], nc)
    torch.cuda.synchronize()

    occ = buf["occ"].cpu().numpy().astype(np.int64)
    agree = buf["agree"].cpu().numpy().astype(np.int64)
    ro, ra = cpu_ref(ctx, lens, bases)

    if not (np.array_equal(occ, ro) and np.array_equal(agree, ra)):
        fails += 1
        print(f"MISMATCH it={it} B={B} lens={lens} wins={wins}")
        print("  occ   gpu", occ, "\n  occ   cpu", ro)
        print("  agree gpu", agree, "\n  agree cpu", ra)
        break

print(f"MATCH_COUNT EQUIV ({it_n} iters, B 1..8): {'PASS' if fails == 0 else 'FAIL'}")
raise SystemExit(1 if fails else 0)
