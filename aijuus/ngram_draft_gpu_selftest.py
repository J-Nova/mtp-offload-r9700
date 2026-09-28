#!/usr/bin/env python3
"""GPU correctness test for the RADIANCE n-gram matcher kernels (radiance_draft_gpu.py).

Run this INSIDE the vllm-radiance container (torch + triton + a gfx1201 GPU). It executes the real
Triton kernels -- _match_scan, _match_scan_x, _match_top2, _match_gather, via match_gpu -- and
compares their output against a pure-Python longest-suffix reference, the same semantics the CPU
selftest (ngram_draft_selftest.py) pins. It does NOT load a model or start a server.

  python3 ngram_draft_gpu_selftest.py

Prints one line per case and "GPU NGRAM SELFTEST: ALL PASS" (or a diff) at the end. Exit code 0/1.
"""
import sys

import numpy as np
import torch

import radiance_draft_gpu as gpu

MAXL, MIN, SH, RH = gpu._MAXL, gpu._MIN, gpu._SH, gpu._RH
LSH = SH + RH
NSPEC = 8


def enc(L, row, q):
    return (int(L) << LSH) | (int(row) << SH) | int(q)


def dec(k):
    if k == 0:
        return 0, 0, 0
    return (k >> LSH, (k >> SH) & ((1 << RH) - 1), k & ((1 << SH) - 1))


def ref_match(rows, lens, nspec=NSPEC, window=0, cross=False):
    """Per row: ((mlen, cont, clen), (mlen, cont, clen)) top-2. Mirrors the kernel: length first,
    then more-recent end position; continuation = tokens after the matched occurrence, capped at
    nspec and at the row end. Keys use the same int64 encoding the kernels use."""
    out = []
    for i in range(len(rows)):
        n = lens[i]
        base = max(0, n - window) if window > 0 else 0
        keys = []
        for j in (range(len(rows)) if cross else [i]):
            nj, rowj = lens[j], rows[j]
            for q in range(base, nj - 1):
                L = 0
                for k in range(MAXL):
                    sk, qi = n - 1 - k, q - k
                    if sk < 0 or qi < 0 or (qi < base and not cross):
                        break
                    if rowj[qi] != rows[i][sk]:
                        break
                    L += 1
                if L >= MIN:
                    keys.append(enc(L, j, q))
        keys.sort(reverse=True)
        cands = []
        for key in keys[:2]:
            L, j, q = dec(key)
            cands.append((L, [int(x) for x in rows[j][q + 1:q + 1 + nspec]], min(nspec, lens[j] - (q + 1))))
        while len(cands) < 2:
            cands.append((0, [], 0))
        out.append((cands[0], cands[1]))
    return out


def run_match(rows, lens, window=0, cross=False, nspec=NSPEC, device="cuda"):
    B = len(rows)
    nmax = max(lens)
    ctx = torch.zeros(B, nmax, dtype=torch.int32, device=device)
    for i, r in enumerate(rows):
        ctx[i, :len(r)] = torch.tensor(r, dtype=torch.int32, device=device)
    n_arr = torch.tensor(lens, dtype=torch.int32, device=device)
    nc_full = (2 * B * gpu._nblk(nmax, 0)) if cross else (2 * gpu._nblk(nmax, 0))
    mb = gpu.make_match_buffers(B, nspec, nc_full, device)
    base = mb["base"]
    base_np = np.maximum(0, np.asarray(lens) - window) if window > 0 else np.zeros(B, np.int64)
    base.copy_(torch.from_numpy(base_np.astype(np.int32)))
    active_nc = nc_full if cross else 2 * gpu._nblk(nmax, window)
    pk = gpu.match_gpu(ctx, n_arr, nspec, nmax, base, active_nc, cross, mb).cpu().numpy()
    cont1, cont2 = pk[:, :nspec], pk[:, nspec:2 * nspec]
    meta = pk[:, 2 * nspec:]
    got = []
    for i in range(B):
        c1 = cont1[i, :int(meta[i, 0])]
        c2 = cont2[i, :int(meta[i, 2])]
        got.append(((int(meta[i, 1]), [int(x) for x in c1], int(meta[i, 0])),
                    (int(meta[i, 3]), [int(x) for x in c2], int(meta[i, 2]))))
    return got, meta


def check(name, rows, lens, window=0, cross=False, nspec=NSPEC):
    got, meta = run_match(rows, lens, window=window, cross=cross, nspec=nspec)
    want = ref_match(rows, lens, nspec=nspec, window=window, cross=cross)
    ok = got == want
    detail = "PASS" if ok else "FAIL"
    print(f"  [{detail}] {name}")
    if not ok:
        for i in range(len(rows)):
            if got[i] != want[i]:
                print(f"         row {i}: got  {got[i]}")
                print(f"         row {i}: want {want[i]}")
    return ok


def main():
    print(f"torch {torch.__version__}  triton {getattr(__import__('triton'), '__version__', '?')}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    ok = True
    ok &= check("self repeat", [[1, 2, 3, 4, 5, 1, 2, 3, 4]], [9])
    ok &= check("no match", [[1, 2, 3, 4, 5, 6]], [6])
    ok &= check("top-2 same block (candidate 2 must be found)",
                [[1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4]], [14])
    ok &= check("window bounds recall (window=6)", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]], [13], window=6)
    ok &= check("full context finds the distant repeat", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]], [13], window=0)
    rows_cross = [[1, 2, 3, 4, 5, 1, 2, 3, 4], [7, 8, 9, 5, 1, 2, 3, 4]]
    ok &= check("cross-request (self-only)", rows_cross, [9, 8], cross=False)
    ok &= check("cross-request (all rows)", rows_cross, [9, 8], cross=True)
    ok &= check("batch of heterogeneous lengths", [[9, 9, 7, 8, 9, 9, 7, 8], [1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7]],
                [8, 8, 3])
    ok &= check("nl > 512 crosses the block boundary", [[(i % 23) + 1 for i in range(700)] + [1, 2, 3, 4, 1, 2, 3, 4]],
                [708])
    long_rep = list(range(1, 31)) + [99, 99] + list(range(1, 31))
    ok &= check("horizon > 24 (30-token repeat, _MAXL=32)", [long_rep], [len(long_rep)])
    # recency metadata: self match q=3 of n=9 -> rec1 = 9-1-3 = 5; candidate 2 absent -> sentinel
    _, meta = run_match([[1, 2, 3, 4, 5, 1, 2, 3, 4]], [9])
    rec_ok = int(meta[0, 5]) == 5 and int(meta[0, 6]) == (1 << 30)
    ok &= rec_ok
    print(f"  [{'PASS' if rec_ok else 'FAIL'}] recency metadata (rec1={int(meta[0, 5])}, rec2={int(meta[0, 6])})")
    print("GPU NGRAM SELFTEST: " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
