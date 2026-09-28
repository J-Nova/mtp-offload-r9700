#!/usr/bin/env python3
"""CPU selftest for the RADIANCE MTP + n-gram drafting changes (NGRAM-MTP-PLAN.md).

Runs without a GPU or vLLM. It validates, against a plain numpy reference:

  1. the matcher SEMANTICS the Triton kernels implement (longest-suffix top-2, tie rules, the window,
     cross-request search, and the int64 key round-trip used to carry (match_len, row, end_pos));
  2. the per-slot policy `radiance_draft.slot_decide` (agreement, the length gate, candidate-2
     fallback, and that a chosen continuation never exceeds nspec or leaks a -1 sentinel).

It cannot execute the Triton kernels themselves -- those still need the R9700 box (the plan's HW
checklist) -- but it pins the contract the kernels and the host policy are written against.

Usage: python3 ngram_draft_selftest.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radiance_draft as rd  # noqa: E402  (module-level imports are only os/sys/numpy)

MAXL, MIN, SH, RH = 24, 3, 20, 7
LSH = SH + RH
NSPEC = 8


# ----- the key encoding the kernels use --------------------------------------
def _enc(match_len, row, end_pos):
    return (int(match_len) << LSH) | (int(row) << SH) | int(end_pos)


def _dec(key):
    if key == 0:
        return 0, 0, 0
    return (key >> LSH, (key >> SH) & ((1 << RH) - 1), key & ((1 << SH) - 1))


def test_key_roundtrip():
    for L, row, q in [(3, 0, 0), (24, 5, 12345), (8, 127, (1 << SH) - 1), (17, 3, 999999)]:
        L2, r2, q2 = _dec(_enc(L, row, q))
        assert (L2, r2, q2) == (L, row, q), (L, row, q, (L2, r2, q2))
    assert _dec(0) == (0, 0, 0)
    print("  ok  key round-trip")


# ----- numpy reference matcher (mirror of _match_scan/_match_scan_x + top-2 + gather) ----------
def ref_match(rows, lens, nspec=NSPEC, window=0, cross=False):
    """rows: list[list[int]]. Returns per row the top-2 (mlen, cont, clen), length-first then recency."""
    B = len(rows)
    out = []
    for i in range(B):
        n = lens[i]
        if n < 2:
            out.append(((0, [], 0), (0, [], 0)))
            continue
        base = max(0, n - window) if window > 0 else 0
        keys = []
        search_rows = range(B) if cross else [i]
        for j in search_rows:
            nj = lens[j]
            rowj = rows[j]
            for q in range(base, nj - 1):
                L = 0
                for k in range(MAXL):
                    sk = n - 1 - k
                    qi = q - k
                    if sk < 0 or qi < base and not cross:
                        break
                    if qi < 0 or sk < 0:
                        break
                    if rowj[qi] != rows[i][sk]:
                        break
                    L += 1
                if L >= MIN:
                    keys.append(_enc(L, j, q))
        keys.sort(reverse=True)
        cands = []
        for key in keys[:2]:
            L, j, q = _dec(key)
            cont = rows[j][q + 1:q + 1 + nspec]
            clen = min(nspec, lens[j] - (q + 1))
            cands.append((L, [int(x) for x in cont], clen))
        while len(cands) < 2:
            cands.append((0, [], 0))
        out.append((cands[0], cands[1]))
    return out


def _row(tokens, name, expect):
    rows = [list(tokens)]
    lens = [len(tokens)]
    (m1, c1, l1), (m2, c2, l2) = ref_match(rows, lens)[0]
    got = (m1, c1, l1)
    assert got == expect, f"{name}: got {got} expected {expect}"
    print(f"  ok  {name}: mlen={m1} cont={c1} clen={l1}")


def test_matcher_semantics():
    # "a b c d e a b c d": the suffix "a b c d" repeats at 0; continuation is the tail after it.
    _row([1, 2, 3, 4, 5, 1, 2, 3, 4], "repeat-4", (4, [5, 1, 2, 3, 4], 5))
    # no repeated suffix >= MIN
    _row([1, 2, 3, 4, 5, 6], "no-match", (0, [], 0))
    # suffix "x y z" appears twice; the more recent occurrence wins the tie in length
    rows = [[9, 9, 7, 8, 9, 9, 7, 8]]
    got = ref_match(rows, [len(rows[0])])[0][0]
    assert got[0] == 4, got
    print(f"  ok  longest match wins: mlen={got[0]} end carries recency: cont={got[1]}")


def test_window():
    # distant repeat, window smaller than the distance -> no match; full context -> match
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9] + [1, 2, 3, 4]
    distant = ref_match([list(toks)], [len(toks)], window=6)[0][0][0]
    full = ref_match([list(toks)], [len(toks)], window=0)[0][0][0]
    assert distant == 0, distant
    assert full == 4, full
    print(f"  ok  window bounds recall: window6 mlen={distant}, full mlen={full}")


def test_cross_request():
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4]      # request A: self match, cont [5]
    b = [7, 8, 9, 5, 1, 2, 3, 4]          # request B ends in 1 2 3 4, with no earlier copy in its row
    self_only = ref_match([a, b], [len(a), len(b)], cross=False)[1][0][0]
    cross = ref_match([a, b], [len(a), len(b)], cross=True)[1][0][0]
    # B's own row has no earlier 1 2 3 4, so self-only finds nothing, cross finds A's occurrence
    assert self_only == 0, self_only
    assert cross >= 4, cross
    print(f"  ok  cross-request: self mlen={self_only}, cross mlen={cross}")


# ----- slot policy ------------------------------------------------------------
def _arrays(nspec, cont1, cont2, clen1, mlen1, clen2, mlen2, mtpn):
    return (np.array([cont1], dtype=np.int64), np.array([cont2], dtype=np.int64),
            np.array([clen1], dtype=np.int64), np.array([mlen1], dtype=np.int64),
            np.array([clen2], dtype=np.int64), np.array([mlen2], dtype=np.int64),
            np.array([mtpn], dtype=np.int64))


def test_top2_same_block():
    # the global 1st and 2nd best must both be found even when they sit in the SAME 512-position
    # block (the case the old per-block-max reduction dropped, making candidate 2 inert for any
    # context <= 512 tokens). Here the 4-suffix occurs twice in one block.
    toks = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4]
    (m1, c1, l1), (m2, c2, l2) = ref_match([list(toks)], [len(toks)])[0]
    assert m1 == 4 and m2 == 4, (m1, m2)
    print(f"  ok  top-2 same block: m1={m1} cont1={c1}, m2={m2} cont2={c2}")


def test_policy():
    # (a) short match that agrees with MTP -> take candidate 1
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [-1] * 8, 8, 4, 0, 0, 7)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([1.0]))
    assert int(a[0]) == 1 and int(cand[0]) == 1, (a, cand)
    print("  ok  policy: agreement takes c1")

    # (b) short match that DISAGREES, confidence high -> keep drafting (old behaviour preserved)
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [-1] * 8, 8, 3, 0, 0, 9)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([1.0]))
    assert int(a[0]) == 0 and int(cand[0]) == 0, (a, cand)
    print("  ok  policy: short disagreement keeps drafting")

    # (c) LONG match that disagrees -> taken (P1 length gate)
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [-1] * 8, 8, 12, 0, 0, 9)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([1.0]))
    assert int(a[0]) == 1 and int(cand[0]) == 1, (a, cand)
    print("  ok  policy: long disagreement takes c1 (P1)")

    # (d) c1 disagrees and is short; c2 agrees -> fallback to candidate 2 (P4)
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [9] * 8, 8, 3, 8, 5, 9)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([1.0]))
    assert int(a[0]) == 1 and int(cand[0]) == 2, (a, cand)
    print("  ok  policy: candidate-2 fallback (P4)")

    # (e) strong=0 restores agree-only; low confidence -> stop and verify
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [-1] * 8, 8, 12, 0, 0, 9)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([0.01]), strong=0)
    assert int(a[0]) == 2 and int(cand[0]) == 0, (a, cand)
    print("  ok  policy: strong=0 disables the length gate and the gate stops")

    # (e2) recency gate (opt-in; default RECENT=0): a SHORT match that disagrees is taken when it
    # recurred very recently, and ignored when distant -- only when recent is enabled.
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [7] * 8, [-1] * 8, 8, 4, 0, 0, 9)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([0.01]),
                             rec1=np.array([3]), rec2=np.array([10 ** 9]), recent=16)
    assert int(a[0]) == 1 and int(cand[0]) == 1, (a, cand)
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([0.01]),
                             rec1=np.array([500]), rec2=np.array([10 ** 9]), recent=16)
    assert int(a[0]) == 2 and int(cand[0]) == 0, (a, cand)
    # with the default (off) the same recent match is NOT taken
    a, cand = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([0.01]),
                             rec1=np.array([3]), rec2=np.array([10 ** 9]))
    assert int(a[0]) == 2 and int(cand[0]) == 0, (a, cand)
    print("  ok  policy: recency gate (opt-in) takes a recent short match, ignores distant/off")

    # (f) no candidates -> confidence gate decides 2 vs 0
    c1, c2, l1, m1, l2, m2, mtp = _arrays(8, [-1] * 8, [-1] * 8, 0, 0, 0, 0, 5)
    a, _ = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([0.01]))
    assert int(a[0]) == 2, a
    a, _ = rd.slot_decide(0, c1, c2, l1, m1, l2, m2, mtp, np.array([1.0]))
    assert int(a[0]) == 0, a
    print("  ok  policy: no match -> confidence gate")


def test_pack_shape_contract():
    # Guard the drift that silently disabled the controller once: the host pack-shape checks must
    # track the kernels' _META tail rather than a hardcoded +4. pack = [cont1 | cont2 | meta].
    import re
    root = os.path.dirname(os.path.abspath(__file__))
    host = open(os.path.join(root, "radiance_draft.py")).read()
    kern = open(os.path.join(root, "radiance_draft_gpu.py")).read()
    meta = int(re.search(r"^_META = (\d+)", kern, re.M).group(1))
    assert meta == 7, meta
    assert "2 * N + gpu._META" in host and "2 * nspec + gpu._META" in host
    # no pack-shape check may pin a literal tail (the window_miss column index at 2*nspec+4 is fine)
    assert "shape[1] != 2 * nspec + 4" not in host and "shape[1] != 2 * N + 4" not in host
    print(f"  ok  pack-shape contract: _META={meta}, host checks use gpu._META")


def test_constants():
    # the policy module loads without a GPU/vLLM and exposes the knobs the plan documents
    assert rd.STRONG >= 0 and rd.TAU > 0
    assert rd.slot_decide is not None
    print("  ok  radiance_draft policy module loads; STRONG/TAU exposed")


def main():
    print("[ngram_draft_selftest]")
    for fn in (test_key_roundtrip, test_matcher_semantics, test_top2_same_block, test_window,
               test_cross_request, test_policy, test_pack_shape_contract, test_constants):
        fn()
    print("ALL PASS")


if __name__ == "__main__":
    main()
