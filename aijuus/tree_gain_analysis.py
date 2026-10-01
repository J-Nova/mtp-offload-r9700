#!/usr/bin/env python3
"""TR0: offline tree-gain quantification from /tmp/rad_probe.jsonl (written by the NGRAM_PROBE run).

Each paired record holds the PREVIOUS step's MTP draft (`prev_m`), the suffix continuation (`prev_c`), and
the tokens actually committed that step (`committed` = accepted drafts + 1 bonus). We compute the accepted
length each single chain would have earned (common prefix) and the 2-branch tree oracle = max of the two.
If mean(tree) does not beat mean(A_mtp) meaningfully, the tree rebuild is unjustified.

  python3 aijuus/tree_gain_analysis.py [path]
"""
import json
import sys


def cpl(a, b):
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main(path="/tmp/rad_probe.jsonl"):
    try:
        rows = [json.loads(line) for line in open(path)]
    except FileNotFoundError:
        print(f"no probe file at {path}")
        return 1
    amtp, asuf, mlen = [], [], []
    for r in rows:
        if "committed" not in r:
            continue
        com = r["committed"]
        amtp.append(cpl(r.get("prev_m", []), com))
        asuf.append(cpl(r.get("prev_c", []), com))
        mlen.append(int(r.get("mlen", 0)))
    if not amtp:
        print("no paired rows in probe")
        return 0
    tree = [max(a, s) for a, s in zip(amtp, asuf)]
    gain = [t - a for t, a in zip(tree, amtp)]
    print(f"paired steps: {len(amtp)}")
    print(f"mean A_mtp    (MTP-only accepted): {mean(amtp):.3f}")
    print(f"mean A_suffix (suffix accepted):   {mean(asuf):.3f}")
    print(f"mean tree max (2-branch oracle):   {mean(tree):.3f}")
    print(f"mean tree gain over MTP: {mean(gain):.3f} tokens/step ({100 * mean(gain) / max(1e-9, mean(amtp)):.1f}%)")
    idx = [i for i, x in enumerate(mlen) if x >= 8]
    if idx:
        print(f"on mlen>=8 steps ({len(idx)}): A_mtp {mean([amtp[i] for i in idx]):.3f} "
              f"A_suffix {mean([asuf[i] for i in idx]):.3f} tree {mean([tree[i] for i in idx]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rad_probe.jsonl"))
