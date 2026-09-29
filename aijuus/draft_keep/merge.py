#!/usr/bin/env python3
"""Merge per-instance token-collector outputs into the W4 draft_keep_file.

Reads every rank*.json the collector wrote (one per vLLM instance), unions the token
ids, and writes a single keep.json -- the default RADIANCE_DRAFT_KEEP_FILE that
radiance_kernels._install_draft_w4 points the reduced-vocab W4 draft head at.

Usage:
  python3 aijuus/draft_keep/merge.py                 # aijuus/draft_keep/rank*.json -> keep.json
  python3 aijuus/draft_keep/merge.py --vocab 124160  # also print vocab coverage
  python3 aijuus/draft_keep/merge.py DIR... --out PATH
"""
import argparse
import glob
import json
import os
import sys

try:  # run as a script from this dir: sys.path[0] is the script dir
    from idset import read_ids
except ImportError:  # run as a module from the repo root
    from aijuus.draft_keep.idset import read_ids


def load_ids(paths):
    ids = set()
    for p in paths:
        try:
            data = read_ids(p)
        except FileNotFoundError:
            print(f"  skip (missing): {p}")
            continue
        except Exception as e:
            print(f"  skip (bad read {e!r}): {p}")
            continue
        n0 = len(ids)
        for v in data:
            if isinstance(v, bool):
                continue
            try:
                ids.add(int(v))
            except Exception:
                pass
        print(f"  {p}: {len(data)} ids (+{len(ids) - n0} new)")
    return ids


def save_atomic(path, ids):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(sorted(ids), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", default=None,
                    help="dirs to glob rank*.json in (default: aijuus/draft_keep)")
    ap.add_argument("--out", default="aijuus/draft_keep/keep.json",
                    help="output keep file (default: aijuus/draft_keep/keep.json)")
    ap.add_argument("--vocab", type=int, default=None,
                    help="optional tokenizer vocab size, for a coverage report")
    a = ap.parse_args()

    paths = []
    for d in (a.dirs or ["aijuus/draft_keep"]):
        paths += sorted(glob.glob(os.path.join(d, "rank*.json")))
    if not paths:
        print("no rank*.json found; nothing to merge")
        return 1

    print(f"merging {len(paths)} file(s):")
    ids = load_ids(paths)
    if not ids:
        print("no ids collected; refusing to write an empty keep file")
        return 1

    save_atomic(a.out, ids)
    print(f"wrote {a.out}: {len(ids)} unique ids (min={min(ids)} max={max(ids)})")
    if a.vocab:
        print(f"coverage: {len(ids)}/{a.vocab} = {100.0 * len(ids) / a.vocab:.1f}% of vocab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
