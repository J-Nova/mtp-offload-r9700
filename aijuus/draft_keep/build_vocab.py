#!/usr/bin/env python3
"""Build the draft-vocabulary keep file for RADIANCE_DRAFT_VOCAB.

Satisfies the R9700-port caveat (a): the branch's 49,152-id list is SEED only -- this unions it
with our own observations. Sources:
  --seed   the branch list (one token id per line)              [default aijuus/draft_keep/qwen38-draft-vocab-49152.txt]
  --rank-glob  our collector output (rank*.json, JSON int list) [default aijuus/draft_keep/rank*.json]
  --extra  any further id lists (txt one-per-line or JSON list), repeatable -- e.g. a list produced
           by tokenizing a diverse corpus with this model's tokenizer, which is the robust way to
           grow "our workload" without depending on the draft-pass hook.
Writes one id per line, sorted, unique (atomic replace).
"""
import argparse
import glob
import json
import os

try:  # run as a script from this dir: sys.path[0] is the script dir
    from idset import read_ids
except ImportError:  # run as a module from the repo root
    from aijuus.draft_keep.idset import read_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="aijuus/draft_keep/qwen38-draft-vocab-49152.txt")
    ap.add_argument("--rank-glob", default="aijuus/draft_keep/rank*.json")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--out", default="aijuus/draft_keep/keep-union.txt")
    ap.add_argument("--json-out", default="aijuus/draft_keep/keep-union.json",
                    help="JSON sibling the reduced-vocab W4 head reads (RADIANCE_DRAFT_KEEP_FILE); "
                         "set to '' to skip")
    a = ap.parse_args()

    ids = set()
    n_seed = 0
    if a.seed and os.path.exists(a.seed):
        s = set(read_ids(a.seed))
        ids |= s
        n_seed = len(s)
    n_rank = 0
    for f in sorted(glob.glob(a.rank_glob)):
        s = set(read_ids(f))
        ids |= s
        n_rank = max(n_rank, len(s))
    n_extra = 0
    for f in a.extra:
        s = set(read_ids(f))
        ids |= s
        n_extra += len(s)

    if not ids:
        raise SystemExit("no ids collected; nothing written")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(str(i) for i in sorted(ids)) + "\n")
    os.replace(tmp, a.out)
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        jtmp = a.json_out + ".tmp"
        with open(jtmp, "w") as fh:
            json.dump(sorted(ids), fh)
        os.replace(jtmp, a.json_out)
    print(f"wrote {a.out}: {len(ids)} ids "
          f"(seed {n_seed}, rank union {n_rank}, extra {n_extra})"
          + (f"; {a.json_out}" if a.json_out else ""))


if __name__ == "__main__":
    main()
