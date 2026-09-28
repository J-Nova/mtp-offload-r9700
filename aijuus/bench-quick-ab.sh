#!/usr/bin/env bash
# Fast A/B: run bench-quick.sh for a baseline and a candidate, then diff the two quick-*.json.
#
# Default comparison is the PENDING-HW depth question -- mtp SPEC=4 vs SPEC=8 -- both with the n-gram
# controller on. Override any cell with env: A_SPEC/B_SPEC, A_STRONG/B_STRONG, A_RECENT/B_RECENT,
# SPEC_METHOD, GEN, REPS, etc.
#
#   bash bench-quick-ab.sh
#   A_SPEC=8 B_SPEC=8 A_STRONG=0 B_STRONG=8 bash bench-quick-ab.sh      # isolate the length gate
#   A_RECENT=0 B_RECENT=16 A_SPEC=8 B_SPEC=8 bash bench-quick-ab.sh     # isolate the recency gate
set -euo pipefail
cd "$(dirname "$0")"

A_LABEL=${A_LABEL:-base}
B_LABEL=${B_LABEL:-cand}
A_SPEC=${A_SPEC:-4}
B_SPEC=${B_SPEC:-8}
A_STRONG=${A_STRONG:-8}
B_STRONG=${B_STRONG:-8}
A_RECENT=${A_RECENT:-0}
B_RECENT=${B_RECENT:-0}

export GEN=${GEN:-256} REPS=${REPS:-2} PORT=${PORT:-6564} CARD=${CARD:-1}
export MODEL=${MODEL:-} SPEC_METHOD=${SPEC_METHOD:-mtp} NGRAM_MAXL=${NGRAM_MAXL:-32}
export WINDOW=${WINDOW:-auto} CROSS_REQ=${CROSS_REQ:-auto} TAU=${TAU:-0.20}

LABEL="$A_LABEL" SPEC="$A_SPEC" STRONG="$A_STRONG" RECENT="$A_RECENT" bash bench-quick.sh
LABEL="$B_LABEL" SPEC="$B_SPEC" STRONG="$B_STRONG" RECENT="$B_RECENT" bash bench-quick.sh

OUT=${OUT:-$PWD/bench}
python3 - "$OUT/quick-$A_LABEL.json" "$OUT/quick-$B_LABEL.json" "$A_LABEL" "$B_LABEL" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print(f"--- A/B: {sys.argv[3]} -> {sys.argv[4]} ---")
print(f"  COMBINED {a['combined_decode_tps']:.1f} -> {b['combined_decode_tps']:.1f} tok/s "
      f"({100*(b['combined_decode_tps']/max(a['combined_decode_tps'],1e-9)-1):+.1f}%)")
for c in sorted(set(a["categories"]) | set(b["categories"])):
    ca = a["categories"].get(c); cb = b["categories"].get(c)
    if not ca or not cb:
        continue
    print(f"  {c:>6} decode {ca['decode_tps']:7.1f} -> {cb['decode_tps']:7.1f} tok/s | "
          f"acc/draft {ca['acc_per_draft']:.3f} -> {cb['acc_per_draft']:.3f} | "
          f"ms/step {ca['ms_step']:6.2f} -> {cb['ms_step']:6.2f} | "
          f"dup8 {ca['dup8']*100:4.1f}% -> {cb['dup8']*100:4.1f}%")
PY
