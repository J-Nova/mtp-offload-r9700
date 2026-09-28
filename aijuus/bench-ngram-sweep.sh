#!/usr/bin/env bash
# RADIANCE n-gram threshold sweep (NGRAM-MTP-PLAN.md item 1).
#
# Runs the blend-OCP MTP target at a fixed speculative depth across a grid of the two policy knobs
# that decide how aggressively a verbatim match is taken, writing one bench-async overview per cell:
#
#   RADIANCE_DRAFT_NGRAM_STRONG  min match length taken without MTP agreement
#   RADIANCE_DRAFT_NGRAM_RECENT  take a short match (>=3) that recurred within N tokens
#
# Compare the cells' decode tok/s and per-category tok/update. The controller's own
# `[radiance.draft] stats ... tokens: mtp=.. ngram=.. (X% ngram, Y tok/draft)` line shows how much
# of each draft the n-gram path supplied, so a high n-gram share with unchanged throughput is the win.
#
# Usage:
#   bash bench-ngram-sweep.sh
#   STRONGS="4 6 8" RECENTS="0 16 32" SPEC=8 bash bench-ngram-sweep.sh
#   EXTRA_ARGS="" bash bench-ngram-sweep.sh          # full benchmark instead of --quick
set -euo pipefail
cd "$(dirname "$0")"

MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ}
SPEC=${SPEC:-8}
PORT=${PORT:-6564}
CARD=${CARD:-1}
RUNTIME=${RUNTIME:-docker}
STRONGS=${STRONGS:-"4 6 8 12"}
RECENTS=${RECENTS:-"0 16"}
EXTRA_ARGS=${EXTRA_ARGS:---quick}
OUT=${OUT:-$PWD/bench}

for s in $STRONGS; do
  for r in $RECENTS; do
    tag="s${s}-r${r}"
    echo "=== n-gram sweep cell: STRONG=$s RECENT=$r (SPEC=$SPEC) -> $OUT/ngram-$tag.overview.json ==="
    RADIANCE_DRAFT_NGRAM_STRONG=$s RADIANCE_DRAFT_NGRAM_RECENT=$r \
      python3 bench-async.py --launch --models "$MODELS" --runtime "$RUNTIME" --card "$CARD" \
        --label "ngram-$tag" --save-dir "$OUT" \
        --overview "$OUT/ngram-$tag.overview.json" --phases all --no-html $EXTRA_ARGS \
        --target "{\"name\":\"ngram-$tag\",\"gpu\":$CARD,\"port\":$PORT,\"model\":\"$MODEL\",\"snap\":\"/models/$MODEL\",\"spec_method\":\"mtp\",\"spec\":\"$SPEC\",\"maxlen\":\"32768\"}"
  done
done
echo "done: $OUT/ngram-s*-r*.overview.json"
