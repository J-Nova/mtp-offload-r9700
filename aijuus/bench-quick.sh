#!/usr/bin/env bash
# ~5-minute A/B launcher for the RADIANCE MTP + n-gram controller.
#
# Boots ONE serve-mxfp4.sh instance, waits for health, runs the short bench-quick.py decode probe
# against it, grabs the controller's own `[radiance.draft] stats` lines from the container log, then
# tears it down -- and writes bench/quick-<label>.json + .log + .ngram.log.
#
# It reuses the repo files (the entrypoint cp fix means radiance_draft.py / _gpu.py are refreshed from
# this checkout), so it measures the working tree, not the baked image.
#
# Usage:
#   LABEL=a SPEC=4 bash bench-quick.sh
#   LABEL=b SPEC=8 STRONG=6 RECENT=16 bash bench-quick.sh
#   bash bench-quick-ab.sh                      # runs a SPEC=4 baseline then a SPEC=8 candidate, diffs
#   KEEP=1 LABEL=debug SPEC=8 bash bench-quick.sh   # leave the server running to poke at manually
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${LABEL:-quick}
MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ}
SPEC_METHOD=${SPEC_METHOD:-mtp}
SPEC=${SPEC:-4}
MAXLEN=${MAXLEN:-32768}
PORT=${PORT:-6564}
CARD=${CARD:-1}
RUNTIME=${RUNTIME:-docker}
GEN=${GEN:-256}
REPS=${REPS:-2}
MAXSEQS=${MAXSEQS:-3}
# n-gram / policy knobs (forwarded by serve-mxfp4.sh)
STRONG=${STRONG:-8}
RECENT=${RECENT:-0}
NGRAM_MAXL=${NGRAM_MAXL:-32}
WINDOW=${WINDOW:-auto}
CROSS_REQ=${CROSS_REQ:-auto}
TAU=${TAU:-0.20}
KEEP=${KEEP:-0}
OUT=${OUT:-$PWD/bench}
NAME="bench-quick-$LABEL"
mkdir -p "$OUT"

echo "=== [quick $LABEL] launch: mtp SPEC=$SPEC STRONG=$STRONG RECENT=$RECENT WINDOW=$WINDOW CROSS=$CROSS_REQ (port $PORT) ==="
# stop any stale quick container holding the port (an interrupted prior run leaves one behind)
for c in $(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^bench-quick' || true); do
  docker stop "$c" >/dev/null 2>&1 || true
done
# Build the launch environment as an array so the optional knob file is passed as a real env
# assignment. A bare ${VAR:+NAME="$VAR"} word starts with '$', so bash treats it as a command (not an
# assignment prefix) and the launch fails with "No such file or directory".
LAUNCH_ENV=(
  MODELS="$MODELS" RUNTIME="$RUNTIME" NAME="$NAME" PORT="$PORT" GPUS="$CARD" TP=1 DETACH=1
  SNAP="$MODELS/$MODEL" SERVED_NAMES="$MODEL"
  SPEC_METHOD="$SPEC_METHOD" SPEC="$SPEC" MAXLEN="$MAXLEN" MAXSEQS="$MAXSEQS"
  RADIANCE_DRAFT_NGRAM_STRONG="$STRONG" RADIANCE_DRAFT_NGRAM_RECENT="$RECENT"
  RADIANCE_DRAFT_NGRAM_MAXL="$NGRAM_MAXL" RADIANCE_DRAFT_NGRAM_WINDOW="$WINDOW"
  RADIANCE_DRAFT_CROSS_REQ="$CROSS_REQ" RADIANCE_DRAFT_TAU="$TAU"
)
[ -n "${RADIANCE_DRAFT_KNOB_FILE:-}" ] && LAUNCH_ENV+=(RADIANCE_DRAFT_KNOB_FILE="$RADIANCE_DRAFT_KNOB_FILE")
if ! env "${LAUNCH_ENV[@]}" ./serve-mxfp4.sh > "$OUT/quick-$LABEL.serve.log" 2>&1; then
  echo "[quick $LABEL] serve-mxfp4.sh failed; see $OUT/quick-$LABEL.serve.log" >&2
  tail -25 "$OUT/quick-$LABEL.serve.log" >&2 || true
  exit 1
fi

echo "    [quick $LABEL] waiting for /health on :$PORT (first boot 1-3 min)"
for i in $(seq 1 150); do
  curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1 && break
  [ $((i % 5)) -eq 0 ] && printf "    [quick %s] booting %3ds ... %s\n" "$LABEL" "$((i * 2))" "$(docker logs --tail 1 "$NAME" 2>&1 | tr -d '\r' | cut -c1-110)"
  sleep 2
done
if ! curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1; then
  echo "[quick $LABEL] server did not become healthy; see $OUT/quick-$LABEL.serve.log" >&2
  docker logs "$NAME" 2>&1 | tail -30 >&2 || true
  docker stop "$NAME" >/dev/null 2>&1 || true
  exit 1
fi

BENCH_MODEL="$MODEL" python3 bench-quick.py --base "http://localhost:$PORT" --gen "$GEN" --reps "$REPS" \
  --out "$OUT/quick-$LABEL.json" 2>&1 | tee "$OUT/quick-$LABEL.log"

docker logs "$NAME" 2>&1 | grep '\[radiance.draft\] stats' | tail -4 > "$OUT/quick-$LABEL.ngram.log" || true
if [ -s "$OUT/quick-$LABEL.ngram.log" ]; then
  echo "--- [quick $LABEL] controller stats ---"
  cat "$OUT/quick-$LABEL.ngram.log"
fi
# the token split tells you how much the n-gram path actually contributed
python3 - "$OUT/quick-$LABEL.json" <<'PY' || true
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[quick] combined decode {d['combined_decode_tps']:.1f} tok/s")
PY

if [ "$KEEP" = "1" ]; then
  echo "[quick $LABEL] KEEP=1 -> server left running as $NAME on port $PORT"
else
  docker stop "$NAME" >/dev/null 2>&1 || true
  echo "[quick $LABEL] done -> $OUT/quick-$LABEL.json"
fi
