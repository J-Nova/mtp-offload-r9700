#!/usr/bin/env bash
# One-command ~5-minute test of the RADIANCE MTP + n-gram controller.
#
# What it covers in a single run:
#   * GPU kernel selftest (fast; SKIP_GPU=1 to skip)
#   * for each SPEC in SPECS (default "4 8"): boot ONE server, then A/B the POLICY knobs LIVE on it
#     via RADIANCE_DRAFT_KNOB_FILE (no relaunch between arms):
#        off    STRONG=0 RECENT=0   -- agree-only (old behaviour)
#        len    STRONG=8 RECENT=0   -- length gate only
#        lenrec STRONG=8 RECENT=16  -- new defaults (length + recency)
# Only SPEC needs a fresh boot; everything else is switched on the running server.
#
# Outputs in $OUT (default ./bench):
#   quick-all.gpu.log, all-s<SPEC>-<arm>.{json,log,ngram.log}, all-s<SPEC>.serve.log
# and a final combined-decode matrix. Because the entrypoint refreshes radiance_draft*.py from this
# checkout, it measures the working tree.
#
# Usage:
#   bash bench-quick-all.sh
#   SPECS="8" ARMS="off lenrec" bash bench-quick-all.sh     # faster, one depth
#   SKIP_GPU=1 GEN=256 REPS=2 bash bench-quick-all.sh
set -euo pipefail
cd "$(dirname "$0")"

MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ}
SPECS=${SPECS:-"4 8"}
ARMS=${ARMS:-"off len lenrec"}
GEN=${GEN:-192}
REPS=${REPS:-1}
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-3}
PORT=${PORT:-6564}
CARD=${CARD:-1}
RUNTIME=${RUNTIME:-docker}
OUT=${OUT:-$PWD/bench}
KNOB=$PWD/.radiance-knobs.json
mkdir -p "$OUT"
# clear stale per-arm outputs so a rerun cannot mix old files into the globs/logs
rm -f "$OUT"/all-s*-*.json "$OUT"/all-s*-*.log "$OUT"/all-s*-*.ngram.log "$OUT"/all-s*.serve.log 2>/dev/null || true

arm_knobs() {
  case "$1" in
    off)    echo '{"strong":0,"recent":0}' ;;
    len)    echo '{"strong":8,"recent":0}' ;;
    lenrec) echo '{"strong":8,"recent":16}' ;;
    *)      echo '{}' ;;
  esac
}

if [ "${SKIP_GPU:-0}" != "1" ]; then
  echo "=== GPU kernel selftest ==="
  docker run --rm --privileged --ipc=host --network=host --device /dev/kfd --device /dev/dri \
    --group-add 993 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 -v "$PWD":/patches:z \
    --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc 'cd /patches && python3 ngram_draft_gpu_selftest.py' \
    > "$OUT/quick-all.gpu.log" 2>&1 || true
  tail -3 "$OUT/quick-all.gpu.log"
fi

for SPEC in $SPECS; do
  NAME="bench-quick-all-$SPEC"
  L="all-s$SPEC"
  echo "=== [$L] launch: mtp SPEC=$SPEC (one boot; policy arms switch live) ==="
  # an interrupted previous run leaves a bench-quick-* container holding the port -> stop them first
  for c in $(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E '^bench-quick' || true); do
    docker stop "$c" >/dev/null 2>&1 || true
  done
  echo '{"strong":8,"recent":16}' > "$KNOB"
  if ! MODELS="$MODELS" RUNTIME="$RUNTIME" NAME="$NAME" PORT="$PORT" GPUS="$CARD" TP=1 DETACH=1 \
       SNAP="$MODELS/$MODEL" SERVED_NAMES="$MODEL" SPEC_METHOD=mtp SPEC="$SPEC" MAXLEN="$MAXLEN" MAXSEQS="$MAXSEQS" \
       RADIANCE_DRAFT_KNOB_FILE=/patches/.radiance-knobs.json \
       ./serve-mxfp4.sh > "$OUT/$L.serve.log" 2>&1; then
    echo "=== [$L] serve-mxfp4.sh failed; see $OUT/$L.serve.log ==="
    tail -25 "$OUT/$L.serve.log" || true
    continue
  fi

  up=0
  echo "    [$L] container up; waiting for /health on :$PORT (first boot loads weights + compiles, 1-3 min)"
  for i in $(seq 1 150); do
    if curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    if [ $((i % 5)) -eq 0 ]; then
      printf "    [%s] booting %3ds ... %s\n" "$L" "$((i * 2))" "$(docker logs --tail 1 "$NAME" 2>&1 | tr -d '\r' | cut -c1-110)"
    fi
    sleep 2
  done
  if [ "$up" != 1 ]; then
    echo "=== [$L] server did not come up (see $OUT/$L.serve.log) ==="
    docker logs "$NAME" 2>&1 | tail -20 || true
    docker stop "$NAME" >/dev/null 2>&1 || true
    continue
  fi
  echo "    [$L] healthy after ~$((i * 2))s"

  for ARM in $ARMS; do
    arm_knobs "$ARM" > "$KNOB"
    sleep 0.5
    echo "--- [$L/$ARM] knobs: $(cat "$KNOB")  (gen=$GEN reps=$REPS x 4 prompts)"
    BENCH_MODEL="$MODEL" python3 bench-quick.py --base "http://localhost:$PORT" --gen "$GEN" --reps "$REPS" \
      --out "$OUT/$L-$ARM.json" 2>&1 | tee "$OUT/$L-$ARM.log" || true
    docker logs "$NAME" 2>&1 | grep '\[radiance.draft\] stats' | tail -1 > "$OUT/$L-$ARM.ngram.log" 2>/dev/null || true
  done
  docker stop "$NAME" >/dev/null 2>&1 || true
done

python3 - "$OUT" "$SPECS" "$ARMS" <<'PY'
import json, os, sys
out, specs, arms = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
print("=== quick-all matrix (COMBINED decode tok/s) ===")
for s in specs:
    cells = []
    for a in arms:
        p = os.path.join(out, f"all-s{s}-{a}.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                cells.append(f"{a}={d['combined_decode_tps']:.1f}")
            except Exception:
                cells.append(f"{a}=?")
    print(f"  SPEC={s}:  " + "   ".join(cells))
print("controller logs (ngram share):", " ".join(
    sorted(os.path.basename(p) for p in __import__('glob').glob(os.path.join(out, 'all-s*-*.ngram.log')))))
PY
