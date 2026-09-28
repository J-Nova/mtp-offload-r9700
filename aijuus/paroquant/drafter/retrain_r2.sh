#!/usr/bin/env bash
# Round-2 drafter retrain: build richer pools, capture with the round-1 drafter (SPEC=7),
# warm-start training on the union of round-1 + round-2 captures, export to FP8.
# Resumable: rerun as-is; generate.py skips responses already recorded in resp-*.jsonl.
set -euo pipefail
cd /home/juup/radiance-vllm-mxfp4

# ---------------- knobs ----------------
MODELS=${MODELS:-$HOME/models}
WORK=${WORK:-$HOME/drafter_ft}
SCRIPT_DIR=$PWD/paroquant/drafter
IMG=stilldeadcode/vllm-radiance:0.9.3
TARGET=Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ
BASE_DRAFTER=$MODELS/Qwen3.8-27B-DFlash2-FP8       # export/quant layout reference
R1_DRAFTER=$MODELS/$TARGET-DFlash2-FP8             # round-1 export: capture + warm start
OUT=$MODELS/$TARGET-DFlash2-FP8-r2
PORT=${PORT:-3454}; GPU=${GPU:-1}; MAXLEN=32768; MAXSEQS=8; CONC=${CONC:-8}
SPEC=7                                             # matches R1_DRAFTER block_size=8
GEN_TOKENS=${GEN_TOKENS:-1536}
CAP_MAX=8192
SAMPLE=${SAMPLE:-2500}                             # per pool; up to 7500 prompts
FORCE_POOLS=${FORCE_POOLS:-0}                      # 1 = rebuild pools even if prompts.jsonl exists
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
R2=$WORK/r2
mkdir -p "$R2" "$HF_CACHE"

say() { echo "$(date +%H:%M:%S) $*"; }

# ---------------- 0. preflight ----------------
[ -f "$R1_DRAFTER/config.json" ] || { echo "missing round-1 drafter at $R1_DRAFTER"; exit 1; }
if rocm-smi --showpids 2>/dev/null | grep -qE '^[0-9]+'; then
  say "WARN: GPU processes are running; stop them for a clean capture"
fi
docker rm -f radiance-drafter-cap 2>/dev/null || true
# stop the capture serve if the script is interrupted, so it never lingers on the GPU/port
trap 'docker stop radiance-drafter-cap >/dev/null 2>&1 || true' INT TERM

# ---------------- 1. build the three richer pools (network + datasets) ----------
pydocker() { docker run --rm \
  -v "$R2":/data:z -v "$SCRIPT_DIR":/scripts:z \
  -v "$HF_CACHE":/root/.cache/huggingface \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  --entrypoint python3 "$IMG" "$@"; }

# Building/merging is skipped when prompts.jsonl already exists, so a resume keeps the EXACT same
# prompt ids. Rebuilding can reshuffle ids (streaming builders yield different samples), which would
# make generate.py's done-set skip the wrong prompts and duplicate others.
if [ -s "$R2/prompts.jsonl" ] && [ "$FORCE_POOLS" != 1 ]; then
  say "1-2/5 reusing existing pool $R2/prompts.jsonl ($(wc -l < "$R2/prompts.jsonl") prompts; FORCE_POOLS=1 to rebuild)"
else

say "1/5 building pools (best-effort; missing datasets shrink the mix)"
for b in build_prompts2.py build_prompts3.py build_pool.py; do
  out="/data/${b#build_}"                 # build_prompts2.py -> prompts2.py  (renamed below)
  case "$b" in
    build_prompts2.py) out=/data/prompts2.jsonl ;;
    build_prompts3.py) out=/data/prompts3.jsonl ;;
    build_pool.py)     out=/data/pool.jsonl     ;;
  esac
  say "  -> $b"
  pydocker "/scripts/$b" "$out" || say "  (builder $b failed; continuing)"
done

# ---------------- 2. sample + merge (multi-turn/tool `messages` preserved) ------
say "2/5 sampling ${SAMPLE}/pool and merging"
python3 - "$R2" "$SAMPLE" <<'PY'
import json, os, random, sys
r2, n = sys.argv[1], int(sys.argv[2]); random.seed(0); rows = []
for f in ("prompts2.jsonl", "prompts3.jsonl", "pool.jsonl"):
    p = os.path.join(r2, f)
    if not os.path.exists(p):
        print("missing", p, "(skipping)"); continue
    rs = [json.loads(l) for l in open(p)]
    random.shuffle(rs)
    for r in rs[:n]:
        r["id"] = len(rows); rows.append(r)
    print(f, "->", min(n, len(rs)), "of", len(rs))
random.shuffle(rows)
with open(os.path.join(r2, "prompts.jsonl"), "w") as fh:
    fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
print("merged pool:", len(rows))
PY

fi

# ---------------- 3. capture: serve target WITH the round-1 drafter, SPEC=7 -----
CAP=$R2/cap-$TARGET; RESP=$R2/resp-$TARGET.jsonl; mkdir -p "$CAP"
say "3/5 capture serve on GPU $GPU port $PORT -> $CAP"
CAPTURE_DIR="$CAP" SERVED_NAMES="$TARGET" DETACH=1 NAME=radiance-drafter-cap \
  PORT=$PORT GPUS=$GPU MAXLEN=$MAXLEN MAXSEQS=$MAXSEQS SPEC=$SPEC SPEC_METHOD=dflash \
  RADIANCE_DFLASH_CAPTURE_MAX_TOKENS=$CAP_MAX SNAP="$MODELS/$TARGET" DRAFTER="$R1_DRAFTER" \
  ./serve-mxfp4.sh

t0=$(date +%s)
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  printf '\r[capture] waiting for /health ... %ss ' "$(( $(date +%s) - t0 ))"
  sleep 5
  docker ps --format '{{.Names}}' | grep -qx radiance-drafter-cap \
    || { echo; echo "capture container died -- docker logs radiance-drafter-cap"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo; echo "health timeout"; exit 1; }
done
printf '\r[capture] healthy after %ss\n' "$(( $(date +%s) - t0 ))"

# A kill can leave a truncated last line in resp-*.jsonl, and generate.py parses every line to
# build its done-set -- one bad line aborts the whole resume. Drop only a partial final line.
if [ -s "$RESP" ]; then python3 - "$RESP" <<'PY'
import json, sys
p = sys.argv[1]; lines = open(p, encoding="utf-8").read().splitlines()
if lines:
    try:
        json.loads(lines[-1])
    except Exception:
        with open(p, "w") as f:
            f.write("\n".join(lines[:-1]) + "\n")
        print("trimmed partial last response line")
PY
fi

BENCH_URL="http://localhost:$PORT/v1/chat/completions" BENCH_MODEL="$TARGET" \
  python3 "$SCRIPT_DIR/generate.py" "$R2/prompts.jsonl" "$RESP" "$CONC" "$GEN_TOKENS"
say "waiting 40s for buffered captures to flush"
sleep 40
docker stop radiance-drafter-cap >/dev/null 2>&1 || true
docker rm -f radiance-drafter-cap >/dev/null 2>&1 || true
say "captures: $(find "$CAP" -maxdepth 1 -name '*.pt' | wc -l) files, $(du -sh "$CAP" 2>/dev/null | cut -f1)"

# ---------------- 4. train: warm start from round-1, UNION of captures ---------
say "4/5 training (warm start from round-1, union of captures)"
docker run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU \
  -v "$WORK":/data:z -v "$MODELS":/models:z -v "$SCRIPT_DIR":/scripts:z \
  --entrypoint bash "$IMG" -lc \
  "cd /data && python3 /scripts/train_drafter.py \
     --capture /data/cap-$TARGET,/data/r2/cap-$TARGET \
     --drafter /models/$TARGET-DFlash2-FP8 \
     --target  /models/$TARGET --out /data/ft2 \
     --epochs 2 --lr 5e-5 --seqs 8 --anchors 96 --gamma 3 --max-len $CAP_MAX \
     --eval-every 150 --save-every 300"

# ---------------- 5. export to the FP8 serving layout --------------------------
say "5/5 exporting -> $OUT"
docker run --rm -v "$WORK":/data:z -v "$MODELS":/models:z -v "$SCRIPT_DIR":/scripts:z \
  --entrypoint python3 "$IMG" /scripts/export_fp8.py /data/ft2 "$BASE_DRAFTER" "$OUT"

say "done -> $OUT"
echo "serve: PORT=8081 SNAP=$MODELS/$TARGET DRAFTER=$OUT ./serve-mxfp4.sh"
