#!/bin/bash
# Build a DFlash2 drafter for one served target, end to end.
#
# A DFlash2 drafter's `fc` reads the TARGET's hidden states, so a drafter only speculates well on
# the target it was trained against. The shipped tcclaviger/Qwen3.8-27B-DFlash2-FP8 is matched to
# Qwen3.8-27B; serving it against a different/requantized target (a blend, an OCP-GPTQ conversion)
# collapses acceptance. This script fixes that for ONE target:
#
#   1. capture  serve the target with serve-mxfp4.sh + CAPTURE_DIR (prefix caching forced off) and
#               drive a prompt pool through it, recording the target's aux hidden states
#   2. train    fine-tune the existing DFlash2-FP8 drafter (dequantized in-process) on those
#               captures  (paroquant/drafter/train_drafter.py)
#   3. export   quantize the fine-tune back to the serving FP8 layout (export_fp8.py)
#
# Usage:
#   build_drafter_for_target.sh <TARGET_DIR_NAME> [OUT_DRAFTER_DIR_NAME]
#
#   e.g. build_drafter_for_target.sh Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ
#        build_drafter_for_target.sh Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-HYBRID
#
# Then serve it:
#   SNAP=$MODELS/<TARGET_DIR_NAME> DRAFTER=$MODELS/<OUT> ./serve-mxfp4.sh
# and verify acceptance with bench-async.py (pass the "drafter" field in a --target JSON).
#
# One capture+train per target: OCP-GPTQ and its HYBRID sibling have different hidden-state
# distributions, so each needs its own run. The prompt pool is built once and reused.
#
# Env (defaults in [ ]):
#   MODELS [~/models]        checkpoint dir (bind-mounted at /models)
#   DRAFTER_WORK [~/drafter_ft]  working dir: prompt pool, captures, fine-tunes
#   DRAFTER_BASE [$MODELS/Qwen3.8-27B-DFlash2-FP8]  drafter to fine-tune FROM (FP8 is fine)
#   IMG [stilldeadcode/vllm-radiance:0.9.3]
#   RUNTIME [auto]
#   PORT [8000]              capture-serve port
#   TP [1] GPUS [1] MAXLEN [32768] MAXSEQS [8]   capture-serve shape (TP=1 on one card fits a 32 GiB box)
#   CONC [8] GEN_TOKENS [768]        generation concurrency / max_tokens
#   SPEC [3]                 capture-time dflash draft depth; capture only needs propose() to run,
#                            and the mismatched drafter accepts <1 tok/draft, so a full 7 is wasted
#                            verify work. Raise toward 7 only if you want realism over speed.
#   EPOCHS [2] LR [5e-5] SEQS [6] ANCHORS [64]   training knobs
#   TRAIN_GPU [1]            HIP index for the training container
#   SKIP_PROMPTS SKIP_CAPTURE SKIP_GENERATE SKIP_TRAIN SKIP_EXPORT [0]   resume after a failure
#   REUSE_CAPTURE [0]        1 = keep an existing capture dir instead of re-capturing
#   VERBOSE [0]              deprecated/ignored -- container output is always streamed in full
#
# Stop production first: it needs the GPUs this captures on.
set -euo pipefail

usage() { sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 1 ] || { usage; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TARGET=$1
OUT=${2:-$TARGET-DFlash2-FP8}

MODELS=${MODELS:-$HOME/models}
DRAFTER_WORK=${DRAFTER_WORK:-$HOME/drafter_ft}
DRAFTER_BASE=${DRAFTER_BASE:-$MODELS/Qwen3.8-27B-DFlash2-FP8}
IMG=${IMG:-stilldeadcode/vllm-radiance:0.9.3}
PORT=${PORT:-8000}
TP=${TP:-1}
GPUS=${GPUS:-1}
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-8}   # capture concurrency; the single-GPU profile would otherwise cap it at 3
CONC=${CONC:-8}
GEN_TOKENS=${GEN_TOKENS:-768}
SPEC=${SPEC:-3}   # capture-time dflash draft depth: capture only needs propose() to run, and the
                  # mismatched drafter accepts <1 tok/draft, so a full 7 is wasted verify work.
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-5}
SEQS=${SEQS:-6}
ANCHORS=${ANCHORS:-64}
TRAIN_GPU=${TRAIN_GPU:-1}
SKIP_PROMPTS=${SKIP_PROMPTS:-0}; SKIP_CAPTURE=${SKIP_CAPTURE:-0}; SKIP_GENERATE=${SKIP_GENERATE:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}; SKIP_EXPORT=${SKIP_EXPORT:-0}; REUSE_CAPTURE=${REUSE_CAPTURE:-0}
VERBOSE=${VERBOSE:-0}   # 1 = stream ALL container output; 0 = key milestones only

MODELS="$(realpath -m "$MODELS")"
DRAFTER_WORK="$(realpath -m "$DRAFTER_WORK")"
SNAP="$MODELS/$TARGET"
POOL="$DRAFTER_WORK/prompts.jsonl"
CAP="$DRAFTER_WORK/cap-$TARGET"
RESP="$DRAFTER_WORK/resp-$TARGET.jsonl"
FT="$DRAFTER_WORK/ft-$TARGET"
OUTDIR="$MODELS/$OUT"
CONTAINER="radiance-drafter-cap"

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else echo "no podman/docker" >&2; exit 1; fi
fi

# `--group-add keep-groups` is podman-only; docker needs the numeric render/video GIDs for
# /dev/kfd + /dev/dri access. Same split serve-mxfp4.sh makes (see its runtime block).
GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    if [ -n "$gid" ]; then GROUP_FLAGS+=(--group-add "$gid"); fi
  done
fi

say()  { echo "$(date +%H:%M:%S) $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

[ -f "$SNAP/config.json" ] || die "no checkpoint at $SNAP"
[ -f "$DRAFTER_BASE/config.json" ] || die "no source drafter at $DRAFTER_BASE (set DRAFTER_BASE)"
have python3 || die "python3 not on PATH (needed to drive generation)"
have curl || die "curl not on PATH (needed to poll /health)"
mkdir -p "$DRAFTER_WORK"

cleanup_capture() {
  # Remove the container first: that ends `logs -f` (EOF), so the tailer exits on its own.
  if [ "$RUNTIME" = podman ] || [ "$RUNTIME" = docker ]; then
    $RUNTIME rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
  if [ -n "${TAIL_PID:-}" ]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
    wait "$TAIL_PID" 2>/dev/null || true
    TAIL_PID=""
  fi
}
TAIL_PID=""
trap 'cleanup_capture' EXIT

# Stream a container's log to this terminal, prefixed. Full output by default -- startup is
# weight load + torch.compile and the failure mode of interest (a crash) is a traceback, so a
# milestone filter would hide exactly the lines worth seeing. VERBOSE is accepted and ignored
# for backwards compatibility.
stream_logs() {
  local c=$1
  "$RUNTIME" logs -f --tail all "$c" 2>&1 | sed -u 's/^/[capture] /' &
  TAIL_PID=$!
}

say "target=$TARGET  out=$OUT  work=$DRAFTER_WORK  runtime=$RUNTIME  img=$IMG"

# ---------------------------------------------------------------- 0. prompt pool (built once)
if [ "$SKIP_PROMPTS" != 1 ] && [ ! -s "$POOL" ]; then
  say "0/4 building prompt pool -> $POOL (image has 'datasets'; needs network/HF cache)"
  $RUNTIME run --rm -v "$DRAFTER_WORK":/data:z -v "$SCRIPT_DIR":/scripts:z \
    -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    --entrypoint python3 "$IMG" /scripts/build_prompts.py /data/prompts.jsonl \
    || die "prompt-pool build failed (offline? set SKIP_PROMPTS=1 with an existing $POOL)"
else
  say "0/4 prompt pool present: $POOL"
fi

# ---------------------------------------------------------------- 1. capture serve + generate
do_capture=1
if [ "$SKIP_CAPTURE" = 1 ]; then do_capture=0; fi
if [ "$REUSE_CAPTURE" = 1 ] && [ -n "$(find "$CAP" -maxdepth 1 -name '*.pt' 2>/dev/null | head -1)" ]; then
  say "1/4 reusing captures in $CAP"; do_capture=0
fi
if [ "$do_capture" = 1 ]; then
  say "1/4 capture serve: TP=$TP GPU=$GPUS port=$PORT maxlen=$MAXLEN -> $CAP"
  mkdir -p "$CAP"
  CAPTURE_DIR="$CAP" SERVED_NAMES="$TARGET" DETACH=1 NAME="$CONTAINER" PORT="$PORT" \
    TP="$TP" GPUS="$GPUS" MAXLEN="$MAXLEN" MAXSEQS="$MAXSEQS" RUNTIME="$RUNTIME" SNAP="$SNAP" \
    SPEC="$SPEC" \
    SPEC_METHOD=dflash "$REPO_DIR/serve-mxfp4.sh"
  sleep 2
  stream_logs "$CONTAINER"          # live startup progress in THIS terminal
  t0=$(date +%s)
  last_hb=$(( $(date +%s) - t0 ))
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    el=$(( $(date +%s) - t0 ))
    printf '\r[capture] waiting for /health ... %ss ' "$el"
    sleep 5
    if ! $RUNTIME ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
      printf '\n'
      echo "[capture] container '$CONTAINER' is gone -- last log lines:" >&2
      "$RUNTIME" logs --tail 250 "$CONTAINER" >&2 2>&1 || true
      die "capture container exited during startup (see the lines above; if it vanished while a bench-* run was starting, bench-async.py's cleanup_stale removed it)"
    fi
    # Heartbeat: vLLM emits nothing for minutes while it loads weights and torch.compile runs, so
    # show that the process is alive (CPU%) and where it is (last log line) every 30s. A high CPU%
    # with a growing log = working; near-0% and a frozen last line = stuck.
    if [ $(( el - last_hb )) -ge 30 ]; then
      last_hb=$el
      hb_stats=$("$RUNTIME" stats --no-stream --format '{{.CPUPerc}} cpu / {{.MemUsage}} mem' "$CONTAINER" 2>/dev/null | tail -1)
      hb_last=$("$RUNTIME" logs --tail 1 "$CONTAINER" 2>/dev/null | tail -1 | cut -c1-140)
      printf '\n[capture] %ss alive | %s | last log: %s\n' "$el" "${hb_stats:-no stats}" "${hb_last:-<none yet -- loading/compiling>}"
    fi
  done
  printf '\r[capture] healthy after %ss%s\n' "$(( $(date +%s) - t0 ))" "                    "
  if [ "$SKIP_GENERATE" != 1 ]; then
    say "      driving prompts (conc $CONC, max_tokens $GEN_TOKENS) -- progress below"
    : > "$RESP"
    BENCH_URL="http://localhost:$PORT/v1/chat/completions" BENCH_MODEL="$TARGET" \
      python3 "$SCRIPT_DIR/generate.py" "$POOL" "$RESP" "$CONC" "$GEN_TOKENS"
    say "      waiting 40s for buffered captures to flush, then stopping the capture serve"
    sleep 40
  fi
  cleanup_capture                    # stops container + tailer
else
  say "1/4 capture skipped"
fi

ncap=$(find "$CAP" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
[ "$ncap" -gt 0 ] || die "no capture files in $CAP -- did the serve come up and generate?"
say "      captures: $ncap files, $(du -sh "$CAP" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------- 2. train
if [ "$SKIP_TRAIN" != 1 ]; then
  say "2/4 training from $DRAFTER_BASE -> $FT (epochs $EPOCHS, lr $LR, GPU $TRAIN_GPU)"
  rm -rf "$FT"
  mkdir -p "$FT"
  $RUNTIME run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
    "${GROUP_FLAGS[@]}" --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="$TRAIN_GPU" \
    -v "$DRAFTER_WORK":/data:z -v "$MODELS":/models:z -v "$SCRIPT_DIR":/scripts:z \
    --entrypoint bash "$IMG" -lc \
    "cd /data && python3 /scripts/train_drafter.py \
       --capture /data/cap-$TARGET --drafter /models/$(basename "$DRAFTER_BASE") \
       --target /models/$TARGET --out /data/ft-$TARGET \
       --epochs $EPOCHS --lr $LR --seqs $SEQS --anchors $ANCHORS \
       --eval-every 150 --save-every 300" \
    || die "training failed (see output above)"
  [ -f "$FT/model.safetensors" ] || die "training produced no $FT/model.safetensors"
else
  say "2/4 training skipped"
fi

# ---------------------------------------------------------------- 3. export FP8
if [ "$SKIP_EXPORT" != 1 ]; then
  say "3/4 exporting $FT -> $OUTDIR"
  rm -rf "$OUTDIR"
  $RUNTIME run --rm -v "$DRAFTER_WORK":/data:z -v "$MODELS":/models:z -v "$SCRIPT_DIR":/scripts:z \
    --entrypoint python3 "$IMG" \
    /scripts/export_fp8.py "/data/ft-$TARGET" "/models/$(basename "$DRAFTER_BASE")" "/models/$OUT" \
    || die "export failed"
else
  say "3/4 export skipped"
fi
[ -f "$OUTDIR/config.json" ] || die "no drafter at $OUTDIR"

# ---------------------------------------------------------------- done
cat <<EOF

$(date +%H:%M:%S) done.
  drafter : $OUTDIR
  captures: $CAP ($ncap files)

Serve the target with its own drafter:
  SNAP=$SNAP DRAFTER=$OUTDIR $REPO_DIR/serve-mxfp4.sh

Verify acceptance (compare spec_acceptance against std-mxfp4 in bench/overview.json):
  cd $REPO_DIR && ./bench-async.py --launch --phases cold,hit,decode,pressure --label dflash-ft \\
    --save-dir "\$PWD/bench" --models "$MODELS" \\
    --target '{"name":"$TARGET-ft","model":"$TARGET","snap":"/models/$TARGET","drafter":"/models/$OUT","spec_method":"dflash","maxlen":"32768"}'
EOF
