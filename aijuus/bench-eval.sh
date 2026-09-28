#!/usr/bin/env bash
# One command to launch a model (one at a time), run the intelligence + timing eval
# and write the combined overview. Targets are passed inline, exactly like
# bench-blend.sh / bench-async.py -- no target-list files:
#
#   MODELS=$HOME/models ./bench-eval.sh --quick \
#     --target '{"name":"blend","model":"Blend","snap":"/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ","spec_method":"dflash"}' \
#     --target '{"name":"std","model":"Standard","snap":"/models/Qwen3.8-27B-MXFP4-mtpfp8","spec_method":"dflash"}'
#
# Env knobs: LABEL, SAVE_DIR, TASKS, LAUNCH=0|1, SPEC_MAX, MODELS.
# Everything else is forwarded to bench-eval.py (--quick, --limit, --thinking,
# --concurrency, --tasks, ...). SPEC_MAX=N clamps each --target's speculative depth
# to at most N (dflash 7, mtp 8, none 0), like bench-blend.sh.
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${LABEL:-eval-ab}
SAVE_DIR=${SAVE_DIR:-$PWD/bench/eval}
TASKS=${TASKS:-}
MODELS=${MODELS:-}
LAUNCH=${LAUNCH:-1}
SPEC_MAX=${SPEC_MAX:-8}

if [ -z "$MODELS" ]; then
  prev=""
  for a in "$@"; do
    [ "$prev" = "--models" ] && MODELS=$a
    prev=$a
  done
fi
if [ "$LAUNCH" = 1 ] && [ -z "$MODELS" ]; then
  echo "[eval] MODELS is required with --launch, e.g. MODELS=\$HOME/models $0 --target '{...}'" >&2
  exit 2
fi

# --- Forwarded args, with any --target JSON spec-clamped if SPEC_MAX is set ---
mapfile -t ARGS < <(SPEC_MAX="$SPEC_MAX" python3 - "$@" <<'PY'
import json, os, sys
spec_max = os.environ.get("SPEC_MAX", "").strip()

def method_default(method):
    return {"dflash": 7, "mtp": 8, "none": 0}.get(method, 8)

args = sys.argv[1:]
out, i = [], 0
while i < len(args):
    a = args[i]
    if a == "--target" and i + 1 < len(args):
        try:
            t = json.loads(args[i + 1])
            if spec_max:
                cur = int(str(t.get("spec") or method_default(t.get("spec_method", "dflash"))))
                t["spec"] = str(min(cur, int(spec_max)))
            out += ["--target", json.dumps(t)]
        except Exception:
            out += [a, args[i + 1]]
        i += 2
        continue
    out.append(a)
    i += 1
for x in out:
    print(x)
PY
)

has_target=0
has_tasks=0
for a in "${ARGS[@]}"; do
  [ "$a" = "--target" ] && has_target=1
  [ "$a" = "--tasks" ] && has_tasks=1
done
[ "$has_target" = 1 ] || { echo "[eval] pass at least one inline --target '{...}'" >&2; exit 2; }

cmd=(python3 bench-eval.py --label "$LABEL" --save-dir "$SAVE_DIR")
[ "$has_tasks" = 1 ] || [ -z "$TASKS" ] || cmd+=(--tasks "$TASKS")
if [ "$LAUNCH" = 1 ]; then
  cmd+=(--launch --models "$MODELS")
fi
cmd+=("${ARGS[@]}")

echo "[eval] label=$LABEL launch=$LAUNCH tasks=${TASKS:-'(default)'} spec_max=${SPEC_MAX:-8} -> $SAVE_DIR"
"${cmd[@]}"

echo
echo "[eval] done. read $SAVE_DIR/overview.json"
