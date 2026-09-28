#!/usr/bin/env bash
# One command to launch a model, run BetterBench and write the combined overview.
# Targets are passed inline, exactly like the old bench-async.py invocation -- no
# target-list files:
#
#   MODELS=$HOME/models ./bench-blend.sh \
#     --target '{"name":"blend-ocp","model":"Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ","snap":"/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ","spec_method":"dflash","maxlen":"32768"}' \
#     --target '{"name":"std-mxfp4","model":"Qwen3.8-27B-MXFP4-mtpfp8","snap":"/models/Qwen3.8-27B-MXFP4-mtpfp8","spec_method":"dflash","maxlen":"32768"}'
#
# Env knobs: LABEL, SAVE_DIR, PHASES, LAUNCH=0|1, SPEC_MAX=N, MODELS, BB_HOME.
# Everything else is forwarded to bench-async.py (--quick, --passes, --note, ...).
# SPEC_MAX=N clamps each --target's speculative depth to at most N. Default N=8
# (per-method depths are dflash 7, mtp 8, none 0), so an unset SPEC_MAX runs each
# method at its own default depth. Lower it to force a shallower sweep.
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${LABEL:-blend-sweep}
SAVE_DIR=${SAVE_DIR:-$PWD/bench}
MODELS=${MODELS:-}
PHASES=${PHASES:-all}
LAUNCH=${LAUNCH:-1}
SPEC_MAX=${SPEC_MAX:-8}
BB_HOME=${BB_HOME:-$HOME/betterbench}

# MODELS is always passed to bench (no implicit ~/models). Take it from the env,
# else from a forwarded `--models PATH`, and require it when launching.
if [ -z "$MODELS" ]; then
  prev=""
  for a in "$@"; do
    [ "$prev" = "--models" ] && MODELS=$a
    prev=$a
  done
fi
if [ "$LAUNCH" = 1 ] && [ -z "$MODELS" ]; then
  echo "[bench] MODELS is required with --launch, e.g. MODELS=\$HOME/models $0 --target '{...}'" >&2
  exit 2
fi

# --- BetterBench must exist where bench-async.py can invoke it --------------
if ! command -v betterbench >/dev/null 2>&1 && [ ! -x "$BB_HOME/.venv/bin/betterbench" ]; then
  echo "[bench] BetterBench not found; installing into $BB_HOME/.venv"
  [ -d "$BB_HOME/.git" ] || git clone --depth 1 https://github.com/GGZ14/BetterBench "$BB_HOME"
  python3 -m venv "$BB_HOME/.venv"
  "$BB_HOME/.venv/bin/pip" install -q -e "$BB_HOME"
fi
if [ -x "$BB_HOME/.venv/bin/betterbench" ]; then
  export BETTERBENCH_BIN="$BB_HOME/.venv/bin/betterbench"
fi

# --- Forwarded args, with any --target JSON spec-clamped if SPEC_MAX is set --
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
has_phases=0
for a in "${ARGS[@]}"; do
  [ "$a" = "--target" ] && has_target=1
  [ "$a" = "--phases" ] && has_phases=1
done
[ "$has_target" = 1 ] || { echo "[bench] pass at least one inline --target '{...}'" >&2; exit 2; }

cmd=(python3 bench-async.py --label "$LABEL" --save-dir "$SAVE_DIR")
[ "$has_phases" = 1 ] || cmd+=(--phases "$PHASES")
if [ "$LAUNCH" = 1 ]; then
  cmd+=(--launch --models "$MODELS")
fi
cmd+=("${ARGS[@]}")

echo "[bench] label=$LABEL phases=$([ "$has_phases" = 1 ] && echo '(from args)' || echo "$PHASES") launch=$LAUNCH models=$([ "$LAUNCH" = 1 ] && echo "$MODELS" || echo '(n/a, no --launch)') spec_max=${SPEC_MAX:-8} -> $SAVE_DIR"
"${cmd[@]}"

echo
echo "[bench] done. read $SAVE_DIR/overview.json"
