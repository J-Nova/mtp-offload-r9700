#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo >&2; echo "ERROR at line ${LINENO} (exit ${rc})" >&2; exit ${rc}' ERR

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "ERROR: run this script as your normal user, not with sudo." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Cleanup flags. `--clean` is for "I will not convert this model again": after a
# successful run it removes the intermediates and run tooling but KEEPS the
# served checkpoint in FINAL_DIR. Also settable via env (CLEAN=1, ...).
# -----------------------------------------------------------------------------
CLEAN="${CLEAN:-0}"
CLEAN_ALL="${CLEAN_ALL:-0}"
CLEAN_HF="${CLEAN_HF:-0}"

usage() {
  cat <<'EOF'
Usage: convert-blend-mxfp4.sh [--clean | --clean-all | --clean-hf]

  (no flag)     run the conversion; keep everything for fast re-runs
  --clean       after success, remove the intermediates and run scaffolding
                (QUANT_DIR, CACHE_DIR) and keep the served model in FINAL_DIR
  --clean-all   --clean plus the tooling: VENV_DIR, QUARK_DIR, and the swapfile
                (swapoff + delete + remove its /etc/fstab line)
  --clean-hf    --clean plus the Hugging Face cache for the source model and the
                AMD recipe (MODEL_ID and AMD_AWQ_REPO)

All paths are the same env-overridable defaults used for the run. Cleanup only
happens after FINAL_DIR validates, so a failed run is left intact for resume.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)     CLEAN=1 ;;
    --clean-all) CLEAN=1; CLEAN_ALL=1 ;;
    --clean-hf)  CLEAN=1; CLEAN_HF=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument '$1'. This script takes no positional arguments." >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# -----------------------------------------------------------------------------
# User-adjustable defaults
# -----------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-JetBrains/Qwen3.8-3.6-27B-blend}"
AMD_AWQ_REPO="${AMD_AWQ_REPO:-amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}"

QUANT_DIR="${QUANT_DIR:-$HOME/models/Qwen3.8-3.6-27B-blend-Quark-AWQ-MXFP4}"
FINAL_DIR="${FINAL_DIR:-$HOME/models/Qwen3.8-3.6-27B-blend-MXFP4-mtpfp8}"

QUARK_DIR="${QUARK_DIR:-$HOME/Quark-qwen38}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/quark-rocm10}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/qwen38-quark-convert}"

# Extra virtual memory for Quark's multi-device AWQ path. Quark moves completed
# decoder blocks to CPU; Linux may transparently page inactive CPU tensors here.
SWAP_FILE="${SWAP_FILE:-/swapfile-quark}"
SWAP_SIZE_GIB="${SWAP_SIZE_GIB:-128}"
SWAPPINESS="${SWAPPINESS:-20}"

FORCE="${FORCE:-0}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_SWAP_SETUP="${SKIP_SWAP_SETUP:-0}"
# Persisting the swapfile in /etc/fstab is a permanent host mutation; keep the
# active swapon (the run needs it) but skip the fstab line unless asked.
SKIP_FSTAB="${SKIP_FSTAB:-0}"
# Re-run the (slow, network-bound) dependency install even if the import check
# passes. Set to 1 after changing the pinned versions below.
FORCE_DEPS="${FORCE_DEPS:-0}"

# -----------------------------------------------------------------------------
# AWQ tuning. The AWQ search is serial per decoder layer and runs ~1 + 2*20 full
# forwards over the calibration set per layer, on whatever device the layer is
# mapped to. Two knobs move the needle without changing the algorithm:
#
# QUARK_LAYER_SPLIT   "gpu0,gpu1,cpu" text-layer counts, summing to 64. Every
#                     layer left on "cpu" runs its whole search on the CPU, so
#                     this is the largest lever. Measured cost for this blend is
#                     ~0.76 GiB/layer (bf16), plus embed 2.54 fixed on GPU0.
#                     AWQ needs ~8 GiB of live workspace atop the weights for the
#                     20-step grid forward, so a 32 GiB card holds ~24-26 layers
#                     before it OOMs; the vision tower is moved to CPU by default
#                     because it is excluded from quantization and unused by the
#                     text calibration pass. Defaults to 24,26,14 (GPU0 ~20.8 GiB
#                     free ~11 GiB, GPU1 ~19.8 GiB free ~12 GiB). Fallbacks if a
#                     card OOMs: 24,24,16 (more headroom) or 16,24,24 (original).
# QUARK_CALIB_BATCH   Calibration batch size. Quark defaults to 1, so
#                     cache_model_inps / _get_input_feat do one tiny forward per
#                     sample (64*128 = 8192 forwards) instead of 64. The 20-step
#                     grid already concatenates the calibration set, so batching
#                     does not raise the dominant peak. Must divide
#                     NUM_CALIB_DATA (the loader is drop_last=True). Quark's own
#                     AWQ benchmark uses NUM_CALIB_DATA; 32 is a safe speedup.
# -----------------------------------------------------------------------------
QUARK_LAYER_SPLIT="${QUARK_LAYER_SPLIT:-24,26,14}"
QUARK_CALIB_BATCH="${QUARK_CALIB_BATCH:-32}"
NUM_CALIB_DATA="${NUM_CALIB_DATA:-128}"
SEQ_LEN="${SEQ_LEN:-512}"
# Vision is excluded from quantization and the text calibration pass never
# executes it, so keep it off the GPU by default; set to "0" to put it back.
QUARK_VISION_DEVICE="${QUARK_VISION_DEVICE:-cpu}"

# KV cache calibration. Serving runs `--kv-cache-dtype fp8`, but a checkpoint
# with no k/v scales makes vLLM quantize the cache at scale 1.0 (it warns and
# loses accuracy). With KV_CACHE_SCHEME=fp8, Quark attaches an FP8 E4M3
# per-tensor output observer to every *k_proj/*v_proj and the export writes the
# merged per-layer scale as `<...>.k_proj.output_scale`, which vLLM's Quark
# loader maps to attn.k_scale / attn.v_scale. KV_CACHE_POST_ROPE=1 runs that
# observer inside the KV cache, after RoPE -- the exact tensor vLLM stores --
# instead of on the pre-RoPE projection output. MIN_KV_SCALE is a floor, not a
# multiplier. Set KV_CACHE_SCHEME= to disable KV calibration.
KV_CACHE_SCHEME="${KV_CACHE_SCHEME:-fp8}"
KV_CACHE_POST_ROPE="${KV_CACHE_POST_ROPE:-1}"
MIN_KV_SCALE="${MIN_KV_SCALE:-1e-3}"

# End-to-end FP8 KV-cache scale calibration. After the final checkpoint validates,
# run the post-RoPE scale capture in the serving image and merge the scales in, so
# one run yields a serve-ready checkpoint (only serving stays separate). This is
# the only stage that needs the serving image; set KV_CALIB=0 to stop after the
# weights. KV_CALIB_EXTRA_ARGS is appended to calibrate_kv.py verbatim.
KV_CALIB="${KV_CALIB:-1}"
VLLM_IMAGE="${VLLM_IMAGE:-stilldeadcode/vllm-radiance:0.9.3}"
KV_CALIB_TP="${KV_CALIB_TP:-2}"
KV_CALIB_GPU_UTIL="${KV_CALIB_GPU_UTIL:-0.75}"
KV_CALIB_EXTRA_ARGS="${KV_CALIB_EXTRA_ARGS:-}"

# Optional quantization plan (see quant_plan.py and QUANT-PLAN.md). When set, it
# supplies the global scheme/algorithm, the exclude list, the version tag and the
# MTP policy, and can import all of them from an existing checkpoint via
# "reference_model". Empty -> the historical hardcoded behavior (mxfp4 + awq,
# vision/lm_head/mtp excluded, MTP rewritten to fp8).
QUANT_PLAN="${QUANT_PLAN:-}"
if [[ -z "$QUANT_PLAN" && -f "$SCRIPT_DIR/quant-plan.json" ]]; then
  QUANT_PLAN="$SCRIPT_DIR/quant-plan.json"
fi

export QUARK_LAYER_SPLIT QUARK_CALIB_BATCH NUM_CALIB_DATA SEQ_LEN QUARK_VISION_DEVICE
export KV_CACHE_SCHEME KV_CACHE_POST_ROPE MIN_KV_SCALE QUANT_PLAN
export KV_CALIB VLLM_IMAGE KV_CALIB_TP KV_CALIB_GPU_UTIL KV_CALIB_EXTRA_ARGS

# Fail fast on a malformed split / non-dividing batch, before any model load.
IFS=',' read -r SPLIT_G0 SPLIT_G1 SPLIT_CPU <<<"$QUARK_LAYER_SPLIT"
for v in "$SPLIT_G0" "$SPLIT_G1" "$SPLIT_CPU"; do
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "ERROR: QUARK_LAYER_SPLIT must be three non-negative ints, got '$QUARK_LAYER_SPLIT'" >&2; exit 1; }
done
if (( SPLIT_G0 + SPLIT_G1 + SPLIT_CPU != 64 )); then
  echo "ERROR: QUARK_LAYER_SPLIT '$QUARK_LAYER_SPLIT' must sum to 64 layers (got $((SPLIT_G0 + SPLIT_G1 + SPLIT_CPU)))" >&2
  exit 1
fi
if (( QUARK_CALIB_BATCH < 1 || NUM_CALIB_DATA % QUARK_CALIB_BATCH != 0 )); then
  echo "ERROR: QUARK_CALIB_BATCH=$QUARK_CALIB_BATCH must be >=1 and divide NUM_CALIB_DATA=$NUM_CALIB_DATA (drop_last=True would silently drop samples)." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# ROCm / Quark runtime environment
# -----------------------------------------------------------------------------
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx1201}"

# PyTorch still exposes ROCm devices through torch.cuda.* APIs.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# The AWQ grid allocates/frees large activation blocks every step; without this
# the 32 GiB cards lose ~1.75 GiB to fragmentation and OOM on the first layer
# (the OOM message itself recommends expandable_segments). Must be set before
# torch initializes CUDA.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Quality-preserving AWQ acceleration (awq_optimize.py): record/replay each
# layer's scale+clip results, keyed by a hash of the whole run configuration.
# The generated wrapper loads it by absolute path before Quark runs.
export QUANT_DIR
export AWQ_OPTIMIZE="$SCRIPT_DIR/awq_optimize.py"
if [[ ! -f "$AWQ_OPTIMIZE" ]]; then
  echo "ERROR: missing $AWQ_OPTIMIZE (needed for AWQ record/replay)." >&2
  exit 1
fi
# QUARK_AWQ_CACHE=0        run Quark's AWQ completely unmodified
# QUARK_AWQ_NO_RESUME=1    ignore an existing checkpoint (still writes a new one)
# QUARK_AWQ_CACHE_FORCE=1  resume despite a config-hash mismatch

export TOKENIZERS_PARALLELISM=false

# Gated/private blends: pass the token through to the child processes explicitly
# so both the preflight and Quark's snapshot_download see it.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
# HF_HUB_OFFLINE=1 forces the source checkpoint to resolve from cache (no network).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Quark's built-in low-memory AWQ mode + quantization buffer reuse.
export QUARK_AWQ_MEMORY_OPTIMIZATION=1
export QUARK_ENABLE_BUFFER_REUSE=1

mkdir -p "$CACHE_DIR" "$HOME/models" "$(dirname "$VENV_DIR")"
AMD_AWQ_CONFIG="$CACHE_DIR/amd-qwen38-awq.json"
WRAPPER="$CACHE_DIR/run-qwen38-quant.py"

banner() {
  printf '\n============================================================\n'
  printf ' %s\n' "$1"
  printf '============================================================\n\n'
}

# Remove what this run created, keeping the served checkpoint in FINAL_DIR.
# Only called when FINAL_DIR validates, and only under --clean/--clean-all/--clean-hf.
cleanup_run() {
  banner "Cleanup"
  if [[ -d "$CACHE_DIR" ]]; then
    echo "  remove run scaffolding: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
  fi
  if [[ -d "$QUANT_DIR" ]]; then
    echo "  remove raw Quark intermediate: $QUANT_DIR"
    rm -rf "$QUANT_DIR"
  fi

  if [[ "$CLEAN_ALL" == "1" ]]; then
    if [[ -d "$VENV_DIR" ]]; then
      echo "  remove venv: $VENV_DIR"
      rm -rf "$VENV_DIR"
    fi
    if [[ -d "$QUARK_DIR" ]]; then
      echo "  remove Quark checkout: $QUARK_DIR"
      rm -rf "$QUARK_DIR"
    fi
    if swapon --show=NAME --noheadings 2>/dev/null | awk '{$1=$1};1' | grep -Fxq "$SWAP_FILE"; then
      echo "  swapoff $SWAP_FILE"
      sudo swapoff "$SWAP_FILE" || true
    fi
    if [[ -f "$SWAP_FILE" ]]; then
      echo "  remove $SWAP_FILE"
      sudo rm -f "$SWAP_FILE"
    fi
    if grep -Fq "$SWAP_FILE" /etc/fstab 2>/dev/null; then
      echo "  remove $SWAP_FILE from /etc/fstab"
      sudo sed -i "\#${SWAP_FILE}#d" /etc/fstab
    fi
  fi

  if [[ "$CLEAN_HF" == "1" ]]; then
    local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    local src_cache="$hub/models--${MODEL_ID//\//--}"
    local amd_cache="$hub/models--${AMD_AWQ_REPO//\//--}"
    for d in "$src_cache" "$amd_cache"; do
      if [[ -e "$d" ]]; then
        echo "  remove HF cache: $d"
        rm -rf "$d"
      fi
    done
  fi

  echo "  kept: $FINAL_DIR"
}

# Cleanup is only safe once the served checkpoint is complete.
maybe_cleanup() {
  [[ "$CLEAN" == "1" ]] || return 0
  if [[ -f "$FINAL_DIR/config.json" ]] && has_safetensors "$FINAL_DIR"; then
    cleanup_run
  else
    echo "WARNING: --clean skipped: $FINAL_DIR is incomplete, nothing removed." >&2
  fi
}

# -----------------------------------------------------------------------------
# Structural validation of a Quark OCP MXFP4 checkpoint.
#
# A substring check for "mxfp4" in config.json is not enough: the failure this
# pipeline already hit -- AMD's bf16 MTP tensors loaded into half-width packed
# parameters -- is invisible to it, because the config still says mxfp4. So this
# inspects the real safetensors headers (single-file or sharded) and the global
# weight scheme, and asserts the MTP tensors are in the state the serving stack
# requires for the given stage:
#   quark  -> MTP still bf16 (it is excluded from the body scheme)
#   final  -> MTP rewritten to fp8_e4m3 per-channel with layer_quant_config
# -----------------------------------------------------------------------------
validate_checkpoint() {
  local dir="$1" stage="$2"
  "$PY" - "$dir" "$stage" <<'PY'
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open

root, stage = Path(sys.argv[1]), sys.argv[2]
cfg_path = root / "config.json"
if not cfg_path.is_file():
    raise SystemExit(f"ERROR: missing {cfg_path}")
qc = json.loads(cfg_path.read_text()).get("quantization_config") or {}
if qc.get("quant_method") != "quark":
    raise SystemExit(
        f"ERROR: {cfg_path} quant_method={qc.get('quant_method')!r}, expected 'quark'")

def weight_spec():
    g = (qc.get("global_quant_config") or {}).get("weight")
    if g:
        return g
    for key in ("layer_type_quant_config", "layer_quant_config"):
        for v in (qc.get(key) or {}).values():
            w = (v or {}).get("weight")
            if w:
                return w
    return {}

w = weight_spec()
gdt, gsz, sfmt = w.get("dtype"), w.get("group_size"), w.get("scale_format")
if (gdt, gsz, sfmt) != ("fp4", 32, "e8m0"):
    raise SystemExit(
        f"ERROR: {root}: body weight scheme is dtype={gdt!r} group_size={gsz!r} "
        f"scale_format={sfmt!r}, expected fp4/32/e8m0")

idx = root / "model.safetensors.index.json"
if idx.is_file():
    wm = json.loads(idx.read_text())["weight_map"]
    shards = sorted(set(wm.values()))
else:
    shards = sorted(p.name for p in root.glob("*.safetensors"))
if not shards:
    raise SystemExit(f"ERROR: {root}: no *.safetensors found")

# safetensors reports short dtype codes ("U8", "BF16", "F8_E4M3"); normalize
# both those and any torch.dtype repr to that vocabulary before comparing.
_ALIAS = {"uint8": "U8", "bfloat16": "BF16", "float16": "F16", "float32": "F32",
          "float8_e4m3fn": "F8_E4M3"}

def norm(d):
    d = str(d)
    if d.startswith("torch."):
        d = d[len("torch."):]
    return _ALIAS.get(d, d)

meta = {}
for s in shards:
    with safe_open(str(root / s), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            meta[k] = (tuple(sl.get_shape()), norm(sl.get_dtype()))

# Pick a real quantized projection dynamically: layer 0 of this blend is a GDN
# layer (linear_attn), not self_attn, so a hardcoded path does not exist.
probe = next((k for k in meta if k.endswith(".self_attn.q_proj.weight")), None)
if probe is None:
    probe = next((k for k in meta if k.endswith(".weight") and meta[k][1] == "U8"
                  and k[: -len(".weight")] + ".weight_scale" in meta), None)
if probe is None:
    raise SystemExit(f"ERROR: {root}: no quantized U8 projection found to probe")
pshape, pdt = meta[probe]
if pdt != "U8":
    raise SystemExit(f"ERROR: {root}: {probe} dtype={pdt!r}, expected U8 (packed fp4)")
scale = probe[: -len(".weight")] + ".weight_scale"
if scale not in meta:
    near = sorted(k for k in meta if k.startswith(probe[: -len(".weight")]))[:4]
    raise SystemExit(
        f"ERROR: {root}: e8m0 scale {scale} missing; near {probe[: -len('.weight')]}: {near}")
ss = meta[scale][0]
if meta[scale][1] != "U8":
    raise SystemExit(f"ERROR: {root}: {scale} dtype={meta[scale][1]!r}, expected U8 (e8m0)")
if pshape[-1] % 16 or ss[-1] != pshape[-1] // 16:
    raise SystemExit(
        f"ERROR: {root}: group-32 scale shape {ss} inconsistent with packed {pshape} "
        f"(expected last dim {pshape[-1] // 16})")

# The MTP module list and policy must come from the resolved plan when one is in
# use, so validation checks exactly the modules fp8_mtp.py rewrote (a plan may
# override mtp.module_list) and does not demand fp8 for a bf16 MTP policy.
_mtp_env = os.environ.get("PLAN_MTP_MODULE_LIST")
MTP = json.loads(_mtp_env) if _mtp_env else [
    "mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj"]
_mtp_rewrite = os.environ.get("PLAN_MTP_REWRITE", "True") not in ("False", "0", "")
missing = [m for m in MTP if m + ".weight" not in meta]
if missing:
    raise SystemExit(f"ERROR: {root}: MTP tensors missing: {missing}")

if stage == "quark":
    bad = [m for m in MTP if meta[m + ".weight"][1] not in ("BF16", "F16")]
    if bad:
        raise SystemExit(
            f"ERROR: {root}: MTP must still be bf16 before the fp8 rewrite; "
            f"these were quantized: {bad}")
    print(f"  OK: quark output -- fp4/e8m0 body, {len(meta)} tensors, MTP bf16")
elif not _mtp_rewrite:
    # bf16 MTP policy: nothing was rewritten, so the head stays unquantized and
    # must be excluded by exact module name in the final config.
    bad = [m for m in MTP if meta[m + ".weight"][1] not in ("BF16", "F16")]
    if bad:
        raise SystemExit(f"ERROR: {root}: MTP not bf16 for the bf16 policy: {bad}")
    ex = set(qc.get("exclude") or [])
    miss = [m for m in MTP if m not in ex]
    if miss:
        raise SystemExit(f"ERROR: {root}: bf16 MTP not excluded from quant: {miss}")
    print(f"  OK: final output -- fp4/e8m0 body, bf16 MTP excluded, {len(meta)} tensors")
else:
    bad = [m for m in MTP if meta[m + ".weight"][1] != "F8_E4M3"]
    if bad:
        raise SystemExit(
            f"ERROR: {root}: MTP not fp8_e4m3 after the rewrite: {bad}")
    lq = qc.get("layer_quant_config") or {}
    miss = [m for m in MTP if m not in lq]
    if miss:
        raise SystemExit(f"ERROR: {root}: layer_quant_config missing entries for {miss}")
    print(f"  OK: final output -- fp4/e8m0 body, fp8_e4m3 MTP, {len(lq)} layer entries")
PY
}

# True when the final checkpoint already carries verified KV scales. Used so a
# re-run calibrates KV only instead of redoing the whole weight pass.
kv_scales_done() {
  "$PY" "$SCRIPT_DIR/verify_kv_scales.py" "$1" >/dev/null 2>&1
}

# FP8 KV-cache scale capture in the serving image (the only stage that needs it),
# then merge the scalars into the checkpoint and verify. The host venv has no
# vLLM, so this is a docker run against VLLM_IMAGE; the capture is post-RoPE and
# names are probe-verified against the model's own load_weights by the worker.
run_kv_calibration() {
  local dir="$1"
  banner "FP8 KV-cache scale calibration"
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required for KV calibration; set KV_CALIB=0 to skip" >&2
    return 1
  fi
  local groups=()
  local rg vg
  rg="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
  vg="$(getent group video  2>/dev/null | cut -d: -f3 || true)"
  [[ -n "$rg" ]] && groups+=(--group-add "$rg")
  [[ -n "$vg" ]] && groups+=(--group-add "$vg")
  echo "  image: $VLLM_IMAGE  tp: $KV_CALIB_TP  gpu_util: $KV_CALIB_GPU_UTIL"
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    ${groups[@]+"${groups[@]}"} \
    --security-opt seccomp=unconfined --ipc=host --shm-size=16g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
    -v "$SCRIPT_DIR":"$SCRIPT_DIR" \
    -v "$(dirname "$dir")":"$(dirname "$dir")" \
    -w "$SCRIPT_DIR" \
    --entrypoint bash "$VLLM_IMAGE" -lc \
    "PYTHONPATH='$SCRIPT_DIR/tools/kv_calib' python3 tools/kv_calib/calibrate_kv.py \
       --modeldir '$dir' --outputdir '$dir' --textOnly -tp '$KV_CALIB_TP' \
       --gpu-memory-utilization '$KV_CALIB_GPU_UTIL' ${KV_CALIB_EXTRA_ARGS}" \
    || { echo "ERROR: KV calibration failed; weights are complete but carry no calibrated KV scales" >&2; return 1; }
  "$PY" "$SCRIPT_DIR/tools/kv_calib/merge_kv_scales.py" --modeldir "$dir" \
    || { echo "ERROR: merging KV scales failed" >&2; return 1; }
  "$PY" "$SCRIPT_DIR/verify_kv_scales.py" "$dir" \
    || { echo "ERROR: KV scales did not verify" >&2; return 1; }
}

banner "Qwen3.8/3.6 Blend -> Quark AWQ MXFP4 -> Radiance"
echo "Source: $MODEL_ID"
echo "Quant:  $QUANT_DIR"
echo "Final:  $FINAL_DIR"
echo "GPUs:   $HIP_VISIBLE_DEVICES"
G0_END=$((SPLIT_G0 - 1)); G1_END=$((SPLIT_G0 + SPLIT_G1 - 1)); CPU_START=$((SPLIT_G0 + SPLIT_G1))
echo "Layout: GPU0 vision+embed+L0-${G0_END} | GPU1 L${SPLIT_G0}-${G1_END} | CPU L${CPU_START}-63+heads [split $QUARK_LAYER_SPLIT]"
echo "AWQ:    calib ${NUM_CALIB_DATA}x${SEQ_LEN}, batch $QUARK_CALIB_BATCH"

# -----------------------------------------------------------------------------
# Swap: 64 GiB RAM was not enough once Quark started moving completed layers to
# CPU. Swap is intentionally Linux paging, NOT Accelerate device_map='disk'.
# -----------------------------------------------------------------------------
if [[ "$SKIP_SWAP_SETUP" != "1" ]]; then
  banner "Swap / virtual memory"

  if ! swapon --show=NAME --noheadings 2>/dev/null | awk '{$1=$1};1' | grep -Fxq "$SWAP_FILE"; then
    if [[ ! -f "$SWAP_FILE" ]]; then
      echo "Creating ${SWAP_SIZE_GIB} GiB swapfile at $SWAP_FILE ..."
      sudo fallocate -l "${SWAP_SIZE_GIB}G" "$SWAP_FILE"
      sudo chmod 600 "$SWAP_FILE"
      sudo mkswap "$SWAP_FILE"
    fi

    sudo swapon "$SWAP_FILE"
  fi

  if [[ "$SKIP_FSTAB" != "1" ]] && ! grep -Fq "$SWAP_FILE" /etc/fstab; then
    printf '%s none swap sw 0 0\n' "$SWAP_FILE" | sudo tee -a /etc/fstab >/dev/null
  fi

  sudo sysctl -w "vm.swappiness=$SWAPPINESS" >/dev/null

  free -h
  echo
  swapon --show
fi

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
if [[ "$SKIP_APT" != "1" ]]; then
  banner "System dependencies"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git build-essential python3-venv python3-dev ninja-build
fi

# -----------------------------------------------------------------------------
# Verify ROCm before touching Python
# -----------------------------------------------------------------------------
banner "ROCm"
if [[ ! -x "$ROCM_PATH/bin/hipcc" ]]; then
  echo "ERROR: $ROCM_PATH/bin/hipcc not found." >&2
  exit 1
fi
"$ROCM_PATH/bin/hipcc" --version | head -n 6

# -----------------------------------------------------------------------------
# Clean Quark release/0.12 checkout
# -----------------------------------------------------------------------------
banner "Clean AMD Quark release/0.12"
if [[ ! -d "$QUARK_DIR/.git" ]]; then
  git clone --branch release/0.12 --single-branch https://github.com/AMD/Quark.git "$QUARK_DIR"
else
  git -C "$QUARK_DIR" fetch origin release/0.12
  git -C "$QUARK_DIR" checkout release/0.12
  git -C "$QUARK_DIR" reset --hard origin/release/0.12
fi

echo "Quark source:"
git -C "$QUARK_DIR" rev-parse --short HEAD

# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------
banner "Python environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP=("$PY" -m pip)

"${PIP[@]}" install -U pip wheel packaging ninja 'setuptools<82'

# -----------------------------------------------------------------------------
# ROCm 10 PyTorch for gfx1201/R9700
# -----------------------------------------------------------------------------
banner "Checking ROCm 10 PyTorch"
if ! "$PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
    sys.exit(1)

ok = (
    torch.__version__ == "2.12.0+rocm10.0.0"
    and torch.version.hip is not None
)
print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
sys.exit(0 if ok else 1)
PY
then
  echo "Installing AMD ROCm 10 PyTorch wheels ..."
  "${PIP[@]}" install -U \
    --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    'torch[device-gfx1201]==2.12.0+rocm10.0.0' \
    'torchvision[device-gfx1201]==0.27.0+rocm10.0.0' \
    'torchaudio==2.11.0+rocm10.0.0'
fi

# -----------------------------------------------------------------------------
# Quark dependencies. Keep Transformers 5.2.0: this is the release/0.12-compatible
# Qwen3.5 version we already reached real AWQ with.
# -----------------------------------------------------------------------------
banner "Quark dependencies"
# This install is network-bound and slow. Skip it when the environment already
# imports what Quark needs, so a resumed run does not re-resolve ~20 packages.
if [[ "$FORCE_DEPS" != "1" ]] && "$PY" - <<'PY' 2>/dev/null
import sys
try:
    import quark, transformers, accelerate, datasets, safetensors
    import compressed_tensors  # noqa: F401
    import evaluate  # noqa: F401
    import lm_eval  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0 if transformers.__version__ == "5.2.0" else 1)
PY
then
  echo "  amd-quark + transformers 5.2.0 + deps already importable; skipping install"
  echo "  (FORCE_DEPS=1 to reinstall anyway)"
else
  "${PIP[@]}" install -U \
    'amd-quark==0.12.post1' \
    'transformers==5.2.0' \
    accelerate \
    addict \
    ai2-olmo \
    'compressed-tensors>=0.15.0' \
    datasets \
    easydict \
    einops \
    'evaluate>=0.4.0' \
    'gguf>=0.10.0' \
    lm-eval \
    matplotlib \
    nltk \
    pillow \
    psutil \
    tiktoken \
    transformers_stream_generator \
    zstandard \
    safetensors \
    huggingface_hub \
    sentencepiece \
    protobuf \
    'setuptools<82'

  # Reassert exact Transformers version after dependency resolution.
  "${PIP[@]}" install -U 'transformers==5.2.0' 'setuptools<82'
fi

# -----------------------------------------------------------------------------
# Qwen3.5 fast-path kernels
# -----------------------------------------------------------------------------
banner "Qwen fast-path kernels"
if ! "$PY" - <<'PY'
import fla
print("Flash Linear Attention: OK")
PY
then
  "${PIP[@]}" install -U --no-deps \
    git+https://github.com/fla-org/flash-linear-attention.git
fi

if ! "$PY" - <<'PY'
import causal_conv1d
print("causal-conv1d: OK")
PY
then
  CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  "${PIP[@]}" install --no-build-isolation --no-cache-dir \
    git+https://github.com/Dao-AILab/causal-conv1d.git
fi

"${PIP[@]}" install -U 'setuptools<82'

# -----------------------------------------------------------------------------
# Fix old root-owned Hugging Face cache files from previous sudo runs.
# -----------------------------------------------------------------------------
if [[ -d "$HOME/.cache/huggingface" ]]; then
  if find "$HOME/.cache/huggingface" ! -user "$USER" -print -quit 2>/dev/null | grep -q .; then
    echo "Fixing Hugging Face cache ownership ..."
    sudo chown -R "$USER:$USER" "$HOME/.cache/huggingface"
  fi
  if [[ -d "$HOME/.cache/huggingface/datasets" ]]; then
    find "$HOME/.cache/huggingface/datasets" -type f -name '*.lock' -delete 2>/dev/null || true
  fi
fi

# -----------------------------------------------------------------------------
# Verify the actual runtime before spending time downloading/loading the model.
# -----------------------------------------------------------------------------
banner "Environment verification"
"$PY" - <<'PY'
import sys
import torch
import transformers

print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Transformers:", transformers.__version__)
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.device_count() < 2:
    raise SystemExit("ERROR: two visible GPUs are required for this script")

for i in range(2):
    name = torch.cuda.get_device_name(i)
    if "R9700" not in name:
        print(f"WARNING: GPU {i} is {name!r}, expected an R9700")

import quark  # noqa: F401
import fla  # noqa: F401
import causal_conv1d  # noqa: F401
print("Quark: OK")
print("FLA: OK")
print("causal-conv1d: OK")
PY

# -----------------------------------------------------------------------------
# Preflight the SOURCE checkpoint before any quantization starts. This is the
# last cheap moment to fail: a wrong model type, a layer-count drift, or a blend
# missing its MTP head would otherwise surface hours into the AWQ pass.
# -----------------------------------------------------------------------------
banner "Source checkpoint preflight"
export MODEL_ID
"$PY" - <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

mid = os.environ["MODEL_ID"]
local = Path(mid)
if local.is_dir():
    root = local
else:
    root = Path(snapshot_download(
        mid,
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.jinja", "*.txt", "*.model"],
    ))

cfg = json.loads((root / "config.json").read_text())
mt = cfg.get("model_type")
if mt != "qwen3_5":
    raise SystemExit(f"ERROR: {mid}: model_type={mt!r}, expected 'qwen3_5'")
nl = (cfg.get("text_config") or {}).get("num_hidden_layers")
if nl != 64:
    raise SystemExit(f"ERROR: {mid}: num_hidden_layers={nl!r}, expected 64")

idx = root / "model.safetensors.index.json"
if idx.is_file():
    wm = json.loads(idx.read_text())["weight_map"]
else:
    wm = {}
    for p in sorted(root.glob("*.safetensors")):
        from safetensors import safe_open
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                wm[k] = p.name
if not wm:
    raise SystemExit(f"ERROR: {mid}: no safetensors tensors found under {root}")

mtp = [k for k in wm if k.startswith("mtp.")]
if not mtp:
    raise SystemExit(
        f"ERROR: {mid}: no mtp.* tensors. SPEC_METHOD=mtp needs the MTP head; "
        "this blend would produce a drafter-less checkpoint.")
print(f"  model_type=qwen3_5, 64 layers, {len(wm)} tensors in {len(set(wm.values()))} shard(s)")
print(f"  mtp tensors: {len(mtp)}")
print(f"  resolved: {root}")
PY

# -----------------------------------------------------------------------------
# Disk preflight. The Quark output is ~19 GiB and the final is another ~19 GiB.
# -----------------------------------------------------------------------------
for d in "$QUANT_DIR" "$FINAL_DIR"; do
  parent="$(dirname "$d")"
  mkdir -p "$parent"
  free_gib=$(df -BG --output=avail "$parent" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "$free_gib" && "$free_gib" -lt 45 ]]; then
    echo "WARNING: only ${free_gib} GiB free on $parent; a full run wants ~45 GiB" >&2
  fi
done

# -----------------------------------------------------------------------------
# VRAM governor. Sizes each GPU against the WORST layer before anything loads,
# instead of discovering the shortfall by OOM-ing mid-AWQ. Per-layer weight bytes
# come from the source headers; the AWQ workspace is derived from the grid
# forward's live tensors (gate/up/outputs for T = num_calib_data * seq_len
# tokens) plus a safety margin. If the chosen QUARK_LAYER_SPLIT does not fit it
# fails with a suggested split; QUARK_GOVERNOR=warn downgrades that to a warning.
# -----------------------------------------------------------------------------
banner "VRAM governor"
export MODEL_ID
QUARK_GOVERNOR="${QUARK_GOVERNOR:-error}"
"$PY" - <<'PY' || { [[ "$QUARK_GOVERNOR" == "warn" ]] || exit 1; }
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

def fail(msg):
    if os.environ.get("QUARK_GOVERNOR", "error") == "warn":
        print("  WARNING: " + msg)
        sys.exit(0)
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)

try:
    import torch
    from safetensors import safe_open
except Exception as exc:
    print(f"  (skipped: {exc!r})")
    sys.exit(0)

mid = os.environ["MODEL_ID"]
src = Path(mid)
if not src.is_dir():
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(
        mid, local_files_only=True,
        allow_patterns=["config.json", "model.safetensors.index.json",
                        "model.safetensors"]))

cfg = json.loads((src / "config.json").read_text())
tc = cfg.get("text_config", {})
H = tc.get("hidden_size") or cfg.get("hidden_size")
I = tc.get("intermediate_size") or cfg.get("intermediate_size")
nl = tc.get("num_hidden_layers")
if not (H and I and nl):
    print("  (skipped: config lacks hidden_size/intermediate_size/num_hidden_layers)")
    sys.exit(0)

IS = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "F16": 2,
      "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "F64": 8, "BOOL": 1}

def nbytes(shape, dt):
    n = 1
    for d in shape:
        n *= d
    return n * IS.get(dt, 2)

per_layer = [0] * nl
fixed = defaultdict(int)
idx = src / "model.safetensors.index.json"
if idx.is_file():
    wm = json.loads(idx.read_text())["weight_map"]
    by_shard = defaultdict(list)
    for k, s in wm.items():
        by_shard[s].append(k)
    files = [(src / s, keys) for s, keys in by_shard.items()]
else:
    files = [(src / "model.safetensors", None)]
for path, keys in files:
    if not path.is_file():
        continue
    with safe_open(str(path), framework="pt") as f:
        for k in (keys if keys is not None else list(f.keys())):
            sl = f.get_slice(k)
            b = nbytes(sl.get_shape(), str(sl.get_dtype()))
            p = k.split(".")
            if len(p) >= 5 and p[:3] == ["model", "language_model", "layers"] and p[3].isdigit():
                li = int(p[3])
                if 0 <= li < nl:
                    per_layer[li] += b
                    continue
            fixed[".".join(p[:3]) if len(p) >= 3 else k] += b

def pref(prefix):
    return sum(v for k, v in fixed.items() if k.startswith(prefix))

embed = pref("model.language_model.embed_tokens")
vision = pref("model.visual")
cpu_fixed = (pref("lm_head") + pref("mtp") + pref("model.language_model.norm")
             + pref("model.language_model.rotary_emb"))

T = int(os.environ.get("NUM_CALIB_DATA", "128")) * int(os.environ.get("SEQ_LEN", "512"))
# Peak of the 20-step grid forward over the fused gate/up + down on T tokens.
workspace = ((2.5 * T * I + 2.0 * T * H) * 2 + 2 * (2 * H * I) * 2) * 1.10
# A little fixed slack for the allocator, accelerate hooks and cached inputs.
workspace += 1.5e9

g0, g1, gcpu = (int(x) for x in os.environ.get("QUARK_LAYER_SPLIT", "24,26,14").split(","))
vision_dev = os.environ.get("QUARK_VISION_DEVICE", "cpu")
devs = list(range(torch.cuda.device_count()))[:2]
caps = {d: torch.cuda.get_device_properties(d).total_memory * 0.97 for d in devs}

layer_bytes = sum(per_layer) / len(per_layer)
def need_for(dev_index, count):
    if count == 0 or dev_index not in devs:
        return 0.0
    fixed_dev = embed if dev_index == 0 else 0.0
    if str(vision_dev) == str(dev_index) or str(vision_dev) == f"cuda:{dev_index}":
        fixed_dev += vision
    return fixed_dev + count * layer_bytes + workspace

print(f"  layer ~{layer_bytes/2**30:.2f} GiB x {nl}; embed {embed/2**30:.2f}; "
      f"vision {vision/2**30:.2f} -> {vision_dev}; workspace est {workspace/2**30:.2f} GiB")
fits = True
for name, dev, count in (("GPU0", 0, g0), ("GPU1", 1, g1)):
    need = need_for(dev, count)
    cap = caps.get(dev, 0)
    if dev in devs:
        verdict = "OK" if need <= cap else "OVER"
        fits &= need <= cap
        print(f"  {name}: {count} layers -> need {need/2**30:5.2f} / {cap/2**30:5.2f} GiB ({verdict})")
print(f"  CPU: {gcpu} layers (~{(gcpu*layer_bytes + cpu_fixed)/2**30:.1f} GiB host)")

if not fits:
    # Suggest a split that fits, largest layers first, least-loaded GPU wins.
    order = sorted(range(nl), key=lambda i: (-per_layer[i], i))
    counts = {0: 0, 1: 0}
    for li in order:
        choices = [d for d in devs if need_for(d, counts[d] + 1) <= caps[d]]
        if not choices:
            break
        d = min(choices, key=lambda d: (need_for(d, counts[d] + 1) / caps[d], d))
        counts[d] += 1
    sug = f"{counts.get(0,0)},{counts.get(1,0)},{nl - counts.get(0,0) - counts.get(1,0)}"
    fail(f"QUARK_LAYER_SPLIT={g0},{g1},{gcpu} does not fit the estimate. "
         f"Try QUARK_LAYER_SPLIT={sug} (or QUARK_GOVERNOR=warn to override).")
print("  governor: split fits")
PY

# -----------------------------------------------------------------------------
# Fetch AMD's published Qwen3.8 AWQ algorithm recipe. We intentionally use the
# INT4 checkpoint ONLY as the source of AWQ scaling metadata. The target scheme
# below remains MXFP4.
# -----------------------------------------------------------------------------
banner "Fetching AMD official Qwen3.8 AWQ configuration"
export AMD_AWQ_REPO AMD_AWQ_CONFIG
"$PY" - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ["AMD_AWQ_REPO"]
out = Path(os.environ["AMD_AWQ_CONFIG"])

config_path = hf_hub_download(repo_id=repo, filename="config.json")
with open(config_path, "r", encoding="utf-8") as f:
    root = json.load(f)

def walk(value, path="root"):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, f"{path}[{i}]")

candidates = []
for path, obj in walk(root):
    if (
        obj.get("name") == "awq"
        and isinstance(obj.get("scaling_layers"), list)
        and obj.get("model_decoder_layers")
    ):
        candidates.append((path, obj))

if not candidates:
    raise RuntimeError(
        f"AMD reference checkpoint {repo!r} contains no usable AWQ algo config."
    )

# Prefer the candidate with the fullest scaling recipe.
path, src = max(candidates, key=lambda item: len(item[1]["scaling_layers"]))

# Keep only the fields understood by Quark AWQConfig. Do not copy the INT4
# quantization scheme itself; this file is algorithm metadata only.
recipe = {
    "name": "awq",
    "scaling_layers": src["scaling_layers"],
    "model_decoder_layers": src["model_decoder_layers"],
}

if recipe["model_decoder_layers"] != "model.language_model.layers":
    raise RuntimeError(
        "Unexpected AMD Qwen3.8 decoder path: "
        + repr(recipe["model_decoder_layers"])
    )

if not recipe["scaling_layers"]:
    raise RuntimeError("AMD AWQ scaling_layers is empty")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")

qmeta = root.get("quantization_config", {})
qver = None
if isinstance(qmeta, dict):
    qver = (
        qmeta.get("quark_version")
        or qmeta.get("version")
        or qmeta.get("quantizer_version")
    )

print("AMD recipe JSON path:", path)
print("AMD decoder path:", recipe["model_decoder_layers"])
print("AMD AWQ scaling groups:", len(recipe["scaling_layers"]))
if qver:
    print("AMD Quark version:", qver)
print("Saved AMD AWQ config:", out)
print(json.dumps(recipe, indent=2))
PY

# -----------------------------------------------------------------------------
# Generate a wrapper around AMD's stock quantize_quark.py.
#
# Why a wrapper:
# - release/0.12 has Qwen3.5 internals but no registered dense qwen3_5 template.
# - release/0.12's preprocess_for_quantization rejects dense qwen3_5 even though
#   this dense model already exposes normal nn.Linear modules.
# - we need an explicit GPU/GPU/CPU map with enough AWQ workspace.
# - Accelerate CPU offload can make model.device resolve to meta; calibration
#   tensors must stay real, so the dataloader wrapper forces meta -> cuda:0.
# -----------------------------------------------------------------------------
banner "Writing Qwen3.5/3.8 Quark wrapper"
export QUARK_DIR
cat > "$WRAPPER" <<'PYWRAPPER'
from __future__ import annotations

import json
import os
import runpy
from typing import Any

import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

from quark.torch import LLMTemplate
import quark.torch.utils.llm as llm_utils


# -----------------------------------------------------------------------------
# 0. Quality-preserving AWQ acceleration.
#
# awq_optimize patches AwqProcessor._search_best_scale / _search_best_clip /
# _compute_loss / _get_input_feat to record each layer's results and replay them
# verbatim on a later run. apply() and everything it calls is Quark's untouched
# code, so a replay is the same arithmetic on the same pristine inputs. The
# checkpoint is keyed by a hash of the whole run configuration and refuses to
# resume across a change. QUARK_AWQ_CACHE=0 bypasses this entirely.
# -----------------------------------------------------------------------------
import importlib.util as _ilu

_awq_opt_path = os.environ.get("AWQ_OPTIMIZE")
if not _awq_opt_path or not os.path.isfile(_awq_opt_path):
    raise SystemExit(f"ERROR: AWQ_OPTIMIZE is not a readable path: {_awq_opt_path!r}")
_spec = _ilu.spec_from_file_location("awq_optimize", _awq_opt_path)
awq_optimize = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(awq_optimize)

AWQ_CACHE = awq_optimize.prepare(
    model_id=os.environ["MODEL_ID"],
    out_dir=os.environ["QUANT_DIR"],
)
if AWQ_CACHE.active:
    awq_optimize.install(AWQ_CACHE)
    print(f"[QWEN38] AWQ cache: {AWQ_CACHE.path} "
          f"({len(AWQ_CACHE.scales)} group(s) recorded, "
          f"{len(AWQ_CACHE.clips)} layer clip set(s))")
else:
    print("[QWEN38] AWQ cache/report disabled; running Quark's AWQ unmodified")


# -----------------------------------------------------------------------------
# 1. Register the missing dense qwen3_5 template.
#    We quantize the language-model linears and leave vision/lm_head/MTP alone.
# -----------------------------------------------------------------------------
if "qwen3_5" not in LLMTemplate.list_available():
    template = LLMTemplate(
        model_type="qwen3_5",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=[
            "model.visual.*",
            "lm_head",
            "mtp.*",
        ],
    )
    LLMTemplate.register_template(template)
    print("[QWEN38] Registered qwen3_5 template")

print("[QWEN38] Exclusions:", LLMTemplate.get("qwen3_5").exclude_layers_name)


# -----------------------------------------------------------------------------
# 1b. Plan-driven version tag. The exporter stamps config["version"] from the
#     QConfig and there is no Quark CLI for it, so override it after get_config
#     returns. This is what lets a converted checkpoint match another
#     checkpoint's version string exactly (e.g. "0.13+unknown").
# -----------------------------------------------------------------------------
_PLAN_VERSION = os.environ.get("PLAN_VERSION") or ""
_PLAN_LAYER_TYPES = os.environ.get("PLAN_LAYER_TYPE_SCHEMES") or ""
if _PLAN_VERSION or (_PLAN_LAYER_TYPES and _PLAN_LAYER_TYPES != "[]"):
    _TYPE_MAP = {
        "Linear": torch.nn.Linear,
        "Conv2d": torch.nn.Conv2d,
        "ConvTranspose2d": torch.nn.ConvTranspose2d,
    }
    _layer_type_config = {}
    if _PLAN_LAYER_TYPES and _PLAN_LAYER_TYPES != "[]":
        for _tname, _sch in json.loads(_PLAN_LAYER_TYPES):
            _cls = _TYPE_MAP.get(_tname)
            if _cls is None:
                raise SystemExit(f"[QWEN38] unknown layer_type {_tname!r}")
            _layer_type_config[_cls] = _sch

    _orig_get_config = LLMTemplate.get_config

    def qwen35_get_config(self, *args, **kwargs):
        config = _orig_get_config(self, *args, **kwargs)
        if _PLAN_VERSION:
            config.version = _PLAN_VERSION
        if _layer_type_config:
            # quantize_quark.py does not forward layer_type_config, and the type
            # keys are nn.Module subclasses, so apply it here after the fact.
            self._set_layer_type_config(config, _layer_type_config)
        return config

    LLMTemplate.get_config = qwen35_get_config
    if _PLAN_VERSION:
        print(f"[QWEN38] version tag override -> {_PLAN_VERSION}")
    if _layer_type_config:
        print(f"[QWEN38] layer_type overrides -> "
              f"{ {t.__name__: s for t, s in _layer_type_config.items()} }")


# -----------------------------------------------------------------------------
# 2. Dense qwen3_5 is not MoE. Quark 0.12's generic preprocessing currently
#    rejects model_type=qwen3_5 before quantization; skip only that rejection.
# -----------------------------------------------------------------------------
_orig_preprocess = llm_utils.preprocess_for_quantization


def qwen35_preprocess_for_quantization(model: torch.nn.Module, reload: bool = False) -> None:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "qwen3_5":
        print("[QWEN38] Dense qwen3_5: skipping Quark MoE preprocessing")
        return None
    return _orig_preprocess(model, reload=reload)


llm_utils.preprocess_for_quantization = qwen35_preprocess_for_quantization


# -----------------------------------------------------------------------------
# 3. Safety net for Accelerate CPU offload.
#    quantize_quark.py uses model.device for calibration under --multi_device.
#    When Accelerate reports that as meta, create calibration tensors on cuda:0.
# -----------------------------------------------------------------------------
_orig_get_calib_dataloader = llm_utils.get_calib_dataloader


def safe_get_calib_dataloader(*args: Any, **kwargs: Any):
    device = kwargs.get("device")
    if device is not None:
        try:
            device_type = torch.device(device).type
        except Exception:
            device_type = str(device)
        if device_type == "meta":
            kwargs["device"] = torch.device("cuda:0")
            print("[QWEN38] Calibration device meta -> cuda:0")
    return _orig_get_calib_dataloader(*args, **kwargs)


llm_utils.get_calib_dataloader = safe_get_calib_dataloader


# -----------------------------------------------------------------------------
# 4. Explicit multi-device loader.
#
#    Text-layer placement is driven by QUARK_LAYER_SPLIT="gpu0,gpu1,cpu"
#    (default 16,24,24 here; convert-blend-mxfp4.sh exports 24,26,14).
#    Every layer left on CPU runs the full AWQ search on the CPU.
#
#    embed_tokens stays on GPU 0 to keep model.device real. The vision tower
#    defaults to CPU (QUARK_VISION_DEVICE), since it is excluded from
#    quantization and idle during the text calibration pass.
# -----------------------------------------------------------------------------
def qwen35_get_model(
    ckpt_path: str,
    data_type: str = "auto",
    device: str = "cuda",
    multi_gpu: str | bool | None = False,
    multi_device: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
):
    if data_type == "float16":
        model_dtype: torch.dtype | str = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32
    elif data_type == "auto":
        model_dtype = "auto"
    else:
        raise ValueError(f"Unsupported data_type={data_type!r}")

    config = AutoConfig.from_pretrained(
        ckpt_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    if getattr(config, "model_type", None) != "qwen3_5":
        return _ORIGINAL_GET_MODEL(
            ckpt_path,
            data_type=data_type,
            device=device,
            multi_gpu=multi_gpu,
            multi_device=multi_device,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"This Qwen3.8 conversion layout requires 2 GPUs; found {torch.cuda.device_count()}"
        )

    text_config = getattr(config, "text_config", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers != 64:
        raise RuntimeError(f"Expected 64 Qwen3.8 decoder layers, got {num_layers!r}")

    # Text-layer placement. Every layer mapped to "cpu" runs its entire AWQ
    # search (1 + 2*20 forwards over the calibration set) on the CPU, so this is
    # the dominant time lever. QUARK_LAYER_SPLIT is "gpu0,gpu1,cpu".
    split = os.environ.get("QUARK_LAYER_SPLIT", "16,24,24")
    try:
        g0, g1, g_cpu = (int(x) for x in split.split(","))
    except ValueError as exc:
        raise RuntimeError(f"QUARK_LAYER_SPLIT={split!r} must be 'gpu0,gpu1,cpu'") from exc
    if min(g0, g1, g_cpu) < 0 or g0 + g1 + g_cpu != 64:
        raise RuntimeError(f"QUARK_LAYER_SPLIT={split!r} must be non-negative and sum to 64")

    vision_device = os.environ.get("QUARK_VISION_DEVICE", "cpu")
    vision_device = int(vision_device) if vision_device.isdigit() else vision_device

    device_map: dict[str, int | str] = {
        # embed_tokens on GPU 0 is what keeps model.device on a real CUDA device
        # during calibration. The vision tower is excluded from quantization and
        # is never executed by the text calibration pass, so it defaults to CPU
        # to leave more of GPU 0's 32 GiB for the AWQ workspace.
        "model.visual": vision_device,
        "model.language_model.embed_tokens": 0,

        # Final text components do not participate in the AWQ MLP search.
        "model.language_model.norm": "cpu",
        "model.language_model.rotary_emb": "cpu",
        "lm_head": "cpu",
        # MTP is a top-level module and is excluded from quantization; it must
        # still be placed explicitly or Accelerate has an unmapped parameter
        # group to resolve while the rest of the model is device-mapped.
        "mtp": "cpu",
    }

    parallel = os.environ.get("QUARK_AWQ_PARALLEL", "0").strip().lower() not in ("", "0", "false", "no")
    total_gpu = g0 + g1
    if parallel:
        # Interleave the GPU block so consecutive layers land on different GPUs;
        # the parallel search scheduler needs both GPUs busy at once. Exact
        # per-GPU counts are preserved via a Bresenham split, so the VRAM
        # governor's sizing is unchanged, and per-layer bit-identity across
        # GPUs was established by the placement-invariance gate.
        for j in range(total_gpu):
            owner = 0 if ((j + 1) * g0) // max(total_gpu, 1) > (j * g0) // max(total_gpu, 1) else 1
            device_map[f"model.language_model.layers.{j}"] = owner
        for i in range(total_gpu, 64):
            device_map[f"model.language_model.layers.{i}"] = "cpu"
    else:
        for i in range(0, g0):
            device_map[f"model.language_model.layers.{i}"] = 0
        for i in range(g0, g0 + g1):
            device_map[f"model.language_model.layers.{i}"] = 1
        for i in range(g0 + g1, 64):
            device_map[f"model.language_model.layers.{i}"] = "cpu"

    print(f"[QWEN38] Explicit multi-device AWQ placement (QUARK_LAYER_SPLIT={split})")
    if parallel:
        print(f"[QWEN38] PARALLEL: interleaved GPU placement (GPU0 {g0}, GPU1 {g1}, CPU {g_cpu})")
    else:
        print(f"[QWEN38] GPU 0: embedding + layers 0-{g0 - 1} ({g0} layers); vision -> {vision_device}")
        print(f"[QWEN38] GPU 1: layers {g0}-{g0 + g1 - 1} ({g1} layers)")
    print(f"[QWEN38] CPU:   layers {g0 + g1}-63 ({g_cpu} layers) + norm + rotary + lm_head + mtp")
    if g_cpu:
        print(
            f"[QWEN38] WARNING: {g_cpu} layer(s) run the AWQ search on CPU. "
            "If VRAM allows, lower the cpu count in QUARK_LAYER_SPLIT."
        )
    print("[QWEN38] Loading Qwen3_5ForConditionalGeneration")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        ckpt_path,
        device_map=device_map,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config._name_or_path = ckpt_path

    print("[QWEN38] Decoder layers:", len(model.model.language_model.layers))
    print("[QWEN38] model.device:", model.device)
    print("[QWEN38] hf_device_map:")
    for name, mapped_device in getattr(model, "hf_device_map", {}).items():
        print(f"  {name}: {mapped_device}")

    bad_disk = {
        name: mapped_device
        for name, mapped_device in getattr(model, "hf_device_map", {}).items()
        if str(mapped_device) == "disk"
    }
    if bad_disk:
        raise RuntimeError(
            "Accelerate unexpectedly used explicit disk offload; this script uses "
            f"CPU + Linux swap instead. Disk entries: {bad_disk}"
        )

    # This should be cuda:0 because embed_tokens is on GPU 0.
    # The calibration-dataloader monkeypatch above is an additional fallback.
    if str(model.device) == "meta":
        print(
            "[QWEN38] WARNING: model.device is still meta; "
            "calibration tensors will be forced to cuda:0"
        )

    model_dtype_real = next(model.parameters()).dtype
    return model, model_dtype_real


_ORIGINAL_GET_MODEL = llm_utils.get_model
llm_utils.get_model = qwen35_get_model


# quantize_quark.py imports these functions from quark.torch.utils.llm when the
# file starts executing, so the monkeypatches above are picked up by runpy.
quantize_script = os.path.join(
    os.environ["QUARK_DIR"],
    "examples",
    "torch",
    "language_modeling",
    "llm_ptq",
    "quantize_quark.py",
)

runpy.run_path(quantize_script, run_name="__main__")
PYWRAPPER

# Syntax-check the generated wrapper before starting a multi-hour conversion.
"$PY" -m py_compile "$WRAPPER"
echo "Wrapper syntax: OK"

# -----------------------------------------------------------------------------
# Output handling + stage resume.
#
# The Quark AWQ pass is the expensive stage and has no mid-run resume, so once
# its output is complete it is never repeated on a re-run. The fp8 rewrite is
# cheap and idempotent, so a partial $FINAL_DIR is simply rebuilt over.
# FORCE=1 wipes both stages.
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$QUANT_DIR")" "$(dirname "$FINAL_DIR")"

has_safetensors() { compgen -G "$1/*.safetensors" >/dev/null 2>&1; }

# A directory holding only AWQ record/replay artifacts is resumable, not a dead
# partial Quark export: awq_optimize.py has recorded completed layers and will
# replay them, and its own config-hash check refuses an incompatible cache.
resumable_awq() {
  [[ -f "$1/awq_cache.pt" ]] || return 1
  ! find "$1" -maxdepth 1 -type f \
      ! -name 'awq_cache.pt' ! -name 'awq_cache.pt.tmp' \
      ! -name 'awq_report.csv' ! -name 'quark_profile.*' \
      -print -quit 2>/dev/null | grep -q .
}

if [[ "$FORCE" == "1" ]]; then
  banner "Removing previous output (FORCE=1)"
  rm -rf "$QUANT_DIR" "$FINAL_DIR"
fi

QUARK_DONE=0
[[ -f "$QUANT_DIR/config.json" ]] && has_safetensors "$QUANT_DIR" && QUARK_DONE=1
FINAL_DONE=0
[[ -f "$FINAL_DIR/config.json" ]] && has_safetensors "$FINAL_DIR" && FINAL_DONE=1

# -----------------------------------------------------------------------------
# Resolve the optional quantization plan before any resume decision. A plan must
# be identical across runs that reuse an existing stage: otherwise the skipped
# Quark pass would silently pair the new plan (version/MTP/exclude) with an old
# body. The resolved plan's hash is recorded in each stage directory for that.
# -----------------------------------------------------------------------------
QUARK_SCHEME="mxfp4"
QUARK_ALGO="awq"
QUARK_EXCLUDE=()
PLAN_RESOLVED_HASH=""
if [[ -n "$QUANT_PLAN" ]]; then
  if [[ ! -f "$QUANT_PLAN" ]]; then
    echo "ERROR: QUANT_PLAN=$QUANT_PLAN is not a file" >&2
    exit 1
  fi
  PLAN_JSON_OUT="$CACHE_DIR/quant-plan-resolved.json"
  banner "Resolving quantization plan"
  "$PY" "$SCRIPT_DIR/quant_plan.py" resolve --plan "$QUANT_PLAN" \
    --json-out "$PLAN_JSON_OUT" --shell-out "$PLAN_JSON_OUT.sh" || exit 1
  set -a
  # shellcheck disable=SC1090
  source "$PLAN_JSON_OUT.sh"
  set +a
  QUARK_SCHEME="$PLAN_SCHEME"
  QUARK_ALGO="$PLAN_ALGO"
  mapfile -t QUARK_EXCLUDE < <("$PY" -c 'import json,os; [print(e) for e in json.loads(os.environ["PLAN_EXCLUDE"])]')
  # The plan is the single source of truth for KV when it is present.
  if [[ -n "${PLAN_KV_SCHEME:-}" ]]; then
    KV_CACHE_SCHEME="$PLAN_KV_SCHEME"
    if [[ "${PLAN_KV_POST_ROPE:-}" == "True" ]]; then KV_CACHE_POST_ROPE=1; else KV_CACHE_POST_ROPE=0; fi
    MIN_KV_SCALE="${PLAN_KV_MIN_SCALE:-$MIN_KV_SCALE}"
  fi
  PLAN_RESOLVED_HASH="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PLAN_JSON_OUT")"
  echo "  scheme=$QUARK_SCHEME algo=$QUARK_ALGO exclude=${#QUARK_EXCLUDE[@]} entries plan=$PLAN_RESOLVED_HASH"

  # Refuse to reuse a stage that was built with a different plan (or before plans
  # existed): the expensive pass would be skipped while its artifact is relabeled.
  if [[ "$QUARK_DONE" == "1" ]]; then
    stored="$(cat "$QUANT_DIR/.quant-plan.sha256" 2>/dev/null || true)"
    if [[ "$stored" != "$PLAN_RESOLVED_HASH" ]]; then
      echo "ERROR: $QUANT_DIR was quantized with a different plan (or predates plans)." >&2
      echo "  The Quark pass would be skipped and the new plan applied to the old body." >&2
      echo "  Re-run with FORCE=1, or point QUANT_DIR at a fresh directory." >&2
      exit 1
    fi
  fi
  if [[ "$FINAL_DONE" == "1" ]]; then
    stored="$(cat "$FINAL_DIR/.quant-plan.sha256" 2>/dev/null || true)"
    if [[ "$stored" != "$PLAN_RESOLVED_HASH" ]]; then
      echo "ERROR: $FINAL_DIR was built with a different plan (or predates plans)." >&2
      echo "  Re-run with FORCE=1 to rebuild." >&2
      exit 1
    fi
  fi
fi

SKIP_WEIGHTS=0
if [[ "$FINAL_DONE" == "1" ]]; then
  if [[ "$KV_CALIB" == "1" ]] && ! kv_scales_done "$FINAL_DIR"; then
    # Weights are done but the KV scales are not: run the calibration stage only.
    SKIP_WEIGHTS=1
    banner "Weights complete; resuming KV calibration only"
  else
    banner "Output already complete -- nothing to do"
    echo "  $FINAL_DIR"
    echo "Re-run with FORCE=1 to rebuild from scratch."
    maybe_cleanup
    exit 0
  fi
fi

if [[ "$SKIP_WEIGHTS" == "0" ]]; then

# A non-empty QUARK_DIR without a valid config is a dead partial pass; refuse it
# rather than silently reusing or overwriting tens of GiB.
if [[ "$QUARK_DONE" == "0" && -e "$QUANT_DIR" && -n "$(ls -A "$QUANT_DIR" 2>/dev/null)" ]] && ! resumable_awq "$QUANT_DIR"; then
  echo "ERROR: $QUANT_DIR exists but is not a complete Quark output." >&2
  echo "  Quark cannot resume a partial pass; move it aside or run with FORCE=1." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Quark AWQ -> MXFP4
# -----------------------------------------------------------------------------
if [[ "$QUARK_DONE" == "1" ]]; then
  banner "Resuming: Quark AWQ output already present"
  echo "  skipping the quantization pass: $QUANT_DIR"
else
  banner "AMD-reference Qwen3.8 AWQ MXFP4 quantization"
  if resumable_awq "$QUANT_DIR"; then
    echo "  resuming AWQ: completed layers in $QUANT_DIR/awq_cache.pt will be replayed"
    echo "  (clear with FORCE=1 or QUARK_AWQ_NO_RESUME=1)"
  fi

  # KV-cache calibration flags (see the KV_CACHE_SCHEME notes above).
  QUARK_KV_ARGS=()
  if [[ -n "$KV_CACHE_SCHEME" ]]; then
    QUARK_KV_ARGS+=(--kv_cache_dtype "$KV_CACHE_SCHEME" --min_kv_scale "$MIN_KV_SCALE")
    if [[ "$KV_CACHE_POST_ROPE" == "1" ]]; then
      QUARK_KV_ARGS+=(--kv_cache_post_rope)
    fi
    echo "  KV cache calibration: scheme=$KV_CACHE_SCHEME post_rope=$KV_CACHE_POST_ROPE min_kv_scale=$MIN_KV_SCALE"
  else
    echo "  KV cache calibration: disabled (export will carry no k/v scales; vLLM serves fp8 KV at 1.0)"
  fi

  # Algorithm(s) + recipe file(s). `algo=none` runs plain RTN MXFP4 with no
  # algorithm at all; passing --quant_algo none would be rejected by Quark, so the
  # flag is omitted entirely. QUARK_ALGO may be a comma list ("gptq,qronos"); the
  # recipe group name must match each algorithm. Non-awq algorithms have no
  # built-in config for qwen3_5, so a plan must supply an algo_config_file entry
  # for each (see algo-configs/).
  QUARK_ALGO_ARGS=()
  if [[ "$QUARK_ALGO" != "none" ]]; then
    QUARK_ALGO_ARGS=(--quant_algo "$QUARK_ALGO")
    declare -A ALGO_FILE=()
    if [[ -n "${PLAN_ALGO_CONFIG_FILES:-}" && "${PLAN_ALGO_CONFIG_FILES}" != "[]" ]]; then
      while IFS=$'\t' read -r _a _f; do
        [[ -n "$_a" ]] && ALGO_FILE["$_a"]="$_f"
      done < <("$PY" -c 'import json,os; [print(a+"\t"+f) for a,f in json.loads(os.environ["PLAN_ALGO_CONFIG_FILES"])]')
    fi
    for _algo in ${QUARK_ALGO//,/ }; do
      _cfg="${ALGO_FILE[$_algo]:-}"
      if [[ -z "$_cfg" && "$_algo" == "awq" ]]; then
        _cfg="$AMD_AWQ_CONFIG"
      fi
      if [[ -z "$_cfg" ]]; then
        echo "ERROR: algo=$_algo requires an algo_config_file entry (see algo-configs/)" >&2
        exit 1
      fi
      if [[ ! -f "$_cfg" && -f "$SCRIPT_DIR/$_cfg" ]]; then
        _cfg="$SCRIPT_DIR/$_cfg"
      fi
      if [[ ! -f "$_cfg" ]]; then
        echo "ERROR: algo_config_file not found for $_algo: $_cfg" >&2
        exit 1
      fi
      QUARK_ALGO_ARGS+=(--quant_algo_config_file "$_algo" "$_cfg")
    done
  fi

  # Only pass --exclude_layers when a plan supplied the list: the flag replaces
  # the template default rather than adding to it, so omitting it preserves the
  # historical vision/lm_head/mtp exclusion.
  QUARK_EXCLUDE_ARGS=()
  if [[ ${#QUARK_EXCLUDE[@]} -gt 0 ]]; then
    QUARK_EXCLUDE_ARGS=(--exclude_layers "${QUARK_EXCLUDE[@]}")
  fi

  # Attention quantization is only wired when a plan requests it.
  QUARK_ATTN_ARGS=()
  if [[ -n "${PLAN_ATTENTION:-}" ]]; then
    QUARK_ATTN_ARGS=(--attention_dtype "$PLAN_ATTENTION")
    echo "  attention dtype: $PLAN_ATTENTION"
  fi

  # Per-name scheme overrides from the plan -> repeated --layer_quant_scheme
  # PATTERN SCHEME. The compiler already rejected fused-sibling mismatches.
  QUARK_LAYER_SCHEME_ARGS=()
  if [[ -n "${PLAN_LAYER_SCHEMES:-}" && "${PLAN_LAYER_SCHEMES}" != "[]" ]]; then
    while IFS=$'\t' read -r _pat _sch; do
      [[ -n "$_pat" ]] && QUARK_LAYER_SCHEME_ARGS+=(--layer_quant_scheme "$_pat" "$_sch")
    done < <("$PY" -c 'import json,os; [print(p+"\t"+s) for p,s in json.loads(os.environ["PLAN_LAYER_SCHEMES"])]')
    echo "  layer scheme overrides: $(( ${#QUARK_LAYER_SCHEME_ARGS[@]} / 2 ))"
  fi

  "$PY" "$WRAPPER" \
    --model_dir "$MODEL_ID" \
    --output_dir "$QUANT_DIR" \
    --quant_scheme "$QUARK_SCHEME" \
    --num_calib_data "$NUM_CALIB_DATA" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$QUARK_CALIB_BATCH" \
    "${QUARK_ALGO_ARGS[@]}" \
    --model_export hf_format \
    --data_type auto \
    --device cuda \
    --multi_gpu \
    --multi_device \
    --trust_remote_code \
    --skip_evaluation \
    ${QUARK_ATTN_ARGS[@]+"${QUARK_ATTN_ARGS[@]}"} \
    ${QUARK_LAYER_SCHEME_ARGS[@]+"${QUARK_LAYER_SCHEME_ARGS[@]}"} \
    ${QUARK_EXCLUDE_ARGS[@]+"${QUARK_EXCLUDE_ARGS[@]}"} \
    ${QUARK_KV_ARGS[@]+"${QUARK_KV_ARGS[@]}"}
fi

# Bind the completed Quark body to the plan that produced it, so a later run with
# a different plan refuses to reuse it.
if [[ -n "$QUANT_PLAN" && "$QUARK_DONE" == "0" ]]; then
  printf '%s\n' "$PLAN_RESOLVED_HASH" > "$QUANT_DIR/.quant-plan.sha256"
fi

# -----------------------------------------------------------------------------
# Validate the Quark output before touching MTP.
# -----------------------------------------------------------------------------
banner "Validating Quark output"
validate_checkpoint "$QUANT_DIR" quark

# -----------------------------------------------------------------------------
# Convert MTP weights to FP8 for the Radiance serving path.
# -----------------------------------------------------------------------------
banner "Converting MTP to FP8 for Radiance"
FP8_MTP="$SCRIPT_DIR/fp8_mtp.py"
if [[ ! -f "$FP8_MTP" ]]; then
  echo "ERROR: expected $FP8_MTP" >&2
  exit 1
fi

"$PY" "$FP8_MTP" "$QUANT_DIR" "$FINAL_DIR"

if [[ -n "$QUANT_PLAN" ]]; then
  printf '%s\n' "$PLAN_RESOLVED_HASH" > "$FINAL_DIR/.quant-plan.sha256"
fi

# -----------------------------------------------------------------------------
# Final validation
# -----------------------------------------------------------------------------
banner "Final validation"
validate_checkpoint "$FINAL_DIR" final

fi  # SKIP_WEIGHTS

# -----------------------------------------------------------------------------
# FP8 KV-cache calibration: one run produces weights + calibrated KV scales.
# Serving stays a separate step.
# -----------------------------------------------------------------------------
if [[ "$KV_CALIB" == "1" ]]; then
  run_kv_calibration "$FINAL_DIR"
else
  echo "  KV calibration skipped (KV_CALIB=0); fp8 KV will serve at scale 1.0"
fi

maybe_cleanup

if [[ "$KV_CALIB" == "1" ]]; then
  echo "KV scales: calibrated (model-kvscales.safetensors, merged into the index)"
else
  echo "KV scales: NOT calibrated (KV_CALIB=0) -- fp8 KV will serve at scale 1.0"
fi

cat <<EOF_DONE

============================================================
 DONE
============================================================

Quantized model:
  $QUANT_DIR

Radiance-ready model:
  $FINAL_DIR

Serve with:
  cd "$SCRIPT_DIR"
  SNAP="$FINAL_DIR" TP=2 SPEC_METHOD=mtp ./serve-mxfp4.sh

(serve-mxfp4.sh defaults its SNAP to a different directory; pass SNAP explicitly
as above, or point MODELS at a tree where the default name resolves.)

Stage resume: re-running this script keeps a completed Quark pass and only
rebuilds the fp8 rewrite. FORCE=1 rebuilds both from scratch.

AWQ record/replay: layers whose scales+clips are already recorded in
  $QUANT_DIR/awq_cache.pt
are replayed verbatim and their 20-step search is skipped, even within a single
otherwise-fresh run. The checkpoint is keyed by a hash of every input that could
change a result (source, recipe, calibration counts, batch, env, split) and the
run aborts with a diff on mismatch; QUARK_AWQ_CACHE_FORCE=1 overrides,
QUARK_AWQ_CACHE=0 disables. Per-group search loss is in
  $QUANT_DIR/awq_report.csv

Invariance gate (before trusting any future cross-GPU search parallelism):
  python awq_optimize.py compare <cacheA.pt> <cacheB.pt>
Run the same model with two different QUARK_LAYER_SPLIT values and compare: a
bit-identical result is the precondition for parallelizing the search across
GPUs. Until that gate is green on your model, the pass stays layer-serial.
EOF_DONE

if [[ "$CLEAN" == "1" ]]; then
  printf '\n(--clean) intermediates removed; the served model listed above is kept.\n'
fi