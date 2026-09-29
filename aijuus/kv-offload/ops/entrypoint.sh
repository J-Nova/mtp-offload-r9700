#!/bin/bash
# Shared vLLM entrypoint for both vllm-0 and vllm-1 services.
# Parameterized by SERVE_PORT (8000 for vllm-0, 8001 for vllm-1).
# All other logic is identical across instances; SERVE_RANK distinguishes them.
set -e

# ---- model selection (hot-swap; see model-registry.json) ----
# The registry (repo root -> /patches) maps each servable model to its
# path / drafter / spec / calibrated KV pin. The desired model comes from
# /model-state/state.json (written by model-controller) so a swap restart
# boots the NEW model; a fresh boot falls back to MODEL_DEFAULT, then the
# first registry entry. Everything below (KV pin, capture ladder, spec cfg)
# then resolves from these values unchanged.
RID=${SERVE_RANK:-0}
INST="vllm-$RID"
[ -f /patches/aijuus/model-registry.json ] || { echo "[run] FATAL: /patches/aijuus/model-registry.json missing (repo bind not mounted?)" >&2; exit 1; }
eval "$(python3 - "$INST" <<'PY'
import json, os, sys
inst = sys.argv[1]
full = json.load(open("/patches/aijuus/model-registry.json"))
reg = {k: v for k, v in full.items() if not k.startswith("_")}
dflt = full.get("_defaults", {})
name = os.environ.get("MODEL_DEFAULT")
try:
    st = json.load(open("/model-state/state.json"))
    e = st.get(inst)
    if e and e.get("model") in reg:
        name = e["model"]
except Exception:
    pass
if not name or name not in reg:
    name = sorted(reg)[0]
e = reg[name]
def q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"
def pick(field, default=None):
    return e.get(field, dflt.get(field, default))
print("MODEL_NAME=" + q(name))
print("VLLM_MODEL_PATH=" + q(e["path"]))
print("VLLM_SERVED_MODEL_NAME=" + q(e.get("served_name", name)))
print("SPEC_DRAFTER=" + q(e.get("drafter", "")))
print("SPEC_METHOD=" + q(e.get("spec_method", "none")))
print("SPEC_TOKENS=" + q(e.get("spec_tokens", "")))
print("KV_CACHE_MEMORY=" + q(e.get("kv_cache_memory", "")))
print("VERIFY_HEAD=" + q(e.get("verify_head", "")))
print("MAX_MODEL_LEN=" + q(pick("max_model_len", "")))
print("KV_CACHE_DTYPE=" + q(pick("kv_cache_dtype", "fp8")))
print("CHAT_TEMPLATE=" + q(pick("chat_template", "")))
print("TOOL_CALL_PARSER=" + q(pick("tool_call_parser", "")))
print("REASONING_PARSER=" + q(pick("reasoning_parser", "")))
gc = pick("generation_config", None)
print("GEN_CFG=" + q(json.dumps(gc) if gc else "{}"))
env = dict(dflt.get("server_env", {}))
env.update(e.get("server_env", {}))
for k in sorted(env):
    print("export " + k + "=" + q(env[k]))
PY
)"
echo "[run] model selected: $MODEL_NAME path=$VLLM_MODEL_PATH spec=$SPEC_METHOD/$SPEC_TOKENS kv=$KV_CACHE_MEMORY vhead=${VERIFY_HEAD:-auto}"

# ---- parametrized serving shape (env vars, defaults = measured TP=1) ----
SEQS=${MAX_NUM_SEQS:-8}
MLEN=${MAX_MODEL_LEN:-160000}
CHUNK=${MAX_NUM_BATCHED_TOKENS:-4096}
SMETHOD=${SPEC_METHOD:-dflash}
case "$SMETHOD" in dflash|mtp|none) ;; *) echo "SPEC_METHOD must be dflash, mtp or none, got: $SMETHOD" >&2; exit 1 ;; esac

# VERIFY_HEAD is a PER-MODEL setting from model-registry.json (`verify_head`):
# the int2 verify-head GEMM is +2.9% decode under dflash and neutral under mtp.
# If a model entry omits it, fall back to the method default.
if [ -z "${VERIFY_HEAD:-}" ]; then
  if [ "$SMETHOD" = dflash ]; then VERIFY_HEAD=1; else VERIFY_HEAD=0; fi
fi
export RADIANCE_VERIFY_HEAD="$VERIFY_HEAD"

if [ "$SMETHOD" = none ]; then
  SPEC=0
elif [ -z "${SPEC_TOKENS:-}" ]; then
  if [ "$SMETHOD" = dflash ]; then SPEC=16; else SPEC=4; fi
else
  SPEC="$SPEC_TOKENS"
fi

# KV sizing. The calibrated pin comes from model-registry.json
# (kv_cache_memory, emitted by the model-selection eval above). It is
# REQUIRED whenever maxlen is 160000, where vLLM's own profiling finds
# only ~4.81 GiB, below the ~5.71 GiB a single 160k request needs.
# A registry entry with no kv_cache_memory (or 0) forces profiling.
if [ -n "${KV_CACHE_MEMORY:-auto}" ] && [ "${KV_CACHE_MEMORY:-auto}" != auto ] && [ "${KV_CACHE_MEMORY:-auto}" != 0 ]; then
  KMEM="$KV_CACHE_MEMORY"
  KV_SRC=explicit
else
  KV_SRC=profiled
  KMEM=""
fi

# Prefill all-reduce gate from the chunk (CHUNK*5120*2/1024+4096) so a
# non-default chunk cannot silently drop prefill onto RCCL.
# Offload tuning knobs, resolved once. Used by both the transfer-config JSON
# and the shape-label metric the Grafana "Offload tuning A/B" row reads, so a
# policy/thread/async change shows up as a label flip on the next redeploy.
OPOLICY=${KV_OFFLOAD_EVICTION_POLICY:-lru}
ASCHED=${VLLM_ASYNC_SCHEDULING:-0}
RTHREADS=${KV_OFFLOAD_READ_THREADS:-32}
WTHREADS=${KV_OFFLOAD_WRITE_THREADS:-16}
FANOUT=${RADIANCE_FS_FANOUT_MAX:-32}
RID=${SERVE_RANK:-0}
OVMODE=off
if [ -n "$KV_OFFLOAD_GIB" ] && [ "$KV_OFFLOAD_GIB" != 0 ]; then OVMODE=on; fi
MAMBA_MODE=${MAMBA_CACHE_MODE:-align}

# ---- Async-scheduling preflight (see PERFORMANCE.md "Async scheduling") -----------------
# vLLM HARD-gates --async-scheduling (config/vllm.py:1081+): the spec method must be an
# EAGLE-family type (mtp/dflash qualify), the executor must support it (uniproc/mp do),
# and disable_padded_drafter_batch must be FALSE -- so async and the UNPADDED drafter are
# ONE switch. Derive UNPAD from ASCHED (mirrors serve-mxfp4.sh) so the flag cannot boot
# into vLLM's hard error. We also refuse the combinations this patched stack has NOT
# validated, so flipping the flag can never silently corrupt hybrid/spec/offload state.
if [ "$ASCHED" = 1 ]; then
  UNPAD=false
  # Hybrid/Mamba spec-decode + async only syncs num_accepted_tokens in align mode
  # (gpu_model_runner.py skips the CPU sync for async non-align). We only run align.
  if [ "$MAMBA_MODE" != align ]; then
    echo "[run] FATAL: VLLM_ASYNC_SCHEDULING=1 requires mamba cache mode align" >&2
    exit 1
  fi
  # async + KV offload is unvalidated here (the offload/eagle patches assume sync
  # scheduler semantics). Refuse unless explicitly forced for a test.
  if [ "$OVMODE" = on ] && [ "${RADIANCE_ASYNC_ALLOW_OFFLOAD:-0}" != 1 ]; then
    echo "[run] FATAL: async scheduling + KV offload is unvalidated (offload patches assume sync)." >&2
    echo "      Unset KV_OFFLOAD_GIB, or set RADIANCE_ASYNC_ALLOW_OFFLOAD=1 to force a test." >&2
    exit 1
  fi
  echo "[run] async scheduling ENABLED (padded drafter; async_allow_offload=${RADIANCE_ASYNC_ALLOW_OFFLOAD:-0})"
else
  UNPAD=true
fi

AR_MAX_KB=$(( CHUNK * 5120 * 2 / 1024 + 4096 ))

# Cudagraph capture ceiling = SEQS*(SPEC+1), stock ladder trimmed to it
# (mirrors serve-mxfp4.sh: SEQS=8 SPEC=16 -> 136, so the ladder runs to 256).
CAP=$(( SEQS * (SPEC + 1) ))
SIZES=""
for s in 1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 \
          80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256; do
  [ "$s" -le "$CAP" ] && SIZES="${SIZES:+$SIZES,}$s"
done
CAPLIST="[$SIZES]"

# Speculative config. mtp needs no drafter checkpoint; dflash does; none disables it.
if [ "$SMETHOD" = none ]; then
  SPEC_CFG=""
elif [ "$SMETHOD" = mtp ]; then
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"R4D\",\"disable_padded_drafter_batch\":$UNPAD}"
else
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$SPEC_DRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":$UNPAD,\"draft_sample_method\":\"greedy\"}"
fi

# Expose the resolved shape for logs + Prometheus labels.
export RADIANCE_AR_MAX_KB="$AR_MAX_KB"
export SERVE_KV_SOURCE="$KV_SRC"
export SERVE_SPEC_RESOLVED="$SPEC"
echo "[run] TP=1 shape: seqs=$SEQS maxlen=$MLEN chunk=$CHUNK kv=${KMEM:-none}($KV_SRC) spec=$SMETHOD/$SPEC ar_max_kb=$AR_MAX_KB captures=$CAPLIST"

SP=/opt/vllm/lib/python3.12/site-packages
cd /patches

# ---- runtime overlay: Radiance/AIJUUS source + configs come from the repo bind, NOT the image ----
# The release image bakes only infrastructure (ROCm, the torch/triton/vLLM venv, the patched vLLM
# source, r4d.so, radiance_mxfp4_fp8.so). Everything under this banner is copied into
# site-packages here so any Radiance/AIJUUS source file can be iterated without a rebuild.
# This MUST run before any python below: radiance_amdsmi.pth has to initialise amdsmi ahead of HIP
# at interpreter startup, and radiance_kernels is imported by vLLM's plugin loader
# (vllm/plugins/__init__.py -> radiance_kernels.install_all()). The fork-local hooks there
# (_install_token_collector) is env-gated, so a plain run is unchanged.
mkdir -p "$SP"/aijuus "$SP"/vllm/model_executor/kernels \
         "$SP"/vllm/model_executor/layers/quantization/utils/configs \
         "$SP"/vllm/model_executor/layers/fused_moe/configs \
         "$SP"/aiter/ops/triton/configs/gemm
cp radiance_*.py "$SP"/
cp radiance_amdsmi.pth "$SP"/
# Optional AIJUUS overlay modules: warn (do not abort the boot under `set -e`) when a file is
# absent, so a deploy from a tree that lacks an optional module still starts.
for f in aijuus/__init__.py aijuus/collect_tokens.py; do
  if [ -f "$f" ]; then cp "$f" "$SP"/aijuus/; else echo "[run] WARN: overlay file missing: $f"; fi
done
cp radiance_preamble.py /opt/radiance_preamble.py
cp fp8-configs/* "$SP"/vllm/model_executor/layers/quantization/utils/configs/ 2>/dev/null || true
cp moe-configs/* "$SP"/vllm/model_executor/layers/fused_moe/configs/ 2>/dev/null || true
cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/ 2>/dev/null || true
# TunableOp table for the skinny fp8 GEMMs (MTP drafter N=5120 + lm_head), from the R9700
# branch (same torch 2.11.0 / HIP 714 / gfx1201 validators). Read-only; PYTORCH_TUNABLEOP_*
# comes from the registry. A validator mismatch makes torch ignore it (fails safe).
mkdir -p /cache/tunableop 2>/dev/null || true
if [ -f aijuus/refs/r9700-tp1/tunableop-skinny0.csv ]; then
  cp -f aijuus/refs/r9700-tp1/tunableop-skinny0.csv /cache/tunableop/skinny0.csv
else
  echo "[run] WARN: tunableop table aijuus/refs/r9700-tp1/tunableop-skinny0.csv missing"
fi
echo "[run] runtime overlay installed from /patches (radiance_*.py + aijuus/ + configs)"

python3 patch_quark_mxfp4.py
python3 patch_nvfp4_mxfp4.py
python3 patch_tp3_pad.py
python3 patch_ar_maxbytes.py
python3 patch_aot_envkey.py || echo "[run] WARN: patch_aot_envkey.py missing/failed; AOT env-key gate not applied"
python3 patch_topk_triton_rows.py
python3 patch_dflash_calib.py
python3 patch_dflash_mxfp4_kv.py
python3 patch_rmsquant_fusion.py
python3 patch_verify_head.py
python3 patch_kv_group_size.py
python3 patch_topk_composite.py
python3 patch_gdn_shared_build.py
python3 patch_dflash_selector_topk.py
python3 patch_gdn_merge_inproj.py
python3 patch_dynwidth.py
python3 patch_async_dynwidth.py
python3 patch_step_trace.py
python3 patch_ar_geometry.py
python3 patch_ar_qbits.py
python3 patch_ar_3rank.py
python3 patch_gdn_glue.py
if [ "${RADIANCE_GDN_LAZY:-0}" = 1 ]; then python3 patch_gdn_lazy.py; fi
python3 patch_qwen3_thinkoff.py || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"

# ---- KV offload patches (mirror serve-mxfp4.sh). All inert unless KV_OFFLOAD_GIB
# turns the OffloadingConnector on; they are what make the CPU/fs tiers correct.
bash aijuus/kv-offload/ops/apply-kv-patches.sh

# (radiance modules, aijuus/, and the configs were already overlaid near the top, before the
#  patch scripts, so the amdsmi .pth is active for every python process. Nothing to copy here.)

hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $([ "${MXFP4_CUMODE:-0}" = 1 ] && echo -mcumode) $(python3 -m pybind11 --includes) radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
  cp /r4d/r4d.so "$SP"/r4d.so
  echo "[radiance] using patched r4d.so from $R4D_SO"
fi

# Shape labels for Prometheus (node-exporter --collector.textfile
# scrapes the shared serve-shape volume).
rm -f /tmp/serve-shape/serve_shape.prom
printf 'radiance_serve_info{instance="%s",max_num_seqs="%s",max_model_len="%s",max_num_batched_tokens="%s",kv_source="%s",spec_method="%s",spec_tokens="%s",offload_policy="%s",offload_mode="%s",async_sched="%s",read_threads="%s",write_threads="%s",fanout_max="%s"} 1\n' \
  "$RID" "$SEQS" "$MLEN" "$CHUNK" "$KV_SRC" "$SMETHOD" "$SPEC" "$OPOLICY" "$OVMODE" "$ASCHED" "$RTHREADS" "$WTHREADS" "$FANOUT" > /tmp/serve-shape/serve_shape_$RID.prom
echo "[run] shape labels: kv=$KV_SRC spec=$SMETHOD/$SPEC"

# ---- stale CPU-KV-offload region sweep (THIS instance only) ----
# vLLM names its CPU offload tier /dev/shm/vllm_offload_<engine_id>.mmap
# and only unlinks it in cleanup(), so a crashed/killed start leaks one
# (it used a fresh uuid each boot). Enough leaks fill the 32G host tmpfs
# and the next start then dies in MADV_POPULATE_WRITE with EFAULT
# (Errno 14) -- the 2026-09-28 outage. engine_id is pinned per instance
# below (radrank$SERVE_RANK), so our own stale region has a known name and is
# removed here BEFORE the engine re-creates it. Only our own name is
# matched: ipc: host shares /dev/shm with the peer instance, whose live
# region must never be touched.
# Hardening: engine_id names the shared /dev/shm region, so it MUST
# be distinct per instance. Fail fast if SERVE_RANK is unset/empty --
# a blank rank would alias the peer (both use 0) and make each
# instance sweep the other's live region. A uuid is un-sweepable and
# a hostname changes on container recreate (would orphan the sweep),
# so the stable, unique per-instance rank is the right token.
: "${SERVE_RANK:?SERVE_RANK must be set to a distinct value per instance (0/1)}"
ENG_ID="radrank$SERVE_RANK"
bash aijuus/kv-offload/ops/clean-stale-ram-tier.sh "$ENG_ID"
cd /

# Final argv assembled here from the resolved shape (no static command:
# compose cannot do arithmetic or conditionals, the entrypoint can).
KV_ARG=""
[ -n "$KMEM" ] && KV_ARG="--kv-cache-memory $KMEM"

# KV offload (env-gated, default off). --kv-offloading-size sets cpu_bytes_to_use AND
# selects OffloadingConnector; the JSON only adds extra_config.
# Head cap (patch_offload_head_cap.py) is what distinguishes the two modes:
#   Mode A (CPU only, no KV_OFFLOAD_DISK_DIR): max_offload_tokens="auto" -- fit the CPU
#     tier exactly, so over-subscription restores a contiguous partial prefix, never nothing.
#   Mode B (CPU + shared fs, KV_OFFLOAD_DISK_DIR set): NO cap -- the disk holds the
#     overflow, and a per-request cap would defeat it. Reaper REQUIRED.
# KV_OFFLOAD_HEAD_CAP overrides the topology default: "auto" | <N> | "off".
# No spaces in the JSON, so word-splitting the unquoted OFF_ARG is safe.
OFF_ARG=""
if [ -n "$KV_OFFLOAD_GIB" ] && [ "$KV_OFFLOAD_GIB" != 0 ]; then
  HC="${KV_OFFLOAD_HEAD_CAP:-}"
  if [ -z "$HC" ]; then
    if [ -n "$KV_OFFLOAD_DISK_DIR" ]; then HC=off; else HC=auto; fi
  fi
  EC="\"offload_prompt_only\":true"
  if [ "$HC" != off ] && [ "$HC" != 0 ]; then
    EC="\"max_offload_tokens\":\"$HC\",$EC"
  fi
  if [ -n "$KV_OFFLOAD_DISK_DIR" ]; then
    EC="$EC,\"spec_name\":\"TieringOffloadingSpec\",\"eviction_policy\":\"$OPOLICY\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_OFFLOAD_DISK_DIR\",\"n_read_threads\":$RTHREADS,\"n_write_threads\":$WTHREADS}]"
    export RADIANCE_LOOKUP_INVALIDATE=1 RADIANCE_FS_FAILED_LOAD_FORGET=1
    echo "[run] fs KV tier ON: root_dir=$KV_OFFLOAD_DISK_DIR (host-side reaper REQUIRED)"
  fi
  OFF_ARG="--kv-offloading-size $KV_OFFLOAD_GIB --kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"engine_id\":\"$ENG_ID\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$EC}}"
  echo "[run] KV offload ON: cpu tier=${KV_OFFLOAD_GIB}GiB, head cap=$HC"
fi

ASYNC_ARG="--no-async-scheduling"
if [ "$ASCHED" = 1 ]; then ASYNC_ARG="--async-scheduling"; fi
SPEC_ARG=""
if [ -n "$SPEC_CFG" ]; then SPEC_ARG="--speculative-config $SPEC_CFG"; fi
TOOL_ARG=""
if [ -n "$TOOL_CALL_PARSER" ]; then TOOL_ARG="--enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER"; fi
if [ -n "$REASONING_PARSER" ]; then TOOL_ARG="$TOOL_ARG --reasoning-parser $REASONING_PARSER"; fi

# shellcheck disable=SC2086
exec /opt/radiance_entrypoint.sh \
  "$VLLM_MODEL_PATH" --served-model-name "$VLLM_SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$SERVE_PORT" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.98 $KV_ARG $OFF_ARG \
  --max-model-len "$MLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$CHUNK" \
  --attention-backend R4D \
  $SPEC_ARG \
  $ASYNC_ARG \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16 \
  --compilation-config "{\"cudagraph_capture_sizes\":$CAPLIST,\"pass_config\":{\"fuse_norm_quant\":true,\"fuse_act_quant\":true}}" \
  --enable-prefix-caching --mamba-cache-mode "$MAMBA_MODE" \
  $TOOL_ARG \
  --override-generation-config "$GEN_CFG" \
  --api-key "$VLLM_API_KEY" \
  --chat-template "$CHAT_TEMPLATE"
