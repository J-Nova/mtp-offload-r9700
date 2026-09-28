# Quantization plan

A single declarative file that drives the Blend conversion's layer selection, and
can clone another checkpoint's selection. This documents what is implemented now
(plan + reference import) and the full design (per-layer / per-type schemes and
the hard boundaries that need upstream patches).

## Status

- **Implemented:** `quant_plan.py` (resolve / reference import / validate /
  effective-selection report), `quant-plan.standard.json` and
  `quant-plan.qronos.json` examples, converter wiring
  (`convert-blend-mxfp4.sh`), plan-driven MTP rewrite (`fp8_mtp.py`), plan hash
  in the AWQ resume key (`awq_optimize.py`), a Standard-matching default exclude
  (`quant_defaults.py`), algorithm plumbing for `qronos`/`gptq`/… (including
  comma-lists like `gptq,qronos`) with `algo-configs/` examples, and per-name
  scheme selection (`layers`) with the fused-sibling guard.
- **Documented here, not implemented:** per-layer / per-type scheme selection
  and the capability matrix (the "full plan", A+B+C). The hard boundaries in
  section 7 explain why each needs a patch or cannot be done.

## Quick start

```bash
# Validate a plan and see exactly what it resolves to (writes a .resolved.json
# and a shell-sourceable .sh next to it):
python3 quant_plan.py resolve --plan quant-plan.standard.json

# Show what a checkpoint actually loads, per layer (reimplements vLLM's matcher):
python3 quant_plan.py report --model /home/juup/models/Qwen3.8-27B-MXFP4-mtpfp8 --verbose

# Run the conversion with a plan (output dirs etc. as usual):
QUANT_PLAN=$PWD/quant-plan.standard.json ./convert-blend-mxfp4.sh
```

With no `QUANT_PLAN` the converter defaults now match AMD's Standard checkpoint:
`mxfp4` + `awq`, body excludes vision/`lm_head`, and the final config carries the
full weight-scoped `mtp.*` set (so the exclude is the Standard's 127 entries).
MTP is excluded from the body pass (stays bf16) and rewritten to fp8.

Algorithms: `awq` (default) and `none` work out of the box; `qronos`, `gptq`,
`gptaq`, `smoothquant`, `autosmoothquant` and `rotation` are supported by Quark
but have no built-in config for `qwen3_5`, so a plan must point at an algo config
file (see `algo-configs/`). See "Algorithms" below.

## Plan schema

```json
{
  "reference_model": "/path/to/another/checkpoint",
  "global": {"scheme": "mxfp4", "algo": "qronos", "algo_config_file": "algo-configs/qronos-qwen38.json"},
  "exclude": {"mode": "append", "entries": []},
  "allow_missing_vision_head": false,
  "mtp": {"mode": "fp8", "module_list": null, "keep_in_exclude": true},
  "kv": {"scheme": "fp8", "post_rope": true, "min_scale": 1e-3},
  "attention": null,
  "layers": [],
  "layer_type": {"Linear": "mxfp4"},
  "version": "0.13+unknown",
  "retain_algo_config": false
}
```

| Field | Meaning |
|---|---|
| `reference_model` | Import the global scheme (reconstructed from its serialized weight dtype), the `exclude` list, and the `version` tag. An ambiguous serialized dtype is rejected unless `global.scheme` is set. |
| `global.scheme` | Quark global weight scheme (overrides the reference). |
| `global.algo` | One name, a comma string, or a list (Quark applies them in order). `awq` (default), `none`, or `qronos`/`gptq`/`gptaq`/`smoothquant`/`autosmoothquant`/`rotation`. `none` omits the algorithm. Non-awq needs a config file entry. |
| `global.algo_config_file` | A path (single algo) or an `{algo: path}` mapping (multi-algo). Required for non-awq. Relative paths resolve against the repo. |
| `exclude.mode` | `append` (to the built-in vision/`lm_head`/`mtp.*`), `reference` (use the reference's exact list), or `replace`. |
| `exclude.entries` | Extra patterns/names to add. |
| `allow_missing_vision_head` | Required `true` to let `exclude.mode=replace` drop the `model.visual.*`/`lm_head` guards; otherwise the plan is rejected. |
| `mtp.mode` | `fp8` (exclude from body pass, then rewrite to fp8) or `bf16` (exclude and do not rewrite). |
| `mtp.module_list` | Override the eight `mtp.*` modules (supports more than one MTP layer); the validator uses the same list. |
| `mtp.keep_in_exclude` | Keep the reference's exact (weight-scoped, load-inert) MTP entries in the final `exclude`, for byte parity. Default `true`. |
| `kv.scheme` / `post_rope` / `min_scale` | KV calibration; `fp8` only (vLLM's hard limit). |
| `attention` | `null` or `fp8`; passed to Quark as `--attention_dtype`. |
| `layers` | Per-name scheme overrides, `[{"pattern": "*down_proj", "scheme": "fp8"}]`, passed to Quark as repeated `--layer_quant_scheme`. Fused siblings (q/k/v, gate/up) with different schemes are rejected. Exclude entries the patterns cover are dropped (runtime-ignore rewrite). |
| `layer_type` | Per-module-type overrides, `{"Linear": "fp8"}`. Quark does not forward `layer_type_config` from the CLI, so the wrapper applies it via `LLMTemplate._set_layer_type_config`; only `Linear`/`Conv2d`/`ConvTranspose2d` round-trip through Quark's export. |
| `version` | Override the `config.json` `quantization.version` string. |
| `retain_algo_config` | Keep Quark's list-of-dicts `algo_config` (default null; retaining can crash vLLM's `WeightsMapper`). |

## Reproducing the Standard checkpoint

`quant-plan.standard.json` imports `Qwen3.8-27B-MXFP4-mtpfp8` and resolves to:

- `scheme=mxfp4`, `version=0.13+unknown`
- quant exclude = the reference's 112 non-MTP entries + a module-level `mtp.*`
  (so Quark leaves MTP bf16 for the fp8 rewrite)
- final exclude = the reference's 112 + the reference's own 15 MTP entries
  (`keep_in_exclude`), i.e. exactly the Standard's 127 entries
- MTP rewritten to fp8

Verified with `quant_plan.py resolve`: `B non-mtp == quant non-mtp: True`,
`final_mtp_entries (15) == B mtp entries: True`, `version == B: True`.

### Algorithms

Quark's algorithm names (`get_supported_algorithm_types()`) are
`awq`, `gptq`, `gptaq`, `qronos`, `smoothquant`, `autosmoothquant`, `rotation`.
The plan's `global.algo` takes one of those (or `none`), and
`global.algo_config_file` supplies the JSON the Quark loader reads
(`load_quant_algo_config_from_file` → `GPTQConfig.from_dict` /
`QronosConfig.from_dict` / …). Quark has no built-in config for `qwen3_5`, so
non-awq requires the file.

### Qronos + OCP MXFP4 (what you asked for)

**Qronos already contains GPTQ.** Its implementation runs the GPTQ error-diffusion
loop internally (`qronos.py`: "GPTQ loop to calculate Q[:, 1:] using the error
diffused W[:, 1:]", with the `H^-1 = L L^T` Cholesky factor), plus its own
cross-covariance `G` and beta stabilisation. So `"algo": "qronos"` **is**
GPTQ+Qronos — no separate GPTQ pass, and no replacement of Quark. Quark can also
run a comma list sequentially (`algorithm/api.py`: "apply algorithms
sequentially"), so `"algo": "gptq,qronos"` with both files is available if you
specifically want a standalone GPTQ pass first; it costs a second Hessian pass
and is usually redundant.

Qronos explicitly supports `mxfp4` per-group, so it composes with this repo's
MXFP4 serving path. Example plan `quant-plan.qronos.json`:

```bash
python3 quant_plan.py resolve --plan quant-plan.qronos.json
QUANT_PLAN=$PWD/quant-plan.qronos.json ./convert-blend-mxfp4.sh
```

`algo-configs/qronos-qwen38.json` holds the Qronos hyperparameters
(`block_size=128`, `desc_act=true`, `static_groups=true`, `alpha=1e-3`,
`beta=1e4`) and the `model_decoder_layers` / `inside_layer_modules` paths for
this architecture. The list covers all 12 quantized projections per decoder
layer: the seven block projections (`self_attn.{q,k,v,o}_proj`,
`mlp.{gate,up,down}_proj`) plus the five GDN linear-attention projections
(`linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`). Quark
matches each entry as a suffix via `fnmatch("*" + name)`, so the relative names
are correct. Note `in_proj_a`/`in_proj_b` are small (48x2560) gating projections;
drop them from the list if you prefer to leave them on RTN.

### GPTQ

`algo-configs/gptq-qwen38.json` is the analogous GPTQ config. Quark documents
GPTQ for `int4`/`uint4` per-group; release notes mention MXFP4 GPTQ as well, so
`gptq` + `mxfp4` is accepted with a warning and may work. If it does not, use
Qronos, which is the MXFP4-native choice.

### Notes / constraints

- The exported weights are plain OCP MXFP4 either way, so **vLLM needs no
  change**; the algorithm only changes the calibration/scaling of the weights.
  `algo_config` is calibration metadata and is nulled by `fp8_mtp.py` unless
  `retain_algo_config` is set (a list-of-dicts value can crash vLLM's
  `WeightsMapper`).
- The AWQ record/replay accelerator (`awq_optimize.py`) patches the AWQ
  processor only. `gptq`/`qronos` run Quark's code unmodified and are not
  cached; expect a full calibration pass.
- Qronos needs calibration data (Hessians), so it uses the same
  `NUM_CALIB_DATA` / `SEQ_LEN` / `QUARK_CALIB_BATCH` as AWQ.
- Validation rejects combinations Quark does not document (e.g. `qronos` with a
  non-per-group scheme, non-awq without a config file) before the run starts.

## Why Blend and Standard looked different

`quant_plan.py report` on the Standard shows: 497 layers `global:fp4`, 112
excluded, 8 MTP at `fp8_e4m3/per_channel`. The Blend final is identical. The
apparent differences are provenance, not layer treatment:

- `algo_config` is present only in the Blend **intermediate** Quark output;
  `fp8_mtp.py` nulls it in the final, so both finals are null.
- The Standard's extra `mtp.*` excludes are **weight-scoped** (`mtp.fc.weight`,
  …). vLLM matches exclude by exact module prefix or `re:` only
  (`quark/model_executor/layers/quantization/quark/utils.py:97-123`), so
  `mtp.fc` ≠ `mtp.fc.weight` and they are inert. MTP loads via
  `layer_quant_config.mtp.*` in both.

So the layer quantization was already the same; the plan exists to make that
controllable and to let you clone any other checkpoint's selection.

## How the pieces fit

1. `quant_plan.py resolve` merges the plan (and reference) into
   `PLAN_SCHEME`, `PLAN_ALGO`, `PLAN_EXCLUDE`, `PLAN_VERSION`, `PLAN_MTP_*`, and
   writes a `.resolved.json` plus a shell file.
2. `convert-blend-mxfp4.sh` sources the shell file and passes
   `--quant_scheme` / `--quant_algo` / `--attention_dtype` / `--exclude_layers`
   to Quark, and exports `PLAN_*` for the MTP stage.
3. The generated wrapper monkeypatches `LLMTemplate.get_config` to stamp the
   version (there is no Quark CLI for it).
4. `fp8_mtp.py` reads `PLAN_QUANT_PLAN_RESOLVED` and applies the MTP policy,
   `keep_in_exclude`, `retain_algo_config` and version.
5. `awq_optimize.py` mixes the plan and resolved file into the AWQ resume hash,
   and `convert-blend-mxfp4.sh` records the resolved plan's hash in
   `QUANT_DIR`/`FINAL_DIR` and refuses to reuse a stage built with a different
   plan (or before plans existed), so a changed plan cannot silently pair a new
   config with an old body.

The plan is deliberately **not** a second source of truth for anything vLLM
resolves: the compiler emits the same values Quark/vLLM already consume, and
`report` uses the loader's real matching rules rather than a parallel grammar.

## Full plan (A+B+C): per-layer and per-type schemes

Everything below is expressible in Quark/vLLM's data model but is not yet wired.

### Producer side

- **Per-name:** already a Quark CLI flag (`--layer_quant_scheme PATTERN SCHEME`,
  `quantize_quark.py:481`). The compiler would emit `PLAN_LAYER_SCHEMES` and the
  converter would append repeated flags. Matching is fnmatch with `*`, else
  substring (`quark.py:604-611`).
- **Per-type:** implemented in Quark (`LLMTemplate.get_config(layer_type_config=)`,
  `template.py:822/1023`) but `quantize_quark.py` never forwards it, and export
  refuses non-standard types (`quant_config_parser.py:188-189`,
  `config.py:251-258` restricts round-trip to `{Conv2d, Linear, ConvTranspose2d}`).
  Wire by monkeypatching `LLMTemplate.get_config` in the wrapper (the version
  override already does this) and relaxing the export/import type allowlist.
- **Custom schemes:** `LLMTemplate.register_scheme` (`template.py:766-789`) must
  be called in Python before parsing; expose via the wrapper.
- **Shared scales:** `get_config(shared_scale_groups=)` exists (`template.py:825`)
  but is not serialized by `QConfig.to_dict` (`config.py:203-235`), so it is
  calibration-time only.

### Capability matrix (validation)

The compiler should reject a plan before a multi-hour run when it violates the
loader's scheme matrices: Fused shards of `qkv_proj` / `gate_up_proj` must share
a scheme; OCP-MX/MXFP4 group 32 + `e8m0`; NVFP4 group 16; W4A8/W8A8/INT8 exact
dtype/qscheme layouts; KV `fp8_e4m3` per-tensor only. All encoded from the audit
in section 7.

## Hard boundaries

These cannot be selected through config today; the patch column says whether the
full plan can lift them.

### vLLM Quark loader

| Boundary | Where | Patchable? |
|---|---|---|
| Fused shards must use the same scheme (`q/k/v`, `gate/up`) | `quark.py:590-597`, `utils.py:78-84` | No via config; needs un-fusing + kernel work |
| One KV scheme for the whole model; `fp8_e4m3` per-tensor only | `quark.py:230-260`, `776-788` | Loader patch + a real backend |
| Output-tensor / bias quantization forbidden | `quark.py:628-632`, `quark_moe.py:108-113` | No |
| `layer_type_quant_config` keyed by raw `str(type(module))`; round-trip limited | `quark.py:613-618`, `config.py:251-258` | Wrapper authors it; export/import needs a patch |
| `exclude` is exact or `re:` only (no globs) | `utils.py:97-123` | Use `re:`; no patch |
| MoE ignore is all-or-nothing per parent | `utils.py:36-48` | No |
| Scheme matrices fixed (OCP-MX 32/NVFP4 16, W4A8/W8A8/INT8 layouts) | `quark.py:291-549`, `quark_nvfp4.py:43` | No |
| Attention quantization only for DeepSeek-V3 dynamic MXFP4 | `quark.py:54,159-172` | No for qwen |
| Per-layer kernel backend | `kernels_linear/__init__.py:583-594` | `auto` + `VLLM_DISABLED_KERNELS` only |

### Quark producer / export

| Boundary | Where | Patchable? |
|---|---|---|
| No upstream `qwen3_5` template; MoE preprocess raises for it | `model_preparation.py:222-225,279-282` | Already bypassed by the wrapper (template registration + `preprocess_for_quantization` monkeypatch) — keep those load-bearing patches |
| `layer_type_config` not forwarded by CLI; refused by export | `quantize_quark.py:122-131`, `quant_config_parser.py:188-189` | Wrapper + export patch |
| KV spec hardcoded to `FP8E4M3PerTensorSpec` | `template.py:963` | Producer patch; vLLM side still only fp8 per-tensor |
| Attention only `fp8` | `template.py:572` | Producer patch |
| `shared_scale_groups` not serialized | `config.py:203-235` | Producer patch (calibration-only concept) |
| No built-in AWQ config for this arch | `algo_configs.py` AWQ map | Custom recipe (already used) |
| `output_tensors` per-group static activations rejected | `config.py:406-423` | No |

### Kernel / hardware

| Boundary | Where | Patchable? |
|---|---|---|
| MXFP4/OCP-MX group 32 + e8m0 + uint8 packed; NVFP4 group 16; divisibility | `mxfp4.py:205-206`, `quark.py:395,459,521` | No |
| Radiance MXFP4/W4/skinny shape allowlists and gating | `radiance_mxfp4.py:653-656`, `radiance_w4.py:75-87,394`, `radiance_gemm.py:46-49` | Measure new bands |
| W4 only inside the drafter load bracket | `patch_dflash_w4.py:28-41` | Bracket another loader |
| Emulation can silently change the numeric scheme | `marlin.py:27-37`, `humming.py:34-44`, `quark_ocp_mx.py:109-119` | Report the effective backend |

## Roadmap for the full plan

1. **Per-name schemes — implemented.** `layers: [{pattern, scheme}]` compiles into
   repeated `--layer_quant_scheme`, with the fused q/k/v and gate/up guard.
2. **Per-type schemes — implemented.** `layer_type: {"Linear": ...}` is applied in
   the wrapper via `LLMTemplate._set_layer_type_config` (the CLI never forwards
   it); only the module types Quark's export round-trips are accepted.
3. **Shared scales / custom schemes** — wrapper registers schemes and passes
   `shared_scale_groups`; calibration-only for shared scales.
4. **Fused-group guard in `report`** — expand `qkv_proj`/`gate_up_proj` via the
   model's packed mapping and assert shard unanimity before a run.
5. **Boundaries that stay rejected** — the compiler emits a reason and, where one
   exists, a workaround; nothing fails silently.

## Notes

- **Single-command flow.** `convert-blend-mxfp4.sh` now finishes by running the
  FP8 KV-cache calibration in the serving image and merging the scales into the
  checkpoint, so one run yields a serve-ready model (only serving stays separate).
  Toggle with `KV_CALIB=0`, and override `VLLM_IMAGE` / `KV_CALIB_TP` /
  `KV_CALIB_GPU_UTIL` / `KV_CALIB_EXTRA_ARGS`. A re-run calibrates KV only when
  the weights are already complete and the scales are missing.
- **KV scale convention.** The fp8 cache stores `x / scale`, so for a checkpoint
  the scale is `observed_amax * PAD / constant`, and PAD is headroom (the
  reference tool uses 1.10). The constant is NOT the fp8 max: vLLM's
  `calc_kv_scales` divides by `Q_/K_/V_SCALE_CONSTANT` = 200/200/100
  (`vllm/envs.py`), so derived amax is `scale * 200` for K and `scale * 100` for
  V. `verify_kv_scales.py` prints those. The reference fork's standalone
  `kv_calib/` tool captures post-RoPE q/k/v through `calc_kv_scales`, verifies
  each emitted name against the model's own `load_weights`, and handles the MTP
  draft's own cache — the intended replacement for the hand-rolled path here.
- **Format naming.** In the reference image `--format mxfp4` is **MXFP4-16**
  (IQ4-NL grid + FP16 per-16 scale, 5.0 bpw), not this stack's OCP MXFP4. The
  OCP e2m1 + E8M0 group-32 format is `--format mxfp4ocp` (4.25 bpw). Do not
  conflate them when driving the image's quantizer.
- `quant_plan.py report` reimplements `should_ignore_layer` and
  `_find_matched_config` deliberately without importing vLLM, so it runs on the
  host; if the loader's rules change, update it alongside.
- Only `fp8` KV and `mxfp4`/`awq` body are exercised by the current pipeline; the
  other scheme names are accepted by the schema but rejected by the converter
  until the full plan wires them.
