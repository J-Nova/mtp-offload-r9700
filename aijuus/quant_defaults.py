#!/usr/bin/env python3
"""Canonical default exclude lists shared by quant_plan.py (plan resolution) and
fp8_mtp.py (the plan-free path), so both produce the same default checkpoint.

The default deliberately matches AMD's Standard checkpoint
(`Qwen3.8-27B-MXFP4-mtpfp8`): vision blocks + lm_head are excluded from the body
pass, and the final config also carries the full weight-scoped `mtp.*` set.

Two lists because they act at different stages:
  * DEFAULT_QUANT_EXCLUDE is passed to Quark. `mtp.*` is a module-level pattern,
    so Quark leaves the MTP head unquantized (bf16), which the fp8 rewrite then
    converts. Without it the fp8 rewrite refuses a quantized MTP.
  * DEFAULT_MTP_EXCLUDE is the final-config form of the same exclusion, using the
    exact weight names AMD ships. vLLM's exclude matcher is exact or `re:` (never
    a glob), so `mtp.fc.weight` is load-inert; MTP still loads through
    layer_quant_config as fp8. Keeping it makes the config match AMD's byte for
    byte.
"""

DEFAULT_QUANT_EXCLUDE = [
    "model.visual.*",
    "lm_head",
    "mtp.*",
]

DEFAULT_MTP_EXCLUDE = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]

DEFAULT_MTP_KEEP_IN_EXCLUDE = True
