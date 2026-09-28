#!/usr/bin/env python3
"""Annotate the EAGLE/MTP draft KV group positionally, for hybrid (Mamba + attention) models.

Ported from zzpanic/qwen3.6-vllm-gfx1201-launchers `kv-cache/patches/patch_eagle_groups.py`
(same vLLM 0.27.1 / radiance 0.9.3 image), re-anchored to this tree.

WHY. `vllm/v1/core/kv_cache_utils.py` has exactly one annotator,
`_annotate_eagle_groups_deepseek_v4`, gated twice: it is only *called* from the
DeepSeek-V4 grouping branch, and it returns early unless a spec carries
`model_version == "deepseek_v4"`. A hybrid Mamba+attention model never annotates a group, so
the offload scheduler's fail-safe fallback flags EVERY group as a draft group:

    if use_eagle and not eagle_groups:
        eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

The fallback is not cosmetic. `GroupOffloadConfig.is_eagle_group` changes both paths:
  * RequestOffloadState.storable_chunks() drops the trailing chunk of an eagle group while
    decoding (draft KV for the last accepted position can be rewritten on spec rejection).
    Applied to all nine groups it withholds the newest chunk of the whole conversation -- the
    one a follow-up turn is most likely to ask for.
  * _lookup() queries one extra chunk for an eagle group and pops it, so the servable prefix
    is a chunk shorter, nine times over.

`patch_offload_eagle_fallback.py` already un-breaks offload by restricting the fallback to
non-Mamba groups. That leaves the two FULL-ATTENTION groups flagged eagle (this model's nine
groups are 0-5 GDN, 6-7 full attention, 8 draft). The annotator is the correct fix: only the
real draft group (g8) is volatile, so only it should be flagged.

WHAT THIS CHANGES (both edits in kv_cache_utils.py):
  A. Drop the `model_version == "deepseek_v4"` early return. The rule the function implements
     -- "the draft model's attention layer is registered last, so flag whichever group holds
     the last layer" -- is a fact about how vLLM registers a draft model, not about DeepSeek.
  B. Call the annotator on the general hybrid page-size path too, right after
     `_get_kv_cache_groups_uniform_page_size`.

Self-checking: after a restart the boot line must read
`KV offloading: EAGLE/MTP draft attention groups [8] detected.` If it still says [0..8] the
annotation did not take.

GATE. `RADIANCE_OFFLOAD_EAGLE_GROUPS` default 0 = upstream DeepSeek-only gate (this tree's
current behaviour: the fallback patch decides, giving [6,7,8]). Set to 1 to enable the
annotation (expect [8]). Enabled, it strictly narrows the eagle set, so it can only store more
chunks and serve longer prefixes -- g8 stays flagged, so no volatile draft chunk can be served.
Because it changes the stored set, re-validate with turnbench/`probe_correct.py` before making
it the deployment default.
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patchlib import apply, apply_any  # noqa: E402

SP = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"]))
KVU = SP / "vllm" / "v1" / "core" / "kv_cache_utils.py"

print("[radiance] offload eagle-group annotation")

apply(
    KVU,
    anchor=(
        "    # Detection uses the merged MLA spec's model_version.\n"
        "    if not any(\n"
        '        getattr(spec, "model_version", None) == "deepseek_v4"\n'
        "        for spec in kv_cache_spec.values()\n"
        "    ):\n"
        "        return\n"
    ),
    new=(
        "    # Detection uses the merged MLA spec's model_version.\n"
        "    # The rule this function applies -- the draft model's attention layer is registered\n"
        "    # last, so flag whichever group holds the last layer -- is a fact about how vLLM\n"
        "    # registers a draft model, not anything specific to DeepSeek. Gating it on\n"
        "    # model_version leaves every other speculative model unannotated, which trips the\n"
        "    # offload scheduler's flag-them-all fallback. RADIANCE_OFFLOAD_EAGLE_GROUPS=0 restores\n"
        "    # the DeepSeek-only gate.\n"
        "    if os.environ.get(\"RADIANCE_OFFLOAD_EAGLE_GROUPS\", \"0\") != \"1\" and not any(\n"
        '        getattr(spec, "model_version", None) == "deepseek_v4"\n'
        "        for spec in kv_cache_spec.values()\n"
        "    ):\n"
        "        return\n"
    ),
    sentinel="RADIANCE_OFFLOAD_EAGLE_GROUPS",
    label="A kv_cache_utils: generalize eagle annotation past the deepseek_v4 gate",
)

_ANNOT = (
    "\n"
    "    # Annotate the EAGLE/MTP draft group on the uniform-page-size path too. Upstream\n"
    "    # annotates only on the DeepSeek-V4 branch above, so a hybrid Mamba+attention model\n"
    "    # lands here with nothing annotated and the offload scheduler flags all of its groups\n"
    "    # as draft groups. filtered_spec (not kv_cache_spec) keeps registration order while\n"
    "    # excluding the hidden-state layers that are not in `groups` yet.\n"
    "    _annotate_eagle_groups_deepseek_v4(vllm_config, filtered_spec, groups)\n"
)

# Two shapes: the shipped tree has already had patch_kv_group_size.py add `, vllm_config`
# to this call; a tree without that patch still has the single-argument form.
apply_any(
    KVU,
    variants=[
        (
            "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec, vllm_config)\n",
            "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec, vllm_config)\n"
            + _ANNOT,
        ),
        (
            "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)\n",
            "    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)\n" + _ANNOT,
        ),
    ],
    sentinel="_annotate_eagle_groups_deepseek_v4(vllm_config, filtered_spec, groups)",
    label="B kv_cache_utils: annotate eagle groups on the hybrid page-size path",
)

print("[radiance] applied -- with RADIANCE_OFFLOAD_EAGLE_GROUPS=1 expect 'draft attention groups [8]'")
