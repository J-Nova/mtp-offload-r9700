#!/usr/bin/env python3
"""FORK-LOCAL runtime patch: DRY (Don't Repeat Yourself) sequence-level repetition penalty.

Ports tcclaviger's `patches/dry_sampler` (vLLM 0.29.0.dev0+g2bdbbc8080) onto our baked vLLM 0.29.0,
from the reference in aijuus/refs/tcclaviger-vllm-29.05.12/. DRY penalizes a token that would
continue an n-gram already seen earlier in the context:

    multiplier * base ** (match_len - allowed_length)

It is **default-off** (Server default `dry_multiplier` 0.0; a request must enable it via SamplingParams
or `--dry-multiplier`), so landing this patch does not change behaviour until enabled.

Requires the new module `vllm/v1/sample/ops/dry.py` (the entrypoint copies it from
aijuus/refs/tcclaviger-vllm-29.05.12/vllm/v1/sample/ops/dry.py) and must run AFTER patch_degen.py
(its arg_utils/config anchors sit on the degen-inserted lines).

Pure Python source edits, all-or-nothing, idempotent, warn-and-continue at the entrypoint.
Env: RADIANCE_LOCAL_DRY=0 skips entirely.
"""
import os
import sys
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
_buf: dict[str, str] = {}


def _src(path):
    if path not in _buf:
        _buf[path] = (SP / path).read_text()
    return _buf[path]


def _edit(path, old, new, marker=None):
    src = _src(path)
    if marker and marker in src:
        return
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"anchor not found/unique in {path} (count={n})")
    _buf[path] = src.replace(old, new, 1)


def _flush():
    for path, src in _buf.items():
        (SP / path).write_text(src)


DRY_IMPORT = "from vllm.v1.sample.ops.dry import DryParams, apply_dry  # FORK-LOCAL (patches/dry_sampler)\n"

SAMPLING_FIELDS = '''
    # FORK-LOCAL (patches/dry_sampler): None inherits the --dry-* server default;
    # an explicit 0/False wins.
    enable_dry: bool | None = None
    dry_multiplier: float | None = None
    """0 disables DRY for this request."""
    dry_base: float | None = None
    """penalty = multiplier * base ** (match_len - allowed_length)."""
    dry_allowed_length: int | None = None
    """Repeats up to this many tokens are free."""
    dry_range: int | None = None
    """Trailing context tokens to scan; -1 scans everything, 0 disables."""
    dry_sequence_breakers: list[str] | None = None
    """Strings a repeat may not span; a single-token breaker is never penalized."""
    _dry_params: Any | None = None
'''

PENALTIES_OLD = '''def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
'''

PENALTIES_NEW = '''def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    dry_params: dict[int, "DryParams"] | None = None,
    apply_classic: bool = True,
) -> torch.Tensor:
    """Applies classic penalties then DRY; ``dry_params`` is keyed by logits row."""
    _, vocab_size = logits.shape

    # FORK-LOCAL (patches/dry_sampler): classic penalties are skippable so a batch
    # with only DRY requests does not pay for them.
    if apply_classic:
        output_tokens_t = _convert_to_tensors(
            output_token_ids, vocab_size, logits.device
        )

        output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

        apply_penalties(
            logits,
            prompt_token_ids,
            output_tokens_t,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )

    if dry_params:
        # DRY must run after repetition penalty, which branches on logit sign.
        apply_dry(
            logits,
            prompt_token_ids,
            output_token_ids,
            dry_params,
            vocab_size,
        )

    return logits
'''

INPUT_METHODS = '''
    # FORK-LOCAL (patches/dry_sampler)
    @staticmethod
    def _dry_xarg(extra_args, name):
        if not extra_args or name not in extra_args:
            return None
        return extra_args[name]

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    def _resolve_dry_params(self, sampling_params: SamplingParams) -> None:
        """Precedence: xargs > SamplingParams.dry_* > --dry-* defaults; 0/False wins."""
        from vllm.v1.sample.ops.dry import (
            DRY_DEFAULT_BREAKERS,
            DryParams,
            build_breaker_seqs,
            ramp_to_base,
        )

        cfg = self.scheduler_config
        extra = sampling_params.extra_args

        def pick(field_name, xarg_name, cfg_value):
            for candidate in (
                self._dry_xarg(extra, xarg_name),
                getattr(sampling_params, field_name, None),
            ):
                if candidate is not None:
                    return candidate
            return cfg_value

        enabled = pick("enable_dry", "enable_dry", None)
        multiplier = pick(
            "dry_multiplier", "dry_multiplier", getattr(cfg, "dry_multiplier", 0.0)
        )
        base = pick("dry_base", "dry_base", None)
        if base is None:
            ramp = pick("dry_ramp", "dry_ramp", getattr(cfg, "dry_penalty_ramp", 0.75))
            base = ramp_to_base(float(ramp))
        allowed_length = pick(
            "dry_allowed_length",
            "dry_allowed_length",
            getattr(cfg, "dry_allowed_length", 2),
        )
        dry_range = pick("dry_range", "dry_range", getattr(cfg, "dry_range", -1))
        breakers = pick(
            "dry_sequence_breakers",
            "dry_sequence_breakers",
            getattr(cfg, "dry_sequence_breakers", None) or list(DRY_DEFAULT_BREAKERS),
        )

        multiplier = float(multiplier)
        if enabled is not None and not self._as_bool(enabled):
            sampling_params._dry_params = None
            return
        if enabled is not None and self._as_bool(enabled) and multiplier == 0.0:
            multiplier = 0.8

        if multiplier == 0.0 or dry_range == 0 or float(base) < 1.0:
            sampling_params._dry_params = None
            return

        if isinstance(breakers, str):
            breakers = [breakers]
        breaker_key = tuple(breakers or ())
        breaker_seqs: dict[int, list[list[int]]] = {}
        if breaker_key and self.tokenizer is not None:
            cached = self._dry_breaker_cache.get(breaker_key)
            if cached is None:
                cached = build_breaker_seqs(breaker_key, self.tokenizer)
                self._dry_breaker_cache[breaker_key] = cached
            breaker_seqs = cached

        sampling_params._dry_params = DryParams(
            multiplier=multiplier,
            base=float(base),
            allowed_length=int(allowed_length),
            range=int(dry_range),
            breaker_seqs=breaker_seqs,
        )

'''


def apply():
    # --- metadata.py ---
    _edit(
        "vllm/v1/sample/metadata.py",
        "import torch\n",
        "import torch\n" + DRY_IMPORT,
        marker="from vllm.v1.sample.ops.dry import DryParams",
    )
    _edit(
        "vllm/v1/sample/metadata.py",
        "    logprob_token_ids: dict[int, list[int]] | None = None\n",
        "    logprob_token_ids: dict[int, list[int]] | None = None\n"
        "\n"
        "    # FORK-LOCAL (patches/dry_sampler): req_index -> DryParams\n"
        "    dry_params: dict[int, DryParams] | None = None\n",
        marker="dry_params: dict[int, DryParams] | None = None",
    )
    # --- penalties.py ---
    _edit("vllm/v1/sample/ops/penalties.py", "import torch\n",
          "import torch\n" + DRY_IMPORT,
          marker="from vllm.v1.sample.ops.dry import DryParams")
    _edit("vllm/v1/sample/ops/penalties.py", PENALTIES_OLD, PENALTIES_NEW,
          marker="dry_params: dict[int, \"DryParams\"] | None = None,")
    # --- sampler.py ---
    _edit(
        "vllm/v1/sample/sampler.py",
        "        any_penalties_or_bad_words = (\n"
        "            bool(bad_words_token_ids) or not sampling_metadata.no_penalties\n"
        "        )\n",
        "        any_penalties_or_bad_words = (\n"
        "            bool(bad_words_token_ids)\n"
        "            or not sampling_metadata.no_penalties\n"
        "            or bool(sampling_metadata.dry_params)  # FORK-LOCAL (patches/dry_sampler)\n"
        "        )\n",
        marker="or bool(sampling_metadata.dry_params)",
    )
    _edit(
        "vllm/v1/sample/sampler.py",
        "        if sampling_metadata.no_penalties:\n"
        "            return logits\n"
        "\n"
        "        assert sampling_metadata.prompt_token_ids is not None\n"
        "        return apply_all_penalties(\n"
        "            logits,\n"
        "            sampling_metadata.prompt_token_ids,\n"
        "            sampling_metadata.presence_penalties,\n"
        "            sampling_metadata.frequency_penalties,\n"
        "            sampling_metadata.repetition_penalties,\n"
        "            output_token_ids,\n"
        "        )\n",
        "        dry_params = sampling_metadata.dry_params  # FORK-LOCAL (patches/dry_sampler)\n"
        "        apply_classic = not sampling_metadata.no_penalties\n"
        "        if not apply_classic and not dry_params:\n"
        "            return logits\n"
        "\n"
        "        assert sampling_metadata.prompt_token_ids is not None\n"
        "        return apply_all_penalties(\n"
        "            logits,\n"
        "            sampling_metadata.prompt_token_ids,\n"
        "            sampling_metadata.presence_penalties,\n"
        "            sampling_metadata.frequency_penalties,\n"
        "            sampling_metadata.repetition_penalties,\n"
        "            output_token_ids,\n"
        "            dry_params=dry_params,\n"
        "            apply_classic=apply_classic,\n"
        "        )\n",
        marker="dry_params=dry_params,",
    )
    # --- sampling_params.py ---
    _edit(
        "vllm/sampling_params.py",
        '    still computed. Conflict checking is performed at the engine level."""\n'
        "\n"
        "    @staticmethod\n"
        "    def from_optional(\n",
        '    still computed. Conflict checking is performed at the engine level."""\n'
        + SAMPLING_FIELDS
        + "\n    @staticmethod\n    def from_optional(\n",
        marker="_dry_params: Any | None = None",
    )
    # --- input_processor.py ---
    _edit(
        "vllm/v1/engine/input_processor.py",
        "        self.observability_config = vllm_config.observability_config\n",
        "        self.observability_config = vllm_config.observability_config\n"
        "        # FORK-LOCAL (patches/dry_sampler)\n"
        "        self._dry_breaker_cache: dict = {}\n",
        marker="self._dry_breaker_cache: dict = {}",
    )
    _edit(
        "vllm/v1/engine/input_processor.py",
        "    def process_inputs(\n",
        INPUT_METHODS + "    def process_inputs(\n",
        marker="def _resolve_dry_params(self, sampling_params",
    )
    _edit(
        "vllm/v1/engine/input_processor.py",
        "            if self.tokenizer is not None:\n"
        "                sampling_params.update_from_tokenizer(self.tokenizer)\n",
        "            if self.tokenizer is not None:\n"
        "                sampling_params.update_from_tokenizer(self.tokenizer)\n"
        "            self._resolve_dry_params(sampling_params)  # FORK-LOCAL (patches/dry_sampler)\n",
        marker="self._resolve_dry_params(sampling_params)  # FORK-LOCAL",
    )
    # --- gpu_input_batch.py ---
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "from vllm.v1.sample.metadata import SamplingMetadata\n",
        "from vllm.v1.sample.metadata import SamplingMetadata\n" + DRY_IMPORT,
        marker="from vllm.v1.sample.ops.dry import DryParams",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "        self.bad_words_token_ids: dict[int, list[list[int]]] = {}\n",
        "        self.bad_words_token_ids: dict[int, list[list[int]]] = {}\n"
        "\n"
        "        # FORK-LOCAL (patches/dry_sampler): req_index -> DryParams\n"
        "        self.dry_params: dict[int, DryParams] = {}\n",
        marker="self.dry_params: dict[int, DryParams] = {}",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "            if sampling_params.bad_words_token_ids:\n"
        "                self.bad_words_token_ids[req_index] = (\n"
        "                    sampling_params.bad_words_token_ids\n"
        "                )\n",
        "            if sampling_params.bad_words_token_ids:\n"
        "                self.bad_words_token_ids[req_index] = (\n"
        "                    sampling_params.bad_words_token_ids\n"
        "                )\n"
        "            # FORK-LOCAL: _dry_params crosses the process boundary as a plain dict.\n"
        "            dry_params = DryParams.from_any(\n"
        "                getattr(sampling_params, \"_dry_params\", None)\n"
        "            )\n"
        "            if dry_params is not None and dry_params.enabled:\n"
        "                self.dry_params[req_index] = dry_params\n",
        marker="dry_params = DryParams.from_any(",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "        self.bad_words_token_ids.pop(req_index, None)\n",
        "        self.bad_words_token_ids.pop(req_index, None)\n"
        "        self.dry_params.pop(req_index, None)  # FORK-LOCAL\n",
        marker="self.dry_params.pop(req_index, None)  # FORK-LOCAL",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "        swap_dict_values(self.bad_words_token_ids, i1, i2)\n",
        "        swap_dict_values(self.bad_words_token_ids, i1, i2)\n"
        "        swap_dict_values(self.dry_params, i1, i2)  # FORK-LOCAL\n",
        marker="swap_dict_values(self.dry_params, i1, i2)",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)\n"
        "            if bad_words_token_ids is not None:\n"
        "                self.bad_words_token_ids[empty_index] = bad_words_token_ids\n",
        "            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)\n"
        "            if bad_words_token_ids is not None:\n"
        "                self.bad_words_token_ids[empty_index] = bad_words_token_ids\n"
        "            dry_params = self.dry_params.pop(last_req_index, None)  # FORK-LOCAL\n"
        "            if dry_params is not None:\n"
        "                self.dry_params[empty_index] = dry_params\n",
        marker="dry_params = self.dry_params.pop(last_req_index, None)",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "        needs_prompt_token_ids = (\n"
        "            not self.no_penalties\n"
        "            or self.logits_processing_needs_token_ids[:num_reqs].any()\n"
        "        )\n",
        "        needs_prompt_token_ids = (\n"
        "            not self.no_penalties\n"
        "            or bool(self.dry_params)  # FORK-LOCAL\n"
        "            or self.logits_processing_needs_token_ids[:num_reqs].any()\n"
        "        )\n",
        marker="or bool(self.dry_params)  # FORK-LOCAL\n            or self.logits_processing_needs_token_ids",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "        needs_output_token_ids = (\n"
        "            not self.no_penalties\n"
        "            or bool(self.bad_words_token_ids)\n",
        "        needs_output_token_ids = (\n"
        "            not self.no_penalties\n"
        "            or bool(self.dry_params)  # FORK-LOCAL\n"
        "            or bool(self.bad_words_token_ids)\n",
        marker="or bool(self.dry_params)  # FORK-LOCAL\n            or bool(self.bad_words_token_ids)",
    )
    _edit(
        "vllm/v1/worker/gpu_input_batch.py",
        "            bad_words_token_ids=self.bad_words_token_ids,\n"
        "            logitsprocs=self.logitsprocs,\n",
        "            bad_words_token_ids=self.bad_words_token_ids,\n"
        "            dry_params=self.dry_params or None,  # FORK-LOCAL\n"
        "            logitsprocs=self.logitsprocs,\n",
        marker="dry_params=self.dry_params or None,",
    )
    # --- rejection_sampler.py ---
    _edit(
        "vllm/v1/sample/rejection_sampler.py",
        "        has_penalties = not sampling_metadata.no_penalties\n"
        "        any_penalties_or_bad_words = (\n"
        "            sampling_metadata.bad_words_token_ids or has_penalties\n"
        "        )\n",
        "        has_penalties = not sampling_metadata.no_penalties\n"
        "        has_dry = bool(sampling_metadata.dry_params)  # FORK-LOCAL (patches/dry_sampler)\n"
        "        any_penalties_or_bad_words = (\n"
        "            sampling_metadata.bad_words_token_ids or has_penalties or has_dry\n"
        "        )\n",
        marker="has_dry = bool(sampling_metadata.dry_params)",
    )
    _edit(
        "vllm/v1/sample/rejection_sampler.py",
        "        need_repeat_indices = (\n"
        "            sampling_metadata.allowed_token_ids_mask is not None or has_penalties\n"
        "        )\n",
        "        need_repeat_indices = (\n"
        "            sampling_metadata.allowed_token_ids_mask is not None\n"
        "            or has_penalties\n"
        "            or has_dry\n"
        "        )\n",
        marker="or has_penalties\n            or has_dry",
    )
    _edit(
        "vllm/v1/sample/rejection_sampler.py",
        "            repeat_indices = repeat_indices_cpu.to(\n"
        "                device=logits.device, non_blocking=True\n"
        "            )\n"
        "            logits = self.apply_penalties(\n"
        "                logits, sampling_metadata, metadata, repeat_indices, output_token_ids\n"
        "            )\n",
        "            repeat_indices = repeat_indices_cpu.to(\n"
        "                device=logits.device, non_blocking=True\n"
        "            )\n"
        "            # FORK-LOCAL (patches/dry_sampler): dry_params is keyed by request\n"
        "            # index; apply_dry wants rows.\n"
        "            dry_by_row: dict[int, object] | None = None\n"
        "            if has_dry:\n"
        "                src = sampling_metadata.dry_params or {}\n"
        "                dry_by_row = {}\n"
        "                for row, req_idx in enumerate(repeat_indices_cpu.tolist()):\n"
        "                    params = src.get(req_idx)\n"
        "                    if params is not None:\n"
        "                        dry_by_row[row] = params\n"
        "            logits = self.apply_penalties(\n"
        "                logits,\n"
        "                sampling_metadata,\n"
        "                metadata,\n"
        "                repeat_indices,\n"
        "                output_token_ids,\n"
        "                dry_by_row,\n"
        "            )\n",
        marker="dry_by_row: dict[int, object] | None = None",
    )
    _edit(
        "vllm/v1/sample/rejection_sampler.py",
        "        repeat_indices: torch.Tensor,\n"
        "        output_token_ids: list[list[int]],\n"
        "    ) -> torch.Tensor:\n"
        "        if sampling_metadata.no_penalties:\n"
        "            return logits\n"
        "\n"
        "        assert sampling_metadata.prompt_token_ids is not None\n"
        "\n"
        "        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]\n"
        "        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]\n"
        "        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]\n"
        "        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]\n"
        "\n"
        "        logits = apply_all_penalties(\n"
        "            logits,\n"
        "            prompt_token_ids,\n"
        "            presence_penalties,\n"
        "            frequency_penalties,\n"
        "            repetition_penalties,\n"
        "            output_token_ids,\n"
        "        )\n"
        "        return logits\n",
        "        repeat_indices: torch.Tensor,\n"
        "        output_token_ids: list[list[int]],\n"
        "        dry_by_row: dict[int, object] | None = None,\n"
        "    ) -> torch.Tensor:\n"
        "        apply_classic = not sampling_metadata.no_penalties  # FORK-LOCAL (patches/dry_sampler)\n"
        "        if not apply_classic and not dry_by_row:\n"
        "            return logits\n"
        "\n"
        "        assert sampling_metadata.prompt_token_ids is not None\n"
        "\n"
        "        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]\n"
        "        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]\n"
        "        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]\n"
        "        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]\n"
        "\n"
        "        logits = apply_all_penalties(\n"
        "            logits,\n"
        "            prompt_token_ids,\n"
        "            presence_penalties,\n"
        "            frequency_penalties,\n"
        "            repetition_penalties,\n"
        "            output_token_ids,\n"
        "            dry_params=dry_by_row,\n"
        "            apply_classic=apply_classic,\n"
        "        )\n"
        "        return logits\n",
        marker="dry_by_row: dict[int, object] | None = None,\n    ) -> torch.Tensor:",
    )
    # --- config/scheduler.py ---
    _edit(
        "vllm/config/scheduler.py",
        "    is_multimodal_model: bool = False\n",
        "    # FORK-LOCAL (patches/dry_sampler): DRY defaults match llama.cpp (common/common.h).\n"
        "    dry_multiplier: float = Field(default=0.0, ge=0.0)\n"
        '    """Server default DRY scale; 0 disables DRY unless a request enables it."""\n'
        "\n"
        "    dry_penalty_ramp: float = Field(default=0.75, ge=0.0)\n"
        '    """DRY exponential base = 1 + ramp; 0.75 gives llama.cpp\'s 1.75."""\n'
        "\n"
        "    dry_allowed_length: int = Field(default=2, ge=1)\n"
        '    """Repeats up to this many tokens are free."""\n'
        "\n"
        "    dry_range: int = Field(default=-1, ge=-1)\n"
        '    """Trailing context tokens DRY scans; -1 scans everything, 0 disables."""\n'
        "\n"
        "    dry_sequence_breakers: list[str] | None = None\n"
        '    """Strings a repeat may not span; None uses llama.cpp\'s defaults."""\n'
        "\n"
        "    is_multimodal_model: bool = False\n",
        marker="dry_multiplier: float = Field(default=0.0, ge=0.0)",
    )
    # --- engine/arg_utils.py (anchors on the degen-inserted lines) ---
    _edit(
        "vllm/engine/arg_utils.py",
        "    degen_min_span: int = SchedulerConfig.degen_min_span\n",
        "    degen_min_span: int = SchedulerConfig.degen_min_span\n"
        "    dry_multiplier: float = SchedulerConfig.dry_multiplier\n"
        "    dry_penalty_ramp: float = SchedulerConfig.dry_penalty_ramp\n"
        "    dry_allowed_length: int = SchedulerConfig.dry_allowed_length\n"
        "    dry_range: int = SchedulerConfig.dry_range\n"
        "    dry_sequence_breakers: list[str] | None = SchedulerConfig.dry_sequence_breakers\n",
        marker="dry_multiplier: float = SchedulerConfig.dry_multiplier",
    )
    _edit(
        "vllm/engine/arg_utils.py",
        '            "--degen-min-span", **scheduler_kwargs["degen_min_span"]\n        )\n',
        '            "--degen-min-span", **scheduler_kwargs["degen_min_span"]\n        )\n'
        "        scheduler_group.add_argument(\n"
        '            "--dry-multiplier", **scheduler_kwargs["dry_multiplier"]\n'
        "        )\n"
        "        scheduler_group.add_argument(\n"
        '            "--dry-penalty-ramp", **scheduler_kwargs["dry_penalty_ramp"]\n'
        "        )\n"
        "        scheduler_group.add_argument(\n"
        '            "--dry-allowed-length", **scheduler_kwargs["dry_allowed_length"]\n'
        "        )\n"
        '        scheduler_group.add_argument("--dry-range", **scheduler_kwargs["dry_range"])\n'
        "        scheduler_group.add_argument(\n"
        '            "--dry-sequence-breakers", **scheduler_kwargs["dry_sequence_breakers"]\n'
        "        )\n",
        marker='"--dry-multiplier"',
    )
    _edit(
        "vllm/engine/arg_utils.py",
        "            degen_min_span=self.degen_min_span,\n",
        "            degen_min_span=self.degen_min_span,\n"
        "            dry_multiplier=self.dry_multiplier,\n"
        "            dry_penalty_ramp=self.dry_penalty_ramp,\n"
        "            dry_allowed_length=self.dry_allowed_length,\n"
        "            dry_range=self.dry_range,\n"
        "            dry_sequence_breakers=self.dry_sequence_breakers,\n",
        marker="dry_multiplier=self.dry_multiplier,",
    )


def main():
    if os.environ.get("RADIANCE_LOCAL_DRY", "1") != "1":
        print("[patch_dry] disabled (RADIANCE_LOCAL_DRY != 1)")
        return
    try:
        apply()
        _flush()
    except Exception as e:
        sys.stderr.write(f"[patch_dry] FAILED, nothing applied: {e!r}\n")
        sys.exit(1)
    print("[patch_dry] applied (DRY sequence-level repetition penalty)")


if __name__ == "__main__":
    main()
