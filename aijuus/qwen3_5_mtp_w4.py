# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3_5 MTP model with W4 draft head integration.

Extends Qwen3_5MTP with the reduced-vocab W4 draft head approach from
tcclaviger/vllm:29.05.12. The draft head is cut from the checkpoint's bf16
lm_head at load time, quantized to int4 (group-128), and served through the
libr4d w4a16 GEMM kernel.
"""

import json
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_size,
    vocab_range_from_global_vocab_size,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)

# FORK-LOCAL: draft-only W4 lm_head
_DRAFT_HEAD = os.environ.get("CLAV_DRAFT_HEAD", "reduced")
_DRAFT_HEAD_REPLICATE = os.environ.get("CLAV_DRAFT_HEAD_REPLICATE", "0")


class Qwen3_5MTPW4(LocalArgmaxMixin, nn.Module):
    """Qwen3_5MTP with reduced-vocab W4 draft head.

    The draft head is cut from the checkpoint's bf16 lm_head at load time,
    quantized to int4 (group-128), and served through the libr4d w4a16 GEMM
    kernel. This avoids the full-vocab all-gather per draft step.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    use_int2_mtp_head = False
    cut_w4_draft_head = True
    draft_keep_file = "/app/tools/draft_vocab/keep_observed_qfn.json"
    draft_keep_pad = 64

    _W4_HEAD_TENSORS = ("weight_q4", "weight_scale", "weight_zero")
    _W4_REDUCED_TENSORS = ("weight_q4", "weight_scale", "weight_zero", "vocab_ids")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTPW4 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        # FORK-LOCAL: (PackedW4, gemm) installed by load_weights, else None/False = bf16.
        self._draft_w4_head: object | None = None
        # Reduced-vocab draft head: row -> vocab id (-1 = padding), set at load.
        self._draft_replicated = False
        self._draft_ids_all: torch.Tensor | None = None
        self._draft_ids_local: torch.Tensor | None = None
        self._draft_pad_local: torch.Tensor | None = None
        self._draft_cols: torch.Tensor | None = None
        self._draft_ids_valid: torch.Tensor | None = None

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    # ── FORK-LOCAL: draft-only W4 lm_head ────────────────────────────────────────

    def _w4_local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """SHARD-LOCAL draft logits through the W4-packed lm_head, or None."""
        if not self._draft_w4_head:
            return None
        packed, gemm_w4a16 = self._draft_w4_head
        if hidden_states.dim() != 2 or hidden_states.shape[-1] != packed.k:
            return None
        logits = gemm_w4a16(hidden_states, packed)
        lp = self.logits_processor
        if lp.head_dtype is not None and logits.dtype != lp.head_dtype:
            logits = logits.to(lp.head_dtype)
        return logits

    def _w4_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Full-vocab draft logits through the W4 head, or None for bf16."""
        logits = self._w4_local_logits(hidden_states)
        if logits is None:
            return None
        lp = self.logits_processor
        if self._draft_ids_all is not None:
            if not self._draft_replicated and self.lm_head.tp_size > 1:
                logits = tensor_model_parallel_all_gather(logits, dim=-1)
            full = logits.new_full((logits.shape[0], lp.org_vocab_size), -float("inf"))
            full.index_copy_(1, self._draft_ids_valid, logits.index_select(1, self._draft_cols))
            return full
        if self.lm_head.tp_size > 1:
            logits = lp._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : lp.org_vocab_size]
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        logits = self._w4_draft_logits(hidden_states)
        if logits is not None:
            return logits
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token without gathering the vocab-parallel logits."""
        logits = self._w4_local_logits(hidden_states)
        if logits is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        if self._draft_ids_all is not None:
            logits = logits.masked_fill(self._draft_pad_local, -float("inf"))
            vals, idx = logits.max(dim=-1)
            tokens = self._draft_ids_local[idx]
            tp_size = self.lm_head.tp_size
            if self._draft_replicated or tp_size == 1:
                return tokens
            pair = torch.stack([vals.float(), tokens.float()], dim=-1)
            gathered = tensor_model_parallel_all_gather(pair, dim=-1)
            gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
            best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            top = gathered[:, :, 1].gather(dim=-1, index=best)
            return top.squeeze(-1).to(torch.int64)
        shard = self.lm_head.shard_indices
        num_pad = shard.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        local_max_vals, local_max_indices = logits.max(dim=-1)
        global_indices = local_max_indices + shard.org_vocab_start_index
        tp_size = self.lm_head.tp_size
        if tp_size == 1:
            return global_indices
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    @staticmethod
    def _w4_kernels():
        """(gemm_w4a16, pack_from_codes, unpack_nibbles), or None with the reason logged."""
        try:
            from vllm.model_executor.kernels.draft_w4_lmhead import (
                available,
                gemm_w4a16,
                pack_from_codes,
                unpack_nibbles,
            )
        except ImportError:
            logger.warning("checkpoint carries an int4 draft lm_head but "
                           "draft_w4_lmhead is not installed; draft head stays bf16")
            return None
        if not available():
            logger.warning("checkpoint carries an int4 draft lm_head but libr4d "
                           "w4a16 is unavailable here; draft head stays bf16")
            return None
        return gemm_w4a16, pack_from_codes, unpack_nibbles

    @staticmethod
    def _pad_rows(t: torch.Tensor, n_pad: int) -> torch.Tensor:
        if t.shape[0] == n_pad:
            return t
        return torch.cat([t, t.new_zeros(n_pad - t.shape[0], *t.shape[1:])], 0)

    def _install_checkpoint_w4_head(self, parts: dict[str, torch.Tensor]) -> None:
        """Full-vocab checkpoint int4 head -> this rank's vocab shard."""
        kernels = self._w4_kernels()
        if kernels is None:
            self._draft_w4_head = False
            return
        gemm_w4a16, pack_from_codes, unpack_nibbles = kernels
        q4, scale, zero = (parts[k] for k in self._W4_HEAD_TENSORS)
        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        n = q4.shape[0]
        mult = DEFAULT_VOCAB_PADDING_SIZE
        n_pad = pad_vocab_size(n, mult)
        lo, hi = vocab_range_from_global_vocab_size(n_pad, rank, tp)
        dev = torch.cuda.current_device()
        q4, scale, zero = (self._pad_rows(t, n_pad)[lo:hi].to(dev) for t in (q4, scale, zero))
        packed = pack_from_codes(unpack_nibbles(q4), scale, zero)
        self._draft_w4_head = (packed, gemm_w4a16)
        logger.info("draft W4 lm_head (full vocab, split / %d) loaded from checkpoint: "
                    "rows %d..%d of %d -> %.0f MB packed", tp, lo, hi, n_pad,
                    packed.nbytes / 1e6)

    def _pack_reduced(self, parts, replicated: bool, kernels) -> dict:
        _, pack_from_codes, unpack_nibbles = kernels
        q4, scale, zero, ids = (parts[k] for k in self._W4_REDUCED_TENSORS)
        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        n = q4.shape[0]
        unit = 16 if replicated else 16 * tp
        n_pad = (n + unit - 1) // unit * unit
        ids = ids.to(torch.int64)
        if n_pad != n:
            ids = torch.cat([ids, ids.new_full((n_pad - n,), -1)])
        lo, hi = (0, n_pad) if replicated else (rank * (n_pad // tp), (rank + 1) * (n_pad // tp))
        dev = torch.cuda.current_device()
        q4, scale, zero = (self._pad_rows(t, n_pad)[lo:hi].to(dev) for t in (q4, scale, zero))
        ids = ids.to(dev)
        cols = torch.nonzero(ids >= 0).squeeze(1)
        return {
            "packed": pack_from_codes(unpack_nibbles(q4), scale, zero),
            "ids_all": ids,
            "ids_local": ids[lo:hi].contiguous(),
            "cols": cols,
            "ids_valid": ids[cols].contiguous(),
            "rows": (lo, hi, n_pad),
        }

    def _use_reduced(self, b: dict, replicated: bool, gemm_w4a16) -> None:
        self._draft_w4_head = (b["packed"], gemm_w4a16)
        self._draft_replicated = replicated
        self._draft_ids_all = b["ids_all"]
        self._draft_ids_local = b["ids_local"]
        self._draft_pad_local = b["ids_local"] < 0
        self._draft_cols = b["cols"]
        self._draft_ids_valid = b["ids_valid"]

    def _time_reduced_layouts(self, built: dict, gemm_w4a16) -> bool:
        """Time one draft step through each layout at M = 1 and 4."""
        dev = torch.cuda.current_device()
        k = built[True]["packed"].k
        ms = {}
        for replicated, b in built.items():
            self._use_reduced(b, replicated, gemm_w4a16)
            total = 0.0
            for m in (1, 4):
                h = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
                step = self.get_top_tokens
                for _ in range(10):
                    step(h)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    step(h)
                end.record()
                torch.cuda.synchronize()
                total += start.elapsed_time(end) / 50
            ms[replicated] = total
        votes = tensor_model_parallel_all_reduce(
            torch.tensor([ms[True], ms[False]], device=dev, dtype=torch.float32))
        replicated = bool(votes[0] <= votes[1])
        logger.info("draft W4 lm_head auto layout: full copy %.1f us, split %.1f us "
                    "(greedy draft step, M=1+4, this rank) -> %s", ms[True] * 1e3,
                    ms[False] * 1e3, "full copy per rank" if replicated else "split")
        return replicated

    def _install_reduced_w4_head(self, parts: dict[str, torch.Tensor]) -> None:
        """Reduced-vocab head: one copy per rank, rows split, or whichever times faster."""
        kernels = self._w4_kernels()
        if kernels is None:
            self._draft_w4_head = False
            return
        gemm_w4a16 = kernels[0]
        tp = get_tensor_model_parallel_world_size()
        mode = "1" if tp == 1 else _DRAFT_HEAD_REPLICATE
        if mode == "auto":
            built = {rep: self._pack_reduced(parts, rep, kernels) for rep in (True, False)}
            replicated = self._time_reduced_layouts(built, gemm_w4a16)
        else:
            replicated = mode == "1"
            built = {replicated: self._pack_reduced(parts, replicated, kernels)}
        b = built[replicated]
        self._use_reduced(b, replicated, gemm_w4a16)
        built.clear()
        lo, hi, n_pad = b["rows"]
        logger.info("draft W4 lm_head (reduced vocab %d ids, %s) loaded from checkpoint: "
                    "rows %d..%d of %d -> %.0f MB packed", b["ids_valid"].numel(),
                    "full copy per rank" if replicated else f"split / {tp}", lo, hi,
                    n_pad, b["packed"].nbytes / 1e6)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        heads: dict[str, dict[str, torch.Tensor]] = {"mtp.lm_head.": {},
                                                      "mtp.lm_head_reduced.": {}}

        def remap_weight_names():
            for name, weight in weights:
                prefix, _, part = name.rpartition(".")
                head = heads.get(prefix + ".")
                if head is not None and part in self._W4_REDUCED_TENSORS:
                    head[part] = weight
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(remap_weight_names())
        for prefix, parts in heads.items():
            loaded.update(prefix + k for k in parts)
        full, reduced = heads["mtp.lm_head."], heads["mtp.lm_head_reduced."]
        if "vocab_ids" in full:
            full, reduced = {}, full
        stored = [k for k, v in (("mtp.lm_head.*", full), ("mtp.lm_head_reduced.*", reduced)) if v]
        if _DRAFT_HEAD != "full" or not full:
            cut = self._cut_reduced_head()
            if cut:
                if stored:
                    logger.info("draft W4 lm_head: checkpoint int4 head(s) %s ignored; "
                                "using the load-time cut", ", ".join(stored))
                reduced, full = cut, {}
            elif not reduced:
                reduced = self._load_reduced_sidecar()
        for parts, need in ((full, self._W4_HEAD_TENSORS),
                            (reduced, self._W4_REDUCED_TENSORS)):
            if parts and not all(k in parts for k in need):
                raise ValueError(f"int4 draft lm_head is incomplete in the checkpoint: "
                                  f"have {sorted(parts)}, need {list(need)}")
        if reduced and (_DRAFT_HEAD != "full" or not full):
            if _DRAFT_HEAD == "full":
                logger.warning("CLAV_DRAFT_HEAD=full but the checkpoint only has the "
                               "reduced draft head; using it")
            self._install_reduced_w4_head(reduced)
        elif full:
            self._install_checkpoint_w4_head(full)
        return loaded

    def _checkpoint_file(self, filename: str) -> str | None:
        """``filename`` beside the checkpoint (local dir, else the hub cache), or None."""
        spec = self.vllm_config.speculative_config
        mc = (spec.draft_model_config if spec is not None and spec.draft_model_config
              is not None else self.vllm_config.model_config)
        if os.path.isdir(mc.model):
            path = os.path.join(mc.model, filename)
            return path if os.path.isfile(path) else None
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(mc.model, filename, revision=mc.revision)
        except Exception:
            return None

    def _cut_reduced_head(self) -> dict[str, torch.Tensor]:
        """Reduced head cut from the checkpoint's 16-bit lm_head.weight at the keep set."""
        if (not self.cut_w4_draft_head or not os.path.isfile(self.draft_keep_file)):
            return {}
        try:
            from vllm.model_executor.kernels.draft_w4_lmhead import available, quantize_w4
        except ImportError:
            return {}
        if not available():
            return {}
        index = self._checkpoint_file("model.safetensors.index.json")
        if index is not None:
            with open(index) as f:
                shard_name = json.load(f)["weight_map"].get("lm_head.weight")
        else:
            shard_name = "model.safetensors"
        shard = self._checkpoint_file(shard_name) if shard_name else None
        if shard is None:
            return {}
        from safetensors import safe_open

        with safe_open(shard, "pt", device="cpu") as f:
            if "lm_head.weight" not in f.keys():
                return {}
            if f.get_slice("lm_head.weight").get_dtype() not in ("BF16", "F16", "F32"):
                return {}
            weight = f.get_tensor("lm_head.weight")
        n_vocab = weight.shape[0]
        with open(self.draft_keep_file) as f:
            keep = sorted(set(json.load(f)))
        if not keep or keep[-1] >= n_vocab:
            return {}
        kset = set(keep)
        extra = [t for t in range(n_vocab) if t not in kset]
        extra = extra[: (-len(keep)) % self.draft_keep_pad]
        ids = torch.tensor(sorted(keep + extra), dtype=torch.int64)
        rows = weight.index_select(0, ids)
        del weight
        dev = torch.cuda.current_device()
        q4, scale, zero = [], [], []
        for r0 in range(0, rows.shape[0], 8192):
            q, s, z = quantize_w4(rows[r0 : r0 + 8192].to(dev))
            q4.append((q[:, 0::2] | (q[:, 1::2] << 4)).cpu())
            scale.append(s.cpu())
            zero.append(z.cpu())
        return {
            "weight_q4": torch.cat(q4),
            "weight_scale": torch.cat(scale),
            "weight_zero": torch.cat(zero),
            "vocab_ids": ids,
        }

    def _load_reduced_sidecar(self) -> dict[str, torch.Tensor]:
        """mtp.lm_head_reduced.* from sidecar next to the checkpoint."""
        path = self._checkpoint_file("mtp-lm-head-reduced.safetensors")
        if path is None:
            return {}
        from safetensors.torch import load_file

        prefix = "mtp.lm_head_reduced."
        parts = {k[len(prefix):]: v for k, v in load_file(path).items()
                 if k.startswith(prefix)}
        logger.info("draft W4 lm_head: reduced head read from %s", path)
        return parts


class Qwen3_5MoeMTPW4(Qwen3_5MTPW4, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()
