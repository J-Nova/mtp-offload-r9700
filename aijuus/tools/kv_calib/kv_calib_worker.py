"""Worker-side half of --kvcalibration.

The Attention layers live in the TP worker processes, so the accumulator has to
live there too and come back over ``collective_rpc``. Mirrors the shape of
tools/expert_map/expert_map_worker.py: a mixin dynamically inherited by the
Worker class, every attribute prefixed so it cannot collide with the Worker's
own names.

What it does
------------
``Attention.forward(query, key, value, ...)`` sees q/k/v before the cache
write, so ``kv_install`` wraps it with a version that folds each tensor's amax
into the accumulator and then delegates to the original. (The pre-v27 seam,
``calc_kv_scales`` / ``maybe_calc_kv_scales``, was removed upstream in v27.)
The wrap installs AFTER engine construction, so profiling/warmup forwards are
excluded; it never disarms, so every calibration forward folds — the same
semantics the old accumulating calc_kv_scales replacement had. The capture
requires enforce_eager: the wrapper is Python, cudagraph replay would skip it.

Why min AND max: both are over PER-SEQUENCE amax values, not over raw elements.
The min of ``abs(t)`` is ~0 on every sequence and says nothing; the spread of the
maxima across the corpus is what tells you whether the observed range was
actually explored, and therefore whether padding the top by 10% is meaningful.
"""

from __future__ import annotations

# layer_name -> {"q"/"k"/"v" -> {"max","min","sum","n"}}
_ACC: dict[str, dict[str, dict]] = {}
_ORIG_FORWARD = None
_ORIG_RUN_QSA = None
_ARMED = False


_PROBE_MAGIC = 0.031415926  # improbable-in-a-checkpoint sentinel

_SUFFIXES = (("q", "q_scale"), ("k", "k_scale"), ("v", "v_scale"))


def _blank() -> dict:
    return {"max": None, "min": None, "sum": 0.0, "n": 0}


def _index_block_spellings(model_path, block_path):
    """Checkpoint spellings of a block, read from the model's own index.

    The registry path and the checkpoint name of the same block can differ by
    a wrapper transformation ("language_model.model.X" internally vs
    "model.language_model.X" on disk). Rather than encode inversions, find the
    block's tail ("layers.11.self_attn") inside the real index keys and take
    whatever prefixes the checkpoint actually uses. Ambiguity (another tower
    with the same tail) is harmless: every spelling is only a probe candidate,
    and the probe rejects the ones that do not land.
    """
    import json
    from pathlib import Path

    parts = block_path.split(".")
    tail = ".".join(parts[-3:]) if len(parts) >= 3 else block_path
    try:
        wmap = json.loads(
            (Path(model_path) / "model.safetensors.index.json").read_text()
        )["weight_map"]
    except (OSError, ValueError, KeyError):
        return []
    out = set()
    marker = "." + tail + "."
    for key in wmap:
        pos = key.find(marker)
        if pos >= 0:
            out.add(key[: pos + len(marker) - 1])
        elif key.startswith(tail + "."):
            out.add(tail)
    return sorted(out)


def _probe_checkpoint_names(model_path, models, registry_layers):
    """For every armed layer, find q/k/v tensor names PROVEN to load.

    Ground truth is each model's own ``load_weights``: a candidate name is
    accepted only when loading a probe scalar through it lands on that layer's
    scale parameter (checked by value on the very module the runtime resolves
    through). No naming convention is assumed, so any arch/wrapper/fusion
    layout that vLLM can load, this can name — and a layer no candidate
    reaches is a hard error, never a silently wrong name.

    ``models`` is [(model, is_draft)]; a layer's names come from whichever
    model's loader routes to it, and that also classifies it as target/draft.
    """
    import torch

    # The scale params only exist DURING load (created by BaseKVCacheMethod,
    # deleted by process_weights_after_loading), so recreate any that are
    # missing for the duration of the probe and remove them afterwards.
    created, saved = [], []
    for mod in registry_layers.values():
        for _, suffix in _SUFFIXES:
            if hasattr(mod, suffix):
                saved.append((mod, suffix, getattr(mod, suffix).detach().clone()))
            else:
                setattr(
                    mod,
                    suffix,
                    torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False),
                )
                created.append((mod, suffix))

    probe_debug: dict[str, str] = {}

    def lands(model, cand, param):
        with torch.no_grad():
            param.copy_(-1.0)
        try:
            model.load_weights([(cand, torch.tensor(_PROBE_MAGIC))])
        except Exception as e:  # noqa: BLE001 - an unroutable name may raise; that IS the signal
            probe_debug[cand] = f"raised: {type(e).__name__}: {e}"[:200]
            return False
        # float32 tolerance: the probe scalar round-trips through fp32
        # storage, so demand closeness, not bit-equality. Real scales are
        # O(0.01..1) and the reset sentinel is -1.0 — 1e-6 cannot confuse them.
        if abs(float(param) - _PROBE_MAGIC) < 1e-6:
            return True
        probe_debug[cand] = f"no-land: param stayed {float(param):.3f}"
        return False

    try:
        names, draft_layer_names, unverified = {}, [], set()
        for lname, mod in registry_layers.items():
            block_path = lname.rsplit(".", 1)[0] if "." in lname else lname
            spellings = [block_path] + [
                s
                for s in _index_block_spellings(model_path, block_path)
                if s != block_path
            ]
            mapping, routed_by = {}, None
            for tag, suffix in _SUFFIXES:
                param = getattr(mod, suffix)
                cands = [
                    f"{sp}.{mid}.{suffix}"
                    for sp in spellings
                    for mid in ("attn", f"{tag}_proj", "qkv_proj")
                ] + [
                    # bare spelling: attention owners that register the scale
                    # buffers on THEMSELVES rather than an .attn/.proj child
                    # (e.g. Qwen4Exp QSA via set_default_quant_scales).
                    f"{sp}.{suffix}"
                    for sp in spellings
                ]
                hit = None
                for model, is_draft in models:
                    for cand in cands:
                        if lands(model, cand, param):
                            hit, routed_by = cand, is_draft
                            break
                    if hit:
                        break
                if not hit:
                    # Do NOT discard the capture over naming: fall back to the
                    # generic spelling and FLAG the layer. The driver writes
                    # flagged layers to a loudly named side file instead of
                    # the checkpoint index, so the data survives and only the
                    # name needs fixing.
                    hit = f"{block_path}.{suffix}"
                    unverified.add(lname)
                mapping[tag] = hit
            names[lname] = mapping
            if routed_by:
                draft_layer_names.append(lname)
        return names, draft_layer_names, sorted(unverified), probe_debug
    finally:
        with torch.no_grad():
            for mod, suffix, val in saved:
                getattr(mod, suffix).copy_(val)
        for mod, suffix in created:
            delattr(mod, suffix)


def _fold(layer_name: str, tag: str, amax: float) -> None:
    s = _ACC.setdefault(layer_name, {}).setdefault(tag, _blank())
    s["max"] = amax if s["max"] is None else max(s["max"], amax)
    s["min"] = amax if s["min"] is None else min(s["min"], amax)
    s["sum"] += amax
    s["n"] += 1


class KVCalibWorkerExtension:
    """Dynamically inherited by the Worker class (worker_base.py:261-287)."""

    def kv_install(self, cfg: dict) -> dict:
        """Wrap Attention.forward with the amax-accumulating version.

        v27 removed calc_kv_scales / maybe_calc_kv_scales (the old capture
        seam), so the wrap happens on forward itself — same q/k/v tensors,
        same point before the cache write. Installing AFTER engine
        construction also means the profiling/warmup forwards are naturally
        excluded from the statistics.

        Returns the layer inventory this rank sees, so the driver can name the
        output tensors from the module tree rather than from a pattern.
        """
        global _ORIG_FORWARD, _ARMED

        import torch

        from vllm.model_executor.layers.attention.attention import Attention

        vllm_config = self.vllm_config

        # enforce_eager: cudagraph replay executes no Python, so maybe_calc_kv_scales
        # would never fire for the captured batch sizes and those sequences would
        # vanish from the statistics without a word.
        if not vllm_config.model_config.enforce_eager:
            raise RuntimeError(
                "kv calibration requires enforce_eager=True; cudagraph replay "
                "runs no Python, so the capture would silently miss sequences."
            )

        # A checkpoint that declares kv_cache_scheme forces kv_cache_dtype to
        # fp8 — it already carries KV scales, and calibrating at fp8 KV would
        # measure the wrong (quantized) distribution.
        scheme = getattr(vllm_config.quant_config, "kv_cache_scheme", None)
        if scheme is not None:
            raise RuntimeError(
                f"this checkpoint declares quantization_config.kv_cache_scheme="
                f"{scheme!r}, so it already carries KV scales; there is "
                f"nothing to calibrate."
            )

        # Qwen4Exp QSA layers are their own attention owner class (not the
        # stock Attention); their q/k/v seam is _run_qsa. Guarded import so
        # every other architecture is untouched.
        try:
            from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention
        except ImportError:
            Qwen4ExpQSAAttention = None

        if not _ARMED:
            _ORIG_FORWARD = Attention.forward

            def forward_accum(self, query, key, value, *args, **kwargs):  # noqa: ANN001
                # Fires on EVERY forward (never disarms), like the old
                # accumulating calc_kv_scales replacement.
                for tag, t in (("q", query), ("k", key), ("v", value)):
                    if t is None:
                        continue
                    _fold(self.layer_name, tag, float(torch.abs(t).max().item()))
                return _ORIG_FORWARD(self, query, key, value, *args, **kwargs)

            Attention.forward = forward_accum

            if Qwen4ExpQSAAttention is not None:
                global _ORIG_RUN_QSA
                _ORIG_RUN_QSA = Qwen4ExpQSAAttention._run_qsa

                def run_qsa_accum(  # noqa: ANN001
                    self, hidden_states, positions, query, key, value, output,
                    gate=None,
                ):
                    for tag, t in (("q", query), ("k", key), ("v", value)):
                        _fold(self.layer_name, tag, float(torch.abs(t).max().item()))
                    return _ORIG_RUN_QSA(
                        self, hidden_states, positions, query, key, value,
                        output, gate,
                    )

                Qwen4ExpQSAAttention._run_qsa = run_qsa_accum
            _ARMED = True

        # The layer registry: name -> Attention instance, populated at build
        # time. Cheaper and more reliable than walking the model, and it is the
        # same mapping maybe_calc_kv_scales resolves through.
        armed_classes = (
            (Attention, Qwen4ExpQSAAttention)
            if Qwen4ExpQSAAttention is not None
            else (Attention,)
        )
        registry = vllm_config.compilation_config.static_forward_context
        layers = {}
        for name, mod in registry.items():
            if not isinstance(mod, armed_classes):
                continue
            # No per-layer re-arm needed on v27: the forward wrap above IS the
            # capture, installed after warmup, and it never disarms.
            layers[name] = {
                "kv_cache_dtype": getattr(mod, "kv_cache_dtype", None),
                "q_range": float(getattr(mod, "q_range", 0.0)),
                "k_range": float(getattr(mod, "k_range", 0.0)),
                "v_range": float(getattr(mod, "v_range", 0.0)),
            }

        if not layers:
            raise RuntimeError(
                "no Attention layer found in static_forward_context. A purely "
                "linear-attention model has no KV cache to calibrate."
            )

        # Checkpoint names are derived HERE, not in the driver: under vLLM v1 the
        # model lives only in the worker processes behind the MP client, and
        # LLMEngine has no model_executor to reach it through.
        #
        # Probed, not pattern-matched: every emitted name is verified against
        # the model's own load_weights before it is written, so it is a name
        # the NEXT serve provably routes to this layer — any arch, any wrapper
        # prefix, fused or split projections, target or MTP draft. See
        # _probe_checkpoint_names.
        registry_mods = {
            name: mod
            for name, mod in registry.items()
            if isinstance(mod, armed_classes)
        }
        models = [(self.model_runner.get_model(), False)]
        get_draft = getattr(self.model_runner, "get_draft_model", None)
        draft = get_draft() if get_draft is not None else None
        if draft is not None:
            models.append((draft, True))
        names, draft_layer_names, unverified, probe_debug = _probe_checkpoint_names(
            vllm_config.model_config.model, models, registry_mods
        )

        return {
            "layers": layers,
            "n_armed": len(layers),
            "names": names,
            "draft_layer_names": draft_layer_names,
            "unverified": unverified,
            "probe_debug": probe_debug,
        }

    def kv_reset(self) -> None:
        _ACC.clear()

    def kv_collect(self) -> dict:
        """Per-layer per-kind {max, min, mean, n} as plain floats."""
        out = {}
        for layer_name, kinds in _ACC.items():
            out[layer_name] = {
                tag: {
                    "max": s["max"],
                    "min": s["min"],
                    "mean": (s["sum"] / s["n"]) if s["n"] else None,
                    "n": s["n"],
                }
                for tag, s in kinds.items()
            }
        return out

    def kv_uninstall(self) -> None:
        global _ARMED
        from vllm.model_executor.layers.attention.attention import Attention

        if _ORIG_FORWARD is not None:
            Attention.forward = _ORIG_FORWARD
        if _ORIG_RUN_QSA is not None:
            from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention

            Qwen4ExpQSAAttention._run_qsa = _ORIG_RUN_QSA
        _ARMED = False
