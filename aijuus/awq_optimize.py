#!/usr/bin/env python3
"""Quality-preserving acceleration for Quark's AWQ pass.

What this changes: nothing about what the AWQ search computes. It installs
class-level patches on ``quark.torch.algorithm.awq.awq.AwqProcessor`` that:

1. record each decoder layer's scale and clip results as they are produced, to a
   small checkpoint (``awq_cache.pt``), written atomically after every layer;
2. replay those recorded results verbatim on a later run, skipping the 20-step
   alpha search and the clip search for every layer already recorded;
3. log the search's own minimum loss per scaling group to ``awq_report.csv``.

Everything else -- ``_get_input_feat``, ``apply_scale``, ``apply_clip``,
``_apply_quant``, the device moves -- is Quark's untouched code.

Why replay is byte-identical to recomputing
-------------------------------------------
In Quark's ``apply()`` loop the inputs to layer *i* are captured before any
scale is applied to layer *i*, and the residual stream fed to layer *i* is
produced by layer *i-1*'s forward, which was itself taken before layer *i-1* was
scaled. Scaling is output-preserving, so the whole ``self.inps`` chain is the
pristine forward and none of it depends on earlier layers' scales. A layer's
scale search therefore depends only on (that layer's pristine weights, its
captured input features, and the shared module kwargs). Replaying a recorded
result is exactly the same arithmetic, on the same inputs, on the same device.

To keep that guarantee honest, the checkpoint is stamped with a sha256 of every
input that can change a result (source index/config, recipe file, calibration
counts, batch size, the env toggles that alter the forward, the Quark build, and
the placement split). On mismatch it refuses to resume and prints a diff, exactly
because resuming across a changed calibration set would silently quantize the
front and back halves of the model differently.

The invariance gate
-------------------
``python awq_optimize.py compare A.awq_cache.pt B.awq_cache.pt`` hashes every
recorded scale/clip tensor in two caches and fails if they differ. Run the same
model twice with two different ``QUARK_LAYER_SPLIT`` values and compare: that is
the empirical check that moving a layer between the two GPUs does not change a
single byte. (CPU<->GPU is a different kernel family; treat it as a separate
axis and compare GPU-only runs first.)

Env:
  QUARK_AWQ_CACHE=0            disable caching/replay (run purely as Quark does)
  QUARK_AWQ_NO_RESUME=1        ignore an existing checkpoint (still writes a new one)
  QUARK_AWQ_CACHE_FORCE=1      resume despite a config-hash mismatch
  QUARK_AWQ_CACHE_PATH=PATH    override the checkpoint location
  QUARK_AWQ_REPORT=0           disable the per-group CSV
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

CACHE_VERSION = 1
CACHE_NAME = "awq_cache.pt"
REPORT_NAME = "awq_report.csv"
REPORT_FIELDS = ["layer", "prev_op", "layers", "ratio", "best_error",
                 "scales_min", "scales_max", "scales_mean"]


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def sha_file(path, n=32):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return None


def _resolve_source_root(model_id):
    """A local directory as-is, or the cached HF snapshot (no network)."""
    p = Path(model_id)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(
            model_id, local_files_only=True,
            allow_patterns=["config.json", "model.safetensors.index.json",
                            "model.safetensors"]))
    except Exception:
        return None


def installed_awq_sha():
    try:
        import quark.torch.algorithm.awq.awq as _awq
        return sha_file(_awq.__file__)
    except Exception:
        return None


def quark_version():
    try:
        import quark
        return getattr(quark, "__version__", "unknown")
    except Exception:
        return "unknown"


def compute_config_hash(*, model_id, recipe_path, env):
    """Return (hash, payload). Everything that changes what a completed layer
    would have contained, per the same reasoning Quark's own AA resume uses."""
    src = _resolve_source_root(model_id)
    payload = {
        "cache_version": CACHE_VERSION,
        "source_config": sha_file(src / "config.json") if src else None,
        "source_index": sha_file(src / "model.safetensors.index.json") if src else None,
        "source_single": sha_file(src / "model.safetensors") if src else None,
        "recipe": sha_file(recipe_path) if recipe_path else None,
        "quark_version": quark_version(),
        "awq_module": installed_awq_sha(),
        # Calibration / forward identity.
        "dataset": env.get("QUARK_CALIB_DATASET", "pileval"),
        "num_calib_data": env.get("NUM_CALIB_DATA"),
        "seq_len": env.get("SEQ_LEN"),
        "batch_size": env.get("QUARK_CALIB_BATCH"),
        "data_type": "auto",
        # Toggles that change the forward the inputs are captured from.
        "awq_memory_opt": env.get("QUARK_AWQ_MEMORY_OPTIMIZATION"),
        "buffer_reuse": env.get("QUARK_ENABLE_BUFFER_REUSE"),
        # Placement is nominally result-neutral, but CPU<->GPU is a different
        # kernel family, so pin it and refuse to resume across a change.
        "layer_split": env.get("QUARK_LAYER_SPLIT_RESOLVED")
                      or env.get("QUARK_LAYER_SPLIT"),
        "parallel": env.get("QUARK_AWQ_PARALLEL"),
    }
    # Only stamp the parallel flag when it is actually on. Adding the key
    # unconditionally changed the hash of every serial run, which invalidated
    # existing caches for no reason; the serial payload must stay byte-stable.
    if not _truthy(env.get("QUARK_AWQ_PARALLEL", "0")):
        payload.pop("parallel", None)
    # Same rule for the quantization plan: only mixes into the hash when one is
    # actually in use, so plan-free runs keep their existing AWQ caches. A plan
    # change (different scheme/exclude/KV/MTP) must force a fresh search.
    if env.get("QUANT_PLAN"):
        payload["quant_plan"] = sha_file(env["QUANT_PLAN"])
    if env.get("PLAN_QUANT_PLAN_RESOLVED"):
        payload["quant_plan_resolved"] = sha_file(env["PLAN_QUANT_PLAN_RESOLVED"])
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32], payload


def _diff(old, new):
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


class AWQCache:
    """Recorded AWQ results for one exact run configuration."""

    def __init__(self, path, cfg_hash, payload, *, write=True, resume=True,
                 force=False, report=True):
        self.path = Path(path)
        self.hash = cfg_hash
        self.payload = payload
        self.write = write
        self.report = report
        # Anything to do at all? If caching, resuming and reporting are all off,
        # leave Quark completely untouched.
        self.active = bool(write or resume or report)
        self.scales = {}   # key -> (scales_cpu, ratio, best_error)
        self.clips = {}    # layer_path -> [(name, max_val_cpu), ...]
        self._report_fh = None
        self._report_writer = None
        self._dirty = False
        self._lock = threading.Lock()   # guards dict mutation + save/report under threads
        self.stats = {"scale_hits": 0, "scale_misses": 0, "clip_hits": 0, "clip_misses": 0}

        if resume and self.path.exists():
            self._load(force=force)
        elif self.path.exists() and not resume:
            print(f"  [AWQ-CACHE] ignoring existing {self.path.name} (no-resume)")

    # -- persistence ---------------------------------------------------------
    def _load(self, *, force):
        try:
            data = torch_load(self.path)
        except Exception as exc:
            print(f"  [AWQ-CACHE] WARNING: unreadable {self.path}: {exc!r}; "
                  "starting fresh")
            return
        old_hash = data.get("hash")
        if old_hash != self.hash and not force:
            changes = _diff(data.get("payload", {}), self.payload)
            raise SystemExit(
                f"\n[AWQ-CACHE] refusing to resume {self.path}\n"
                f"  the checkpoint was written by a different run configuration:\n    "
                + "\n    ".join(changes or ["<hash differs>"])
                + "\n  This would quantize the model's front and back halves under "
                  "different settings.\n"
                  "  Move the checkpoint aside, or pass QUARK_AWQ_CACHE_FORCE=1 if "
                  "the change cannot affect the result.\n")
        if old_hash != self.hash:
            print("  [AWQ-CACHE] --force: resuming across a config change:")
            for line in _diff(data.get("payload", {}), self.payload):
                print(f"      {line}")
        self.scales = data.get("scales", {})
        self.clips = data.get("clips", {})
        n = len(self.scales)
        print(f"  [AWQ-CACHE] loaded {self.path.name}: {n} layer group(s) "
              f"recorded, hash {self.hash}")

    def save(self):
        if not self.write:
            return
        import torch
        with self._lock:
            snapshot = {"version": CACHE_VERSION, "hash": self.hash,
                        "payload": self.payload, "scales": dict(self.scales),
                        "clips": dict(self.clips)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(snapshot, tmp)
        os.replace(tmp, self.path)

    # -- lookups -------------------------------------------------------------
    def get_scales(self, key):
        hit = self.scales.get(key)
        if hit is None:
            self.stats["scale_misses"] += 1
            return None
        self.stats["scale_hits"] += 1
        return hit[0], hit[1]

    def put_scales(self, key, scales, ratio, best_error):
        with self._lock:
            self.scales[key] = (scales.detach().cpu(), float(ratio),
                                None if best_error is None else float(best_error))
        self._write_report_row(key, scales, ratio, best_error)

    def get_clips(self, layer_path):
        hit = self.clips.get(layer_path)
        if hit is None:
            self.stats["clip_misses"] += 1
            return None
        self.stats["clip_hits"] += 1
        return hit

    def put_clips(self, layer_path, clip_list):
        with self._lock:
            self.clips[layer_path] = [(n, t.detach().cpu()) for n, t in clip_list]

    # -- reporting -----------------------------------------------------------
    def _write_report_row(self, key, scales, ratio, best_error):
        if not self.report:
            return
        layer, prev, layers = key
        s = scales.float()
        row = [layer, prev, "|".join(layers), f"{ratio:.3f}",
               "" if best_error is None else f"{best_error:.6e}",
               f"{s.min().item():.6g}", f"{s.max().item():.6g}",
               f"{s.mean().item():.6g}"]
        with self._lock:
            if self._report_writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                csv_path = self.path.parent / REPORT_NAME
                new = not csv_path.exists()
                self._report_fh = open(csv_path, "a", newline="")
                self._report_writer = csv.writer(self._report_fh)
                if new:
                    self._report_writer.writerow(REPORT_FIELDS)
            self._report_writer.writerow(row)
            self._report_fh.flush()

    def close(self):
        if self._report_fh is not None:
            self._report_fh.close()
            self._report_fh = None

    def summary(self):
        s = self.stats
        return (f"scale hits/misses {s['scale_hits']}/{s['scale_misses']}, "
                f"clip hits/misses {s['clip_hits']}/{s['clip_misses']}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve(root, full):
    """Return (owner_module, 'p'|'b', leaf_name) for a dotted parameter/buffer
    path, or (None, None, None) if it does not resolve under ``root``."""
    parts = full.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p, None)
        if mod is None:
            return None, None, None
    leaf = parts[-1]
    if leaf in getattr(mod, "_parameters", {}):
        return mod, "p", leaf
    if leaf in getattr(mod, "_buffers", {}):
        return mod, "b", leaf
    return None, None, None


def _materialize_module(root, module):
    """Bring a layer's accelerate-offloaded weights back to real tensors on each
    hook's execution_device and clear the offload flag.

    Called one decoder layer at a time, just before that layer is processed:
    materializing every offloaded layer up front would not fit in VRAM, and
    Quark's AWQ loop needs real (non-meta) weights for apply_scale / clip /
    quantize and for the final `.to("cpu")`. The hook is kept (its pre_forward
    still moves inputs to the layer's device); execution_device is left as
    accelerate set it (the main GPU), so the frozen GDN/attention kernels run on
    the GPU as they must.
    """
    import torch
    import torch.nn as nn

    n = 0
    for name, sub in list(module.named_modules()):
        hook = getattr(sub, "_hf_hook", None)
        if hook is None or not getattr(hook, "offload", False):
            continue
        dev = getattr(hook, "execution_device", "cpu")
        if isinstance(dev, int):
            dev = torch.device(f"cuda:{dev}")
        elif not isinstance(dev, torch.device):
            dev = torch.device(dev)

        hook.offload = False
        wm = getattr(hook, "weights_map", None)
        if wm is not None:
            pref = getattr(wm, "prefix", "") or ""
            for k in list(wm.keys()):
                full = k if _resolve(root, k)[0] is not None else f"{name}.{k}"
                owner, kind, leaf = _resolve(root, full)
                if owner is None:
                    continue
                try:
                    # PrefixedDataset.__iter__ yields prefixed keys but
                    # __getitem__ re-adds the prefix, so strip it first.
                    val = wm[k[len(pref):]] if (pref and k.startswith(pref)) else wm[k]
                except Exception:
                    continue
                if val is None:
                    continue
                val = val.to(dev)
                if kind == "p":
                    owner._parameters[leaf] = nn.Parameter(val, requires_grad=False)
                else:
                    owner._buffers[leaf] = val
                n += 1

        # Real tensors not covered by weights_map (e.g. non-persistent buffers)
        # must share the hook's execution device or the forward mixes devices.
        for bn, b in list(sub.named_buffers(recurse=True)):
            if b is None or b.device.type == "meta" or b.device == dev:
                continue
            owner, kind, leaf = _resolve(root, f"{name}.{bn}")
            if owner is None:
                owner, kind, leaf = _resolve(sub, bn)
            if owner is None:
                continue
            if kind == "p":
                owner._parameters[leaf] = nn.Parameter(b.to(dev), requires_grad=False)
            elif kind == "b":
                owner._buffers[leaf] = b.to(dev)
    return n


# ---------------------------------------------------------------------------
# The patch. Class-level so PROCESSOR_MAP's reference to AwqProcessor sees it.
# ---------------------------------------------------------------------------
def install(cache):
    import torch  # noqa: F401
    import quark.torch.algorithm.awq.awq as _awq
    from quark.torch.utils.torch_utils import get_op_name

    P = _awq.AwqProcessor
    if getattr(P, "_xtc_awq_patched", False):
        return
    P._xtc_awq_patched = True

    orig_init = P.__init__
    orig_get_input_feat = P._get_input_feat
    orig_compute_loss = P._compute_loss
    orig_search_scale = P._search_best_scale
    orig_search_clip = P._search_best_clip
    orig_apply = P.apply

    # apply_scale / apply_clip are called with device=common_device, which for a
    # still-offloaded layer is "meta" (its params have not been materialized when
    # the loop computes the device). They also move the scales/clips there, which
    # would destroy them. Redirect both to the module's real device.
    _orig_apply_scale = getattr(_awq, "apply_scale", None)
    _orig_apply_clip = getattr(_awq, "apply_clip", None)

    if _orig_apply_scale is not None:
        def _apply_scale(module, scales_list, input_feat_dict=None, device=None, **kw):
            try:
                real = next(module.parameters()).device
            except StopIteration:
                real = device
            return _orig_apply_scale(module, scales_list,
                                     input_feat_dict=input_feat_dict,
                                     device=real, **kw)
        _awq.apply_scale = _apply_scale
    if _orig_apply_clip is not None:
        def _apply_clip(module, clip_list, device=None, **kw):
            try:
                real = next(module.parameters()).device
            except StopIteration:
                real = device
            return _orig_apply_clip(module, clip_list, device=real, **kw)
        _awq.apply_clip = _apply_clip

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Per-thread, because the parallel path runs one layer's search per
        # worker thread and they must not share the loss history / layer key.
        self._xtc_local = threading.local()

    def _get_input_feat(self, layer, named_linears):
        # The captured layer path keys every result for this layer.
        self._xtc_local.layer = get_op_name(self.model, layer)
        # Bring any accelerate-offloaded weights of THIS layer back to real
        # tensors before it is forwarded/searched. Done here because this runs
        # at the top of each AWQ iteration, and only one layer is resident at a
        # time, so the extra VRAM is transient.
        try:
            _materialize_module(self.model, layer)
        except Exception as exc:
            print(f"[AWQ-MAT] materialize failed: {exc!r}")
        return orig_get_input_feat(self, layer, named_linears)

    def _compute_loss(self, fp16_output, int_w_output, device):
        value = orig_compute_loss(self, fp16_output, int_w_output, device)
        hist = getattr(self._xtc_local, "hist", None)
        if hist is None:
            self._xtc_local.hist = hist = []
        try:
            hist.append(float(value))
        except (TypeError, ValueError):
            pass
        return value

    def _search_best_scale(self, module, prev_op, layers, inp,
                           module2inspect=None, kwargs={}):
        prev_name = get_op_name(module, prev_op)
        layer_names = tuple(get_op_name(module, m) for m in layers)
        key = (get_op_name(self.model, module), prev_name, layer_names)

        cached = cache.get_scales(key)
        if cached is not None:
            self._xtc_local.hist = []
            scales, ratio = cached
            return (prev_name, layer_names, scales, ratio)

        self._xtc_local.hist = []
        result = orig_search_scale(self, module, prev_op, layers, inp,
                                   module2inspect, kwargs)
        hist = self._xtc_local.hist
        best_error = min(hist) if hist else None
        cache.put_scales(key, result[2], result[3], best_error)
        cache.save()
        return result

    def _search_best_clip(self, named_linears, input_feat):
        layer_path = getattr(self._xtc_local, "layer", None)
        cached = cache.get_clips(layer_path) if layer_path is not None else None
        if cached is not None:
            return list(cached)
        result = orig_search_clip(self, named_linears, input_feat)
        if layer_path is not None:
            cache.put_clips(layer_path, result)
            cache.save()
        return result

    P.__init__ = __init__
    P._get_input_feat = _get_input_feat
    P._compute_loss = _compute_loss
    P._search_best_scale = _search_best_scale
    P._search_best_clip = _search_best_clip

    if _truthy(os.environ.get("QUARK_AWQ_PARALLEL", "0")):
        # Quark calls clear_memory() after every one of the ~2,560 grid steps,
        # and it does gc.collect() + empty_cache(). gc.collect() holds the GIL
        # across a huge object graph, which serialises the worker threads and
        # defeats the parallelism. Swap in a cheap empty_cache() for the hot
        # path; the parallel worker does one gc.collect() per completed layer.
        def _light_clear(*_a, **_k):
            torch.cuda.empty_cache()
        _awq.clear_memory = _light_clear
        P.apply = _make_parallel_apply(_awq, orig_apply)
        print("  [AWQ-PAR] parallel AWQ search enabled (one layer in flight per device)")


# ---------------------------------------------------------------------------
# Parallel AWQ: step 1 (forward + activation capture) stays serial in the main
# thread, and steps 2-4 (scale search -> apply_scale -> clip -> quantize) are
# dispatched per layer to a single worker thread per device. Layers' searches
# are independent given their captured inputs (see the module docstring), so
# this changes only scheduling, not arithmetic. One in-flight layer per device
# keeps each GPU's workspace the same as the serial path. The invariance gate
# passed on this hardware (GPU0 vs GPU1 byte-identical), which is the
# precondition for enabling this.
# ---------------------------------------------------------------------------
def _make_parallel_apply(m, orig_apply):
    import torch
    from concurrent.futures import ThreadPoolExecutor

    get_op_name = m.get_op_name

    def apply(self):
        from tqdm import tqdm

        # Fall back to Quark's serial loop whenever the assumptions this path
        # relies on do not hold, rather than risk a wrong result.
        reason = None
        if not getattr(self, "using_accelerate", True):
            reason = "not using Accelerate device placement"
        elif getattr(m, "QUARK_SAVE_ACTIVATION_SCALES", False):
            reason = "QUARK_SAVE_ACTIVATION_SCALES is set"
        elif any(("attn" in str(s.get("module2inspect", "")).lower()
                  or "attention" in str(s.get("module2inspect", "")).lower())
                 for s in (self.scaling_layers or [])):
            reason = "a scaling group inspects an attention module (uses shared self.inps)"
        if reason is not None:
            print(f"  [AWQ-PAR] falling back to serial AWQ: {reason}")
            return orig_apply(self)

        n = len(self.modules)
        devs = {}
        for i in range(n):
            common = next(self.modules[i].parameters()).device
            if common is None or str(common) == "cpu":
                common = self.device_map[f"{self.model_decoder_layers}.{i}"]
                self.modules[i] = self.modules[i].to(common)
            devs[i] = str(common)

        execs = {d: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"awq-{d}")
                 for d in sorted(set(devs.values()))}
        in_flight = threading.Semaphore(len(execs) + 1)

        def work(i, named_linears, named_input_layers, input_feat):
            d = devs[i]
            if d.startswith("cuda"):
                torch.cuda.set_device(int(d.split(":")[1]))
            self._xtc_local.layer = get_op_name(self.model, self.modules[i])
            module_config = m.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs, self.scaling_layers)
            scales_list = []
            for layer in module_config:
                scales = self._search_best_scale(self.modules[i], **layer)
                if scales is not None:
                    scales_list.append(scales[:-1])
            m.apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat,
                          device=d, num_attention_heads=self.num_attention_heads,
                          num_key_value_heads=self.num_key_value_heads)
            clip_list = self._search_best_clip(named_linears, input_feat)
            m.apply_clip(self.modules[i], clip_list, d)
            self._apply_quant(named_linears)
            self.modules[i] = self.modules[i].to("cpu")
            m.clear_memory()
            import gc
            gc.collect()   # once per layer, instead of per grid step

        futs = []
        # The bar counts COMPLETED layers, not submitted ones: step 1 (forward)
        # is fast and would otherwise race the bar far ahead of the workers,
        # making progress (and any stop trigger) meaningless.
        bar = tqdm(total=n, desc="AWQ")
        bar_lock = threading.Lock()

        def _done(_f):
            in_flight.release()
            with bar_lock:
                bar.update(1)

        try:
            for i in range(n):
                d = devs[i]
                if d.startswith("cuda"):
                    torch.cuda.set_device(int(d.split(":")[1]))
                named_linears = m.get_named_quant_linears(self.modules[i])
                moe_input_layers = m.get_moe_layers(self.modules[i])
                named_input_layers = {**named_linears, **moe_input_layers}
                input_feat = self._get_input_feat(self.modules[i], named_input_layers)
                m.clear_memory()
                in_flight.acquire()
                fut = execs[d].submit(work, i, dict(named_linears),
                                      named_input_layers, input_feat)
                fut.add_done_callback(_done)
                futs.append(fut)
            for f in futs:
                f.result()
        finally:
            for ex in execs.values():
                ex.shutdown(wait=True)
            bar.close()

        self.model.config._attn_implementation = self.recover_attn_implementation

    return apply


def prepare(model_id, out_dir, env=None):
    """Build the cache from the environment and return it (not yet installed)."""
    env = env if env is not None else os.environ
    enabled = not ("QUARK_AWQ_CACHE" in env and not _truthy(env["QUARK_AWQ_CACHE"]))
    resume = not _truthy(env.get("QUARK_AWQ_NO_RESUME", "0"))
    force = _truthy(env.get("QUARK_AWQ_CACHE_FORCE", "0"))
    report = not ("QUARK_AWQ_REPORT" in env and not _truthy(env["QUARK_AWQ_REPORT"]))
    if not enabled:
        resume = False  # disabled means "run exactly as Quark does"

    path = env.get("QUARK_AWQ_CACHE_PATH") or str(Path(out_dir) / CACHE_NAME)
    cfg_hash, payload = compute_config_hash(
        model_id=model_id,
        recipe_path=env.get("AMD_AWQ_CONFIG"),
        env=env,
    )
    cache = AWQCache(path, cfg_hash, payload, write=enabled, resume=resume,
                     force=force, report=report)
    if not enabled:
        print("  [AWQ-CACHE] disabled (QUARK_AWQ_CACHE=0)")
    return cache


# ---------------------------------------------------------------------------
# Invariance gate: compare two checkpoints byte-for-byte.
# ---------------------------------------------------------------------------
def _h(t):
    import torch
    t = t.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def compare(path_a, path_b):
    a, b = torch_load(path_a), torch_load(path_b)
    fails = []
    if a.get("hash") != b.get("hash"):
        print(f"NOTE: config hashes differ ({a.get('hash')} vs {b.get('hash')}); "
              "comparing recorded tensors anyway")
    ka, kb = set(a.get("scales", {})), set(b.get("scales", {}))
    # Extra keys are not a mismatch: comparing a partial cache against a resumed
    # one is expected to have different sizes. Only shared keys must agree.
    if ka ^ kb:
        print(f"NOTE: caches differ in size ({len(ka)} vs {len(kb)} groups); "
              f"comparing the {len(ka & kb)} shared group(s)")
    diff = [k for k in ka & kb
            if a["scales"][k][0].shape != b["scales"][k][0].shape
            or _h(a["scales"][k][0]) != _h(b["scales"][k][0])]
    if diff:
        fails.append(f"{len(diff)} scale group(s) differ")
        for k in diff[:8]:
            print(f"  scale {k}")
    ca, cb = a.get("clips", {}), b.get("clips", {})
    for lp in set(ca) & set(cb):
        da = {n: _h(t) for n, t in ca[lp]}
        db = {n: _h(t) for n, t in cb[lp]}
        if da != db:
            fails.append(f"clip differs in {lp}")
            print(f"  clip {lp}: {da} != {db}")
    print(f"scales: {len(ka)} vs {len(kb)} groups; "
          f"clips: {len(ca)} vs {len(cb)} layers")
    if fails:
        print("FAILED:", "; ".join(fails))
        return 1
    print(f"INVARIANT -- {len(ka)} scale groups and {len(ca)} clip layers identical")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit("usage: awq_optimize.py compare <cache-a.pt> <cache-b.pt>")
