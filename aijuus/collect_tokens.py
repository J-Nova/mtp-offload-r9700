"""Token ID collector for generating draft_keep_file.

Usage:
1. Set RADIANCE_COLLECT_TOKENS=1 before starting vLLM.
2. Run traffic (benchmark).
3. The set is flushed periodically (RADIANCE_COLLECT_TOKENS_INTERVAL, default 20s),
   on SIGTERM/SIGINT, and at interpreter exit, to RADIANCE_COLLECT_TOKENS_FILE
   (default /tmp/kilo/draft_keep_our_model.json). Atomic write (tmp + os.replace),
   so a reader never sees a partial file and a SIGKILL loses at most one interval.

Only a process that has actually collected ids writes the file, so the API-server
process (which never runs the draft head) cannot clobber the EngineCore's file.

WHERE THE IDS COME FROM
-----------------------
On this stack the MTP draft token ids are produced by

    v1/spec_decode/llm_base_proposer.py::_greedy_sample

which calls ``self.model.compute_logits(hidden_states).argmax(dim=-1)`` by default
(``use_local_argmax_reduction`` defaults to False and is not set in the deployment
spec config). Only when that flag is enabled does it call ``get_top_tokens()``.
radiance_kernels already wraps ``get_top_tokens``; this module wraps
``compute_logits`` so the DEFAULT path is captured too. Both write the same set.
"""
import atexit
import json
import os
import signal
import threading
import time

# rlock: a signal can interrupt the main thread while it holds the lock and the
# handler then calls save() on the same thread -- a plain Lock would deadlock.
_lock = threading.RLock()
_collected_ids = set()
_has_data = False
_save_file = os.environ.get(
    "RADIANCE_COLLECT_TOKENS_FILE", "/tmp/kilo/draft_keep_our_model.json"
)
_enabled = os.environ.get("RADIANCE_COLLECT_TOKENS", "0") == "1"
try:
    _interval = float(os.environ.get("RADIANCE_COLLECT_TOKENS_INTERVAL", "20"))
except ValueError:
    _interval = 20.0
_thread = None
_prev_handlers = {}
_hooks_installed = False


def _iter_ints(x):
    """Yield ints from an int / flat or nested list / tensor (no numpy import)."""
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        yield x
        return
    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            return
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            yield v
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        else:
            try:
                yield int(v)
            except Exception:
                pass


def add_ids(ids):
    """Add token IDs to the collection. ids may be a tensor, list, or int."""
    global _has_data
    if not _enabled:
        return
    vals = list(_iter_ints(ids))
    if not vals:
        return
    with _lock:
        _collected_ids.update(vals)
        _has_data = True


def count():
    """Return the number of unique token IDs collected."""
    with _lock:
        return len(_collected_ids)


def save(path=None, force=False):
    """Atomically write the collected token IDs to a JSON file.

    No-op unless ids were collected (or force=True), so an engine-less process
    never writes an empty file over the engine's data.
    """
    path = path or _save_file
    with _lock:
        if not _has_data and not force:
            return None
        ids = sorted(_collected_ids)
        tmp = "%s.tmp.%d" % (path, os.getpid())
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(ids, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:
            print(
                "[RADIANCE_COLLECT_TOKENS] save failed: %r" % (e,), flush=True
            )
            return None
    print(
        "[RADIANCE_COLLECT_TOKENS] saved %d unique token IDs to %s"
        % (len(ids), path),
        flush=True,
    )
    return path


def reset():
    """Reset the collection."""
    global _has_data
    with _lock:
        _collected_ids.clear()
        _has_data = False


def enable():
    """Enable token collection and install the save hooks if not already done."""
    global _enabled
    _enabled = True
    ensure_installed()


def disable():
    """Disable token collection."""
    global _enabled
    _enabled = False


def _handle_signal(signum, frame):
    try:
        save()
    except Exception:
        pass
    # Chain to whatever handler was installed before us (vLLM installs its own
    # SIGTERM handler after plugin load, so in practice ours is replaced and the
    # periodic thread is the backstop; this covers the case where it is not).
    prev = _prev_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
    else:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _periodic():
    while True:
        time.sleep(_interval)
        try:
            save()
        except Exception:
            pass


def _install_hooks():
    """Register signal handlers, atexit, and the periodic-flush thread. Idempotent."""
    global _thread, _hooks_installed
    if not _enabled or _hooks_installed:
        return
    _hooks_installed = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            _prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except Exception:
            pass
    atexit.register(save)
    if _thread is None:
        _thread = threading.Thread(
            target=_periodic, name="radiance-token-save", daemon=True
        )
        _thread.start()


def _capturing():
    """True if a CUDA/HIP graph is being captured (do not sync to host then)."""
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _install_model_hooks():
    """Wrap the MTP draft head's compute_logits to record each step's argmax ids.

    compute_logits returns [num_tokens, vocab] draft logits; argmax(-1) is exactly
    the proposed draft token id on the default (non local-argmax) path. Wrapped on
    the concrete classes so it also applies in the EngineCore process.
    """
    targets = []
    for mod_name, cls_name in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP"),
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MoeMTP"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            targets.append(getattr(mod, cls_name))
        except Exception:
            continue

    for cls in targets:
        # A wrapped base class sets this attribute; the subclass then inherits the
        # wrapper and the flag, so it is correctly skipped (no double wrap).
        if getattr(cls, "_radiance_token_cl_wrapped", False):
            continue
        orig = cls.__dict__.get("compute_logits")
        if orig is None:
            # Inherited from an already-wrapped target; nothing to do.
            continue

        def wrapped_compute(self, hidden_states, *args, _orig=orig, **kwargs):
            out = _orig(self, hidden_states, *args, **kwargs)
            if out is not None and not _capturing():
                try:
                    add_ids(out.argmax(dim=-1))
                except Exception:
                    pass
            return out

        cls.compute_logits = wrapped_compute
        cls._radiance_token_cl_wrapped = True


def _install_get_top_tokens_hooks():
    """Wrap the MTP draft head's get_top_tokens (the local-argmax / W4 reduction path).

    The compute_logits hook above covers the default path; this covers the reduction path, with the
    SAME class list and the same _capturing() guard + error swallowing, so the two hooks cannot
    diverge. Wrapped on the concrete classes so it also applies in the EngineCore process.
    """
    for mod_name, cls_name in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP"),
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MoeMTP"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        if getattr(cls, "_radiance_token_gt_wrapped", False):
            continue
        orig = getattr(cls, "get_top_tokens", None)
        if orig is None:
            continue

        def wrapped_topk(self, hidden_states, _orig=orig):
            out = _orig(self, hidden_states)
            if out is not None and not _capturing():
                try:
                    add_ids(out)
                except Exception:
                    pass
            return out

        cls.get_top_tokens = wrapped_topk
        cls._radiance_token_gt_wrapped = True


def ensure_installed():
    """Install every collector hook. Idempotent single entry point.

    radiance_kernels delegates here so there is only one class list and one set of guards.
    """
    if not _enabled:
        return
    _install_hooks()
    _install_model_hooks()
    _install_get_top_tokens_hooks()


if _enabled:
    ensure_installed()
