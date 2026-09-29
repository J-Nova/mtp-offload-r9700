#!/usr/bin/env python3
"""FORK-LOCAL runtime patch: server-wide degenerate-loop detection (degen).

Ports tcclaviger's `patches/degen_detect` (vLLM 0.29.0.dev0+g2bdbbc8080) onto our baked vLLM
0.29.0. Once per appended output token the scheduler checks for a periodic tail and finishes the
request with finish_reason "repetition" when found. Server-wide defaults (CLI --degen-*, or the
SchedulerConfig fields): max_period 100, min_repeats 6, min_span 128; `--degen-max-period 0`
disables it.

Pure Python source edits, applied at container start by the entrypoint (no image rebuild).
All-or-nothing: every anchor is resolved in memory first, then the files are written. Idempotent:
an edit whose marker is already present is skipped. If any anchor is missing the script applies
nothing and exits non-zero (stock behaviour unchanged; the entrypoint warns and continues).

Env: RADIANCE_LOCAL_DEGEN=0 skips entirely.
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
        return  # already applied on a previous boot
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"anchor not found/unique in {path} (count={n})")
    _buf[path] = src.replace(old, new, 1)


def _flush():
    for path, src in _buf.items():
        (SP / path).write_text(src)


DEGEN_FUNCS = '''
# FORK-LOCAL (patches/degen_detect): server-wide degenerate-loop detection.
class DegenParams(NamedTuple):
    """Degenerate-loop thresholds; max_period == 0 disables detection."""

    max_period: int
    min_repeats: int
    min_span: int


def check_degeneration(request: Request, params: DegenParams) -> bool:
    """Call exactly once per appended output token, in order."""
    max_period = params.max_period
    if max_period <= 0:
        return False

    output = request.output_token_ids
    n = len(output)
    if n < 2:
        return False

    counter = request.degen_counter
    if counter is None:
        counter = request.degen_counter = [0] * (max_period + 1)

    last = output[-1]
    min_span = params.min_span
    min_repeats = params.min_repeats
    max_p = min(max_period, n - 1)
    for p in range(1, max_p + 1):
        if output[-1 - p] == last:
            c = counter[p] + 1
            counter[p] = c
            if c >= min_span and c >= min_repeats * p:
                return True
        else:
            counter[p] = 0
    return False


'''


def apply():
    # 1) request field
    _edit(
        "vllm/v1/request.py",
        "        self.cache_salt: str | None = cache_salt\n",
        "        self.cache_salt: str | None = cache_salt\n"
        "\n"
        "        # FORK-LOCAL (patches/degen_detect):\n"
        "        # degen_counter[p] = consecutive trailing tokens with token[t] == token[t-p].\n"
        "        self.degen_counter: list[int] | None = None\n",
        marker="self.degen_counter: list[int] | None = None",
    )
    # 2) scheduler detection helpers
    _edit(
        "vllm/v1/core/sched/utils.py",
        "from vllm.v1.request import Request, RequestStatus\n",
        "from typing import NamedTuple\n"
        "\n"
        "from vllm.v1.request import Request, RequestStatus\n"
        + DEGEN_FUNCS,
        marker="def check_degeneration(",
    )
    # 3) check_stop signature + tail
    _edit(
        "vllm/v1/core/sched/utils.py",
        "def check_stop(request: Request, max_model_len: int) -> bool:\n",
        "def check_stop(\n"
        "    request: Request,\n"
        "    max_model_len: int,\n"
        "    degen_params: DegenParams | None = None,\n"
        ") -> bool:\n",
        marker="degen_params: DegenParams | None = None,",
    )
    _edit(
        "vllm/v1/core/sched/utils.py",
        '        request.status = RequestStatus.FINISHED_REPETITION\n'
        '        request.stop_reason = "repetition_detected"\n'
        "        return True\n"
        "\n"
        "    return False\n",
        '        request.status = RequestStatus.FINISHED_REPETITION\n'
        '        request.stop_reason = "repetition_detected"\n'
        "        return True\n"
        "\n"
        "    # FORK-LOCAL (patches/degen_detect)\n"
        "    if degen_params is not None and check_degeneration(request, degen_params):\n"
        "        request.status = RequestStatus.FINISHED_REPETITION\n"
        '        request.stop_reason = "repetition_detected"\n'
        "        return True\n"
        "\n"
        "    return False\n",
        marker="if degen_params is not None and check_degeneration",
    )
    # 4) scheduler hook
    _edit(
        "vllm/v1/core/sched/scheduler.py",
        "from vllm.v1.core.sched.utils import check_stop, remove_all\n",
        "from vllm.v1.core.sched.utils import DegenParams, check_stop, remove_all\n",
        marker="DegenParams, check_stop, remove_all",
    )
    _edit(
        "vllm/v1/core/sched/scheduler.py",
        "        self.max_model_len = vllm_config.model_config.max_model_len\n",
        "        self.max_model_len = vllm_config.model_config.max_model_len\n"
        "        # FORK-LOCAL (patches/degen_detect)\n"
        "        self.degen_params = DegenParams(\n"
        "            max_period=self.scheduler_config.degen_max_period,\n"
        "            min_repeats=self.scheduler_config.degen_min_repeats,\n"
        "            min_span=self.scheduler_config.degen_min_span,\n"
        "        )\n",
        marker="self.degen_params = DegenParams(",
    )
    _edit(
        "vllm/v1/core/sched/scheduler.py",
        "stopped = check_stop(request, self.max_model_len)\n",
        "stopped = check_stop(request, self.max_model_len, self.degen_params)\n",
        marker="self.max_model_len, self.degen_params)",
    )
    # 5) config fields
    _edit(
        "vllm/config/scheduler.py",
        '    is_multimodal_model: bool = False\n    """True if the model is multimodal."""\n',
        '    is_multimodal_model: bool = False\n    """True if the model is multimodal."""\n'
        "\n"
        "    # FORK-LOCAL (patches/degen_detect): server-wide degenerate-loop detection.\n"
        "    degen_max_period: int = Field(default=100, ge=0)\n"
        '    """Largest period checked for a periodic output tail; 0 disables detection."""\n'
        "\n"
        "    degen_min_repeats: int = Field(default=6, ge=2)\n"
        '    """Minimum full cycles of period p before a trigger."""\n'
        "\n"
        "    degen_min_span: int = Field(default=128, ge=1)\n"
        '    """Period p triggers after max(degen_min_span, degen_min_repeats * p) tokens."""\n',
        marker="degen_max_period: int = Field(default=100",
    )
    # 6) EngineArgs fields
    _edit(
        "vllm/engine/arg_utils.py",
        "    watermark: float = SchedulerConfig.watermark\n",
        "    watermark: float = SchedulerConfig.watermark\n"
        "\n"
        "    # FORK-LOCAL (patches/degen_detect)\n"
        "    degen_max_period: int = SchedulerConfig.degen_max_period\n"
        "    degen_min_repeats: int = SchedulerConfig.degen_min_repeats\n"
        "    degen_min_span: int = SchedulerConfig.degen_min_span\n",
        marker="degen_max_period: int = SchedulerConfig.degen_max_period",
    )
    # 7) CLI options
    _edit(
        "vllm/engine/arg_utils.py",
        '            "--stream-interval", **scheduler_kwargs["stream_interval"]\n        )\n',
        '            "--stream-interval", **scheduler_kwargs["stream_interval"]\n        )\n'
        "        # FORK-LOCAL (patches/degen_detect)\n"
        "        scheduler_group.add_argument(\n"
        '            "--degen-max-period", **scheduler_kwargs["degen_max_period"]\n'
        "        )\n"
        "        scheduler_group.add_argument(\n"
        '            "--degen-min-repeats", **scheduler_kwargs["degen_min_repeats"]\n'
        "        )\n"
        "        scheduler_group.add_argument(\n"
        '            "--degen-min-span", **scheduler_kwargs["degen_min_span"]\n'
        "        )\n",
        marker='"--degen-max-period"',
    )
    # 8) SchedulerConfig construction
    _edit(
        "vllm/engine/arg_utils.py",
        "            stream_interval=self.stream_interval,\n",
        "            stream_interval=self.stream_interval,\n"
        "            # FORK-LOCAL (patches/degen_detect)\n"
        "            degen_max_period=self.degen_max_period,\n"
        "            degen_min_repeats=self.degen_min_repeats,\n"
        "            degen_min_span=self.degen_min_span,\n",
        marker="degen_max_period=self.degen_max_period,",
    )


def main():
    if os.environ.get("RADIANCE_LOCAL_DEGEN", "1") != "1":
        print("[patch_degen] disabled (RADIANCE_LOCAL_DEGEN != 1)")
        return
    try:
        apply()
        _flush()
    except Exception as e:
        sys.stderr.write(f"[patch_degen] FAILED, nothing applied: {e!r}\n")
        sys.exit(1)
    print("[patch_degen] applied (server-wide degenerate-loop detection)")


if __name__ == "__main__":
    main()
