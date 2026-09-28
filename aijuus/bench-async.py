#!/usr/bin/env python3
"""BetterBench-driven serving benchmark for the knobs we tune in this deployment.

Measurement is delegated to BetterBench (GGZ14/BetterBench): one `betterbench run`
per target covers its percentile single-stream decode, prompt-processing (prefill)
depth sweep and concurrency sweep. This tool keeps the parts BetterBench does not
do -- it launches one plain (offload-free) vLLM per target via serve-mxfp4.sh,
picks the widest-link GPU, waits for /health, streams container warnings into one
log, samples GPU busy, tears everything down -- then folds every target's
results.json into ONE combined overview.json for side-by-side reading.

Scenarios (--phases), as BetterBench names them:
  decode       single-stream (batch = 1) percentile decode per category
  prefill      prompt-processing throughput vs input depth (2K..64K), cold cache
  concurrency  aggregate throughput / TTFT percentiles at increasing load
  all          all three
`pressure` is accepted as an alias for `concurrency`. The old cold/hit/evict/restore
phases are gone: BetterBench has no cache-tier model, so offload/cache-hit timing is
not measured here anymore.

Targets (--target, repeatable; JSON), same shape as before:
  {"name":"gpu1","base":"http://vllm-1:8001/v1","model":"Qwen3.8-A",
   "prom_filter":"model_name=\"Qwen3.8-A\"","instance":"1"}

Run it against a deploy on the compose network (services are expose-only):

  docker run --rm --network coolify -v "$PWD":/work -w /work python:3.13-alpine \
    sh -c 'pip install betterbench && \
    python3 bench-async.py --phases all --label two-model --save-dir /work/bench \
      --target '"'"'{"name":"gpu0","base":"http://vllm-0:8000/v1","model":"Qwen3.8-A"}'"'"' \
      --target '"'"'{"name":"gpu1","base":"http://vllm-1:8001/v1","model":"Qwen3.8-B"}'"'"''

Or let it stand up its own PLAIN vLLM per GPU on the host -- one model each, no shared
disk, no offload, no load balancer (the "just a plain vLLM" mode):

  ./bench-async.py --launch --phases all --label plain --save-dir /work/bench --models ~/models \
    --target '{"name":"gpu1","port":8001,"model":"Qwen3.8-A","snap":"/models/Qwen3.8-A","spec_method":"dflash"}'

--launch shells out to serve-mxfp4.sh per target (NAME, PORT, GPUS, TP=1, SNAP, DRAFTER,
SPEC_METHOD, SPEC, MAXLEN, MAXSEQS, CHUNK, KV_MEM, GPU_UTIL, SERVED_NAMES), waits for
/health, benchmarks each instance, then removes the containers unless --keep.

Offload is OFF by default (plain vLLM). Turn it on with --offload-gib N, optionally
--offload-disk-host DIR for the shared fs secondary tier, plus --offload-policy lru|arc,
--offload-read-threads, --offload-write-threads, --offload-head-cap. Because offload KV
state is MODEL-SPECIFIC, offload mode allows exactly ONE --target; to compare two
models, leave offload off. --offload-disk-host also starts the mandatory fs-tier reaper
for the window (--reaper auto: systemd kvcache-reap.timer if installed, else a loop).

BetterBench invocation: found on PATH as `betterbench`, else $BETTERBENCH_BIN, else
~/betterbench/.venv/bin/betterbench, else `python3 -m betterbench.cli`. Override the
whole command with --betterbench. Pass --passes/--warmup/--quick, --bb-config,
--bb-corpus/--bb-categories and --max-model-len through to it; --note KEY=VALUE is
recorded in its results.json (and a few are auto-added from the target spec).

Every run writes one combined overview.json (override with --overview): the run
metadata, each target's config + BetterBench env fingerprint + aggregated per-phase
metrics, and a `comparison` list of cells keyed by target name. Per-target BetterBench
results.json + HTML report are kept beside it. Re-run the same command and compare:

  python3 bench-async.py --compare /work/bench/gpu0.json /work/bench/gpu1.json

BetterBench also ships `betterbench ab` for interleaved, drift-cancelled A/B; this tool
is for reading many targets side by side, not for resolving sub-percent deltas.

No third-party Python deps here (stdlib only); BetterBench itself is invoked as a
subprocess and needs numpy, so install it (`pip install betterbench`) wherever this runs.
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings as _warnings

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None


class Progress:
    """Overall benchmark progress.

    Uses tqdm when importable; otherwise draws a simple carriage-return bar on a
    TTY, and falls back to plain `[progress] n/total` lines when not a TTY.
    """

    def __init__(self, total: int, desc: str = "benchmark"):
        self.total = total
        self.n = 0
        self.desc = desc
        self._bar = None
        self._tty = False
        if _tqdm is not None and total > 0:
            self._bar = _tqdm(total=total, desc=desc, position=0, leave=True,
                              dynamic_ncols=True)
        elif total > 0 and sys.stderr.isatty():
            self._tty = True

    def step(self, desc: str | None = None) -> None:
        self.n += 1
        if desc:
            self.desc = desc
        if self._bar is not None:
            self._bar.set_description(self.desc)
            self._bar.update(1)
        elif self._tty:
            self._draw()
        else:
            print(f"[progress] {self.n}/{self.total} {self.desc}", flush=True)

    def _draw(self) -> None:
        width = 28
        filled = int(width * self.n / self.total) if self.total else width
        bar = "#" * filled + "-" * (width - filled)
        end = "\n" if self.n >= self.total else ""
        sys.stderr.write(f"\r{self.desc[:24]:<24} [{bar}] {self.n}/{self.total}{end}")
        sys.stderr.flush()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def debug(args, msg: str) -> None:
    if getattr(args, "debug", False):
        print(f"[debug] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Warnings log. Every container WARNING/ERROR/CRITICAL line, the bench's own
# warnings, and in-process Python warnings are appended to a single file so the
# run can be inspected afterwards.
# ---------------------------------------------------------------------------
_WARN = {"path": None, "fh": None, "lock": threading.Lock()}
_WARN_RE = re.compile(r"\b(WARNING|WARN|ERROR|CRITICAL|FATAL)\b")


def warn_log_init(path: str) -> None:
    _WARN["path"] = path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    _WARN["fh"] = open(path, "w")
    _WARN["fh"].write(f"# warnings log {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    _WARN["fh"].flush()
    atexit.register(warn_log_close)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        warn(f"python {category.__name__}: {message} "
             f"({os.path.basename(filename)}:{lineno})")

    _warnings.showwarning = _showwarning


def _warn_write(target: str, msg: str) -> None:
    with _WARN["lock"]:
        fh = _WARN["fh"]
        if fh is None or fh.closed:
            return
        prefix = f"[{target}] " if target else ""
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {prefix}{msg}\n")
        fh.flush()


def warn(msg: str, target: str = "") -> None:
    """Record a bench warning and show it on stderr."""
    _warn_write(target, msg)
    print(f"[warn] {('[' + target + '] ') if target else ''}{msg}",
          file=sys.stderr, flush=True)


def warn_file(target: str, msg: str) -> None:
    """Record a line from a container's logs without echoing it again."""
    _warn_write(target, msg)


def warn_log_close() -> None:
    with _WARN["lock"]:
        if _WARN["fh"] is not None and not _WARN["fh"].closed:
            _WARN["fh"].close()


# ---------------------------------------------------------------------------
# Phase model + BetterBench integration.
# ---------------------------------------------------------------------------
ALL_PHASES = ["decode", "prefill", "concurrency"]
PHASE_ALIASES = {"pressure": "concurrency"}
_REMOVED_PHASES = {
    "cold": "the unique-prompt prefill baseline is now BetterBench's `prefill` phase",
    "hit": "prefix-cache hit latency is not measured by BetterBench",
    "evict": "cache eviction policy is not measured by BetterBench",
    "restore": "offload / disk restore is not measured by BetterBench",
}


def resolve_phases(spec: str) -> list[str]:
    if spec.strip() in ("", "all"):
        return list(ALL_PHASES)
    out: list[str] = []
    for raw in spec.split(","):
        p = raw.strip()
        if not p:
            continue
        if p in _REMOVED_PHASES:
            raise SystemExit(f"phase {p!r} was removed: {_REMOVED_PHASES[p]}. "
                             f"Use one of {', '.join(ALL_PHASES)} (or all).")
        if p in PHASE_ALIASES:
            print(f"[phases] {p!r} is BetterBench's concurrency sweep; using 'concurrency'")
            p = PHASE_ALIASES[p]
        if p not in ALL_PHASES:
            raise SystemExit(f"unknown phase {p!r}; choose from "
                             f"{', '.join(ALL_PHASES)} (or all)")
        if p not in out:
            out.append(p)
    if not out:
        raise SystemExit("no phases selected")
    return out


def resolve_betterbench(explicit: str) -> list[str]:
    """Locate the betterbench entry point: flag, env, PATH, known venv, module."""
    if explicit:
        return shlex.split(explicit)
    env = os.environ.get("BETTERBENCH_BIN")
    if env:
        return shlex.split(env)
    found = shutil.which("betterbench")
    if found:
        return [found]
    for cand in (os.path.expanduser("~/betterbench/.venv/bin/betterbench"),):
        if os.path.exists(cand):
            return [cand]
    return [sys.executable, "-m", "betterbench.cli"]


def _default_bb_out(args, target: dict) -> str:
    name = f"{target['name']}.betterbench.json"
    if args.bb_out:
        return args.bb_out
    if args.save_dir:
        return os.path.join(args.save_dir, name)
    if args.save:
        return re.sub(r"\.json$", f".{target['name']}.betterbench.json", args.save)
    return os.path.join(os.getcwd(), name)


def _target_notes(target: dict, user_notes: list[str]) -> list[str]:
    """Target-spec metadata as BetterBench notes, plus the user's --note values."""
    notes = list(user_notes or [])
    for field, key in (("spec_method", "spec_method"), ("spec", "spec"),
                       ("maxlen", "maxlen"), ("maxseqs", "maxseqs"), ("tp", "tp"),
                       ("kv_mem", "kv_mem"), ("gpu_util", "gpu_util")):
        if target.get(field):
            notes.append(f"{key}={target[field]}")
    return notes


def run_betterbench(args, target: dict, phases: list[str]) -> tuple[dict, str]:
    """Run one `betterbench run` for this target and return (results, path).

    One invocation covers every requested phase; BetterBench writes results.json
    plus an HTML report beside it and we read the JSON back. stdout/stderr stream
    straight through so the run's own banner and markdown tables are visible.
    """
    cmd = resolve_betterbench(args.betterbench)
    out = _default_bb_out(args, target)
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)

    argv = cmd + ["run", "--endpoint", target["base"], "--model", target["model"]]
    for p in phases:
        argv.append("--" + p)
    if args.passes is not None:
        argv += ["--passes", str(args.passes)]
    if args.warmup is not None:
        argv += ["--warmup", str(args.warmup)]
    if args.quick:
        argv += ["--quick"]
    if args.bb_config:
        argv += ["--config", args.bb_config]
    if args.bb_corpus:
        argv += ["--corpus", args.bb_corpus]
    if args.bb_categories:
        argv += ["--categories", *args.bb_categories]
    if args.max_model_len is not None:
        argv += ["--max-model-len", str(args.max_model_len)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    argv += ["--name", args.bb_name or target["name"]]
    for note in _target_notes(target, args.note):
        argv += ["--note", note]
    if args.no_html:
        argv += ["--no-html"]
    argv += ["--out", out, "--no-update-check"]

    print(f"[betterbench {target['name']}] " + " ".join(shlex.quote(a) for a in argv),
          flush=True)
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"betterbench exited {proc.returncode}")
    if not os.path.exists(out):
        raise RuntimeError(f"betterbench wrote no results at {out}")
    with open(out) as fh:
        return json.load(fh), out


# ---------------------------------------------------------------------------
# Small stats helpers (stdlib only, so this file stays dependency-free).
# ---------------------------------------------------------------------------
def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))) )]


def _nums(recs, key):
    return [r[key] for r in recs
            if isinstance(r.get(key), (int, float)) and r.get(key) is not None]


def _gaps(recs):
    out: list[float] = []
    for r in recs:
        out.extend(r.get("update_gaps_ms") or [])
    return out


def summarize_single_stream(bb: dict) -> dict:
    ss = bb.get("single_stream") or {}
    weights = (bb.get("config") or {}).get("weights") or {}
    cats: dict[str, dict] = {}
    comb_num = comb_den = 0.0
    for cat, recs in ss.items():
        ok = [r for r in recs if r.get("ok")]
        tps = _nums(ok, "decode_tps")
        ttfts = _nums(ok, "ttft_ms")
        gaps = _gaps(ok)
        tpu = _nums(ok, "tokens_per_update")
        med_tps = statistics.median(tps) if tps else float("nan")
        cats[cat] = {
            "n": len(recs), "n_ok": len(ok),
            "decode_tps_median": med_tps,
            "ttft_p50_ms": pct(ttfts, 0.50), "ttft_p99_ms": pct(ttfts, 0.99),
            "update_p50_ms": pct(gaps, 0.50), "update_p99_ms": pct(gaps, 0.99),
            "tokens_per_update": statistics.median(tpu) if tpu else float("nan"),
            "batched": any(r.get("chunking") == "batched" for r in ok),
        }
        w = weights.get(cat)
        if w and med_tps == med_tps:
            comb_num += w * med_tps
            comb_den += w
    ok_all = [r for recs in ss.values() for r in recs if r.get("ok")]
    all_ttft = _nums(ok_all, "ttft_ms")
    all_gaps = _gaps(ok_all)
    return {
        "categories": cats,
        "combined_decode_tps": (comb_num / comb_den) if comb_den else float("nan"),
        "ttft_p50_ms": pct(all_ttft, 0.50), "ttft_p99_ms": pct(all_ttft, 0.99),
        "update_p50_ms": pct(all_gaps, 0.50), "update_p99_ms": pct(all_gaps, 0.99),
        "n": sum(len(v) for v in ss.values()), "n_ok": len(ok_all),
    }


def summarize_prefill(bb: dict) -> list[dict]:
    rows = []
    for d in bb.get("prefill") or []:
        if d.get("skipped"):
            rows.append({"target_depth": d.get("target_depth"), "skipped": True,
                         "reason": d.get("reason")})
            continue
        pp = d.get("pp_tps") or []
        ttft = d.get("ttft_ms") or []
        pt = d.get("prompt_tokens") or []
        rows.append({
            "target_depth": d.get("target_depth"), "skipped": False,
            "pp_tps_median": statistics.median(pp) if pp else float("nan"),
            "ttft_p50_ms": pct(ttft, 0.50),
            "prompt_tokens_median": statistics.median(pt) if pt else float("nan"),
            "n": len(pp),
        })
    return rows


def summarize_concurrency(bb: dict) -> list[dict]:
    rows = []
    for lvl in bb.get("concurrency") or []:
        ttft = lvl.get("ttft_ms") or []
        dec = lvl.get("decode_tps") or []
        rows.append({
            "level": lvl.get("level"), "requests": lvl.get("requests"),
            "ok": lvl.get("ok"), "wall_s": lvl.get("wall_s"),
            "aggregate_tps": lvl.get("aggregate_tps"),
            "ttft_p50_ms": pct(ttft, 0.50), "ttft_p99_ms": pct(ttft, 0.99),
            "decode_tps_median": statistics.median(dec) if dec else float("nan"),
        })
    return rows


def summarize_result(result: dict) -> dict:
    """Aggregate a target's BetterBench results into per-phase summaries."""
    bb = result.get("phases") or {}
    out: dict[str, dict] = {}
    if bb.get("single_stream"):
        out["decode"] = summarize_single_stream(bb)
    if bb.get("prefill"):
        out["prefill"] = summarize_prefill(bb)
    if bb.get("concurrency"):
        out["concurrency"] = summarize_concurrency(bb)
    return out


# ---------------------------------------------------------------------------
# GPU busy sampling (host sysfs, or a Prometheus gpu-exporter). BetterBench does
# not sample GPU, so this runs across the whole target measurement as one
# aggregate rather than per phase.
# ---------------------------------------------------------------------------
def sample_gpu(gpu_url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(gpu_url, timeout=10.0) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return {}
    vals = {}
    for line in text.splitlines():
        if line.startswith("amd_gpu_busy_percent"):
            m = re.match(r'amd_gpu_busy_percent\{gpu="([^"]+)"\}\s+([0-9.eE+-]+)', line)
            if m:
                try:
                    vals[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return vals


def sample_gpu_sysfs() -> dict[str, float]:
    vals = {}
    for path in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        card = path.split("/")[4]
        try:
            with open(path) as fh:
                vals[card] = float(fh.read().strip())
        except (OSError, ValueError):
            continue
    return vals


class GpuSampler:
    def __init__(self, source: str, url: str, interval: float):
        self.source, self.url, self.interval = source, url, interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                s = sample_gpu(self.url) if self.source == "prom" else sample_gpu_sysfs()
                if s:
                    self.samples.append(s)
                self._stop.wait(self.interval)
        if self.source == "prom" or os.path.isdir("/sys/class/drm"):
            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def summary(self) -> dict:
        out = {}
        for card in sorted({c for s in self.samples for c in s}):
            vals = [s[card] for s in self.samples if card in s]
            if vals:
                out[card] = {
                    "mean_busy": sum(vals) / len(vals),
                    "idle_frac": sum(1 for v in vals if v < 90.0) / len(vals),
                    "samples": len(vals),
                }
        return out


def prom_query(prom: str, expr: str, timeout: float = 30.0):
    url = prom.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        obj = json.load(resp)
    if obj.get("status") != "success":
        raise RuntimeError(f"prometheus query failed: {obj}")
    return obj["data"]["result"]


def collect_labels(prom: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        for r in prom_query(prom, "radiance_serve_info"):
            metric = dict(r["metric"])
            out[str(metric.get("instance", len(out)))] = metric
    except Exception as exc:  # noqa: BLE001
        warn(f"radiance_serve_info: {exc}")
    return out


# ---------------------------------------------------------------------------
# Per-target execution.
# ---------------------------------------------------------------------------
def execute_target(args, target: dict, phases: list[str]) -> dict:
    args.base = target["base"]
    args.model = target["model"]
    args.prom_filter = target.get("prom_filter", "")

    inst = str(target.get("instance", ""))
    base_cfg: dict = {}
    if args.prom:
        labels = collect_labels(args.prom)
        base_cfg = labels.get(inst) or (next(iter(labels.values())) if labels else {})
    config = {k: base_cfg.get(k) for k in
              ("instance", "async_sched", "offload_policy", "offload_mode",
               "read_threads", "write_threads", "fanout_max", "kv_source",
               "spec_method", "spec_tokens", "max_num_seqs")}
    config["instance"] = inst or target["name"]
    config["model"] = args.model
    config["betterbench"] = " ".join(resolve_betterbench(args.betterbench))
    if args.launch:
        config["offload_mode"] = "off"

    src = args.gpu_source if args.gpu_source in ("prom", "sysfs") else "sysfs"
    print(f"\n=== target {target['name']} -> {args.base} "
          f"(model={args.model}, phases={','.join(phases)}, gpu_busy={src}) ===")
    print("active serve config: " + json.dumps(config))
    args._target_name = target["name"]
    debug(args, f"phases={phases} betterbench={config['betterbench']}")

    with GpuSampler(src, args.gpu, args.gpu_interval) as sampler:
        bb, bb_path = run_betterbench(args, target, phases)
    gpu = sampler.summary()

    return {
        "target": target,
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config,
        "workload": {
            "phases": phases,
            "betterbench_version": bb.get("betterbench_version"),
            "corpus_version": bb.get("corpus_version"),
            "betterbench_config": bb.get("config"),
        },
        "phases": bb,
        "betterbench_path": bb_path,
        "gpu": gpu,
    }


def parse_target(spec: str, default_model: str, launch: bool = False) -> dict:
    obj = json.loads(spec)
    name = obj.get("name") or (f"gpu{obj['gpu']}" if obj.get("gpu") is not None
                               else f"target{obj.get('port', '')}")
    port = str(obj.get("port", ""))
    base = obj.get("base") or (f"http://localhost:{port}/v1" if port else "")
    if not base and not launch:
        raise SystemExit(f"--target needs a base URL (or a port with --launch): {spec}")
    # gpu is optional: --launch always selects the optimal card automatically.
    return {
        "name": name or base,
        "base": base.rstrip("/"),
        "model": obj.get("model", default_model),
        "prom_filter": obj.get("prom_filter", ""),
        "instance": str(obj.get("instance", "")),
        "gpu": obj.get("gpu"),
        "port": port,
        "tp": obj.get("tp", ""),
        "snap": obj.get("snap", ""),
        "drafter": obj.get("drafter", ""),
        "served": obj.get("served", ""),
        "spec_method": obj.get("spec_method", ""),
        "spec": obj.get("spec", ""),
        "maxlen": obj.get("maxlen", ""),
        "maxseqs": obj.get("maxseqs", ""),
        "chunk": obj.get("chunk", ""),
        "kv_mem": obj.get("kv_mem", ""),
        "gpu_util": obj.get("gpu_util", ""),
    }


def detect_runtime(preferred: str = "") -> str:
    if preferred:
        return preferred
    for cand in ("podman", "docker"):
        if shutil.which(cand):
            return cand
    return "docker"


# ---------------------------------------------------------------------------
# Card selection. On a box where one GPU is on a x1 link and another on x16, the
# benchmark must always use the wide one; two ~20 GiB models cannot share a
# single 32 GiB card, so targets run sequentially on the same chosen GPU.
# ---------------------------------------------------------------------------
def _sysfs_read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _link_speed_gt(s: str) -> float:
    m = re.match(r"\s*([0-9.]+)", s or "")
    return float(m.group(1)) if m else 0.0


def _pcie_bottleneck(bdf: str) -> tuple[int, float]:
    """Widest/slowest link on the path from the GPU up to the root complex.

    A GPU behind a PCIe switch reports x16 at its endpoint even when the
    switch's host link is x1; the minimum width along the path is the real one.
    """
    widths: list[int] = []
    speeds: list[float] = []
    cur = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    seen = set()
    while cur and cur not in seen and os.path.exists(os.path.join(cur, "vendor")):
        seen.add(cur)
        w = _sysfs_read(os.path.join(cur, "current_link_width"))
        if w.isdigit():
            widths.append(int(w))
        speeds.append(_link_speed_gt(_sysfs_read(os.path.join(cur, "current_link_speed"))))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return (min(widths) if widths else 0, max(speeds) if speeds else 0.0)


def list_rocm_gpus() -> list[tuple[int, str]]:
    """[(hip_index, pci_bdf)] from rocm-smi, else lspci ordering."""
    smi = shutil.which("rocm-smi")
    pairs: list[tuple[int, str]] = []
    if smi:
        try:
            out = subprocess.run([smi, "--showbus"], capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:  # noqa: BLE001
            out = ""
        rx = re.compile(r"GPU\[(\d+)\].*?([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])")
        for line in out.splitlines():
            m = rx.search(line)
            if m:
                pairs.append((int(m.group(1)), m.group(2)))
    if not pairs:
        lspci = shutil.which("lspci")
        if lspci:
            try:
                out = subprocess.run([lspci, "-D", "-nn", "-d", "1002:"],
                                     capture_output=True, text=True, timeout=10).stdout
            except Exception:  # noqa: BLE001
                out = ""
            bdfs = [ln.split()[0] for ln in out.splitlines()
                    if "[0300]" in ln or "[0380]" in ln]
            pairs = list(enumerate(sorted(bdfs)))
    return sorted(pairs)


def pick_best_card(override=None):
    """Return (hip_index, ranking) for the widest-link GPU."""
    if override not in (None, ""):
        try:
            return int(override), None
        except (TypeError, ValueError):
            raise SystemExit(f"--card must be a HIP index, got {override!r}")
    gpus = list_rocm_gpus()
    if not gpus:
        return None, None
    ranking = []
    for idx, bdf in gpus:
        width, speed = _pcie_bottleneck(bdf)
        ranking.append({"gpu": idx, "bdf": bdf, "width": width, "speed": speed})
    ranking.sort(key=lambda r: (-r["width"], -r["speed"], r["gpu"]))
    return ranking[0]["gpu"], ranking


def report_card(best: int, ranking) -> None:
    if not ranking:
        print(f"[card] using GPU {best} (explicit --card)")
        return
    top = next(r for r in ranking if r["gpu"] == best)
    print(f"[card] optimal card: GPU {best} ({top['bdf']}) host-link width "
          f"x{top['width']} (PCIe {top['speed']:g} GT/s)")
    for r in ranking:
        mark = "  <- selected" if r["gpu"] == best else ""
        print(f"[card]   GPU {r['gpu']} {r['bdf']}: x{r['width']} "
              f"(PCIe {r['speed']:g} GT/s){mark}")


class ContainerLogTailer:
    """Stream one container's logs into the bench output, prefixed by target.

    `docker logs -f` prints the existing log then follows, so nothing produced
    before the tailer starts is lost. By default the per-line tqdm / weight-load
    spam is dropped; --verbose-logs keeps it.
    """

    def __init__(self, runtime: str, container: str, name: str, width: int = 0,
                 verbose: bool = False):
        self.runtime = runtime
        self.container = container
        self.name = name
        self.width = width
        self.verbose = verbose
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def _is_noise(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if "Loading weights:" in s or "Loading safetensors" in s:
            return True
        if "%|" in s and "it/s" in s:
            return True
        return False

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [self.runtime, "logs", "-f", "--tail", "all", self.container],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        tag = f"[{self.name}]".ljust(self.width + 2)
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = _ANSI.sub("", line.rstrip("\r\n"))
            if _WARN_RE.search(line):
                warn_file(self.name, line)   # every container warning/error line
            if not self.verbose and self._is_noise(line):
                continue
            print(f"{tag} {line}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def port_in_use(port: str, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, int(port))) == 0
    except (OSError, ValueError):
        return False


def list_containers(runtime: str, needle: str = "") -> list[str]:
    try:
        out = subprocess.run([runtime, "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = out.stdout.splitlines()
    return [r for r in rows if not needle or needle in r]


def preflight_ports(runtime: str, targets: list[dict]) -> bool:
    """Refuse to launch when a target's port is taken, before starting anything,
    so an aborted run cannot leave orphaned containers behind."""
    conflicts = [(t["name"], str(t.get("port")))
                 for t in targets if t.get("port") and port_in_use(str(t["port"]))]
    if not conflicts:
        return True
    warn("preflight: port(s) already in use: "
         + ", ".join(f"{n}:{p}" for n, p in conflicts))
    print("\n[preflight] refusing to launch: these ports are already in use:")
    for name, port in conflicts:
        print(f"  {name:<16} port {port}")
    rows = list_containers(runtime, needle="bench-")
    if rows:
        print("  running bench containers (likely a stale launch):")
        for row in rows:
            print(f"    {row}")
    stops = " ".join(f"{runtime} stop bench-{n}" for n, _ in conflicts)
    print(f"  fix: {stops}")
    print("       (or serve on another port with PORT=<n> ./serve-mxfp4.sh)")
    return False


def host_checkpoint_path(path: str, models: str) -> str:
    """Map a container-style /models/... path to the host MODELS root.

    serve-mxfp4.sh runs on the HOST and checks the checkpoint under MODELS (bind-mounted at
    /models inside the container), so a --target "snap"/"drafter" has to be a host path. The
    docstring examples use the container form; translate it to the --models root so both work.
    """
    if path.startswith("/models/"):
        root = os.path.realpath(models) if models else \
            os.path.join(os.path.expanduser("~"), "models")
        return os.path.join(root, path[len("/models/"):])
    return path


def launch_instance(args, target: dict, index: int) -> None:
    if target.get("gpu") is None:
        raise SystemExit("--launch needs --target specs with a \"gpu\" index")
    port = str(target.get("port") or 8000 + index)
    container = f"bench-{target['name']}"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KV_OFFLOAD", "RADIANCE_FS_", "RADIANCE_LOOKUP_INVALIDATE",
                                "RADIANCE_ASYNC_ALLOW", "VLLM_MODEL_PATH", "VLLM_SERVED_MODEL"))}
    env.update({
        "NAME": container, "PORT": port, "GPUS": str(target["gpu"]),
        "TP": str(target.get("tp") or 1), "DETACH": "1",
        "SERVED_NAMES": target.get("served") or target["model"],
    })
    if args.models:
        env["MODELS"] = args.models
    if args.runtime:
        env["RUNTIME"] = args.runtime
    for env_key, field in (("SNAP", "snap"), ("DRAFTER", "drafter"),
                           ("SPEC_METHOD", "spec_method"), ("SPEC", "spec"),
                           ("MAXLEN", "maxlen"), ("MAXSEQS", "maxseqs"),
                           ("CHUNK", "chunk"), ("KV_MEM", "kv_mem"),
                           ("GPU_UTIL", "gpu_util")):
        if target.get(field):
            env[env_key] = str(target[field])
    for env_key in ("SNAP", "DRAFTER"):
        if env.get(env_key):
            env[env_key] = host_checkpoint_path(env[env_key], args.models)
    if args.offload_gib:
        env["KV_OFFLOAD_GIB"] = str(args.offload_gib)
        if args.offload_head_cap:
            env["KV_OFFLOAD_HEAD_CAP"] = str(args.offload_head_cap)
        if args.offload_policy:
            env["KV_OFFLOAD_EVICTION_POLICY"] = args.offload_policy
        if args.offload_read_threads:
            env["KV_OFFLOAD_READ_THREADS"] = str(args.offload_read_threads)
        if args.offload_write_threads:
            env["KV_OFFLOAD_WRITE_THREADS"] = str(args.offload_write_threads)
        if args.offload_disk_host:
            env["KV_OFFLOAD_DISK_HOST_DIR"] = args.offload_disk_host
        if args.offload_fanout_max:
            env["RADIANCE_FS_FANOUT_MAX"] = str(args.offload_fanout_max)
        if args.offload_fanout_target_mb:
            env["RADIANCE_FS_FANOUT_TARGET_MB"] = str(args.offload_fanout_target_mb)
    target["offload"] = bool(args.offload_gib)
    off_desc = (f"offload={args.offload_gib}GiB"
                + ("+fs" if args.offload_disk_host else " cpu-only")
                + (f" policy={args.offload_policy}" if args.offload_policy else "")) \
        if args.offload_gib else "offload=off"
    print(f"[launch {target['name']}] gpu={target['gpu']} port={port} "
          f"snap={env.get('SNAP', '(default)')} spec={env.get('SPEC_METHOD', '(default)')} {off_desc}")
    subprocess.run([os.path.abspath(args.serve_script)], env=env, check=True)
    target["base"] = f"http://localhost:{port}/v1"
    target["port"] = port
    target["container"] = container
    target["metrics_url"] = f"http://localhost:{port}/metrics"


def wait_healthy(args, target: dict) -> None:
    url = f"http://localhost:{target['port']}/health"
    deadline = time.time() + args.launch_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    print(f"[launch {target['name']}] healthy at {target['base']}")
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5.0)
    raise SystemExit(f"[launch {target['name']}] not healthy after {args.launch_timeout:.0f}s")


def stop_instances(runtime: str, targets: list[dict]) -> None:
    for target in targets:
        container = target.get("container")
        if not container:
            continue
        print(f"[teardown] {runtime} rm -f {container}")
        subprocess.run([runtime, "rm", "-f", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Clean start / guaranteed teardown.
#
# The tool owns the `bench-*` container namespace. On startup it removes any
# leftovers from a previous run so a port clash cannot block a launch, and it
# registers every container/tailer/reaper it creates so a normal exit, an
# exception, Ctrl-C or SIGTERM all tear the whole run down.
# ---------------------------------------------------------------------------
_CLEANUP = {"runtime": "docker", "containers": set(), "tailers": [], "reaper": None,
            "cleaned": False}
_CLEANUP_LOCK = threading.Lock()


def _track_container(name: str) -> None:
    with _CLEANUP_LOCK:
        _CLEANUP["containers"].add(name)


def _track_tailer(tailer: "ContainerLogTailer") -> None:
    with _CLEANUP_LOCK:
        _CLEANUP["tailers"].append(tailer)


def _track_reaper(reaper) -> None:
    with _CLEANUP_LOCK:
        _CLEANUP["reaper"] = reaper


def _untrack_containers() -> None:
    with _CLEANUP_LOCK:
        _CLEANUP["containers"].clear()


def _untrack_container(name: str) -> None:
    with _CLEANUP_LOCK:
        _CLEANUP["containers"].discard(name)


def _untrack_tailer(tailer: "ContainerLogTailer") -> None:
    with _CLEANUP_LOCK:
        try:
            _CLEANUP["tailers"].remove(tailer)
        except ValueError:
            pass


def _stop_container(runtime: str, name: str) -> None:
    print(f"[teardown] {runtime} rm -f {name}", flush=True)
    try:
        subprocess.run([runtime, "rm", "-f", name], capture_output=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        warn(f"teardown: {exc}")


def _do_cleanup(reason: str = "") -> None:
    """Idempotent teardown: stop tailers/reaper and remove tracked containers."""
    with _CLEANUP_LOCK:
        if _CLEANUP["cleaned"]:
            return
        _CLEANUP["cleaned"] = True
        runtime = _CLEANUP["runtime"]
        containers = sorted(_CLEANUP["containers"])
        tailers = list(_CLEANUP["tailers"])
        reaper = _CLEANUP["reaper"]
    for tailer in tailers:
        try:
            tailer.stop()
        except Exception:  # noqa: BLE001
            pass
    if reaper is not None:
        try:
            reaper.stop()
        except Exception:  # noqa: BLE001
            pass
    if containers:
        print(f"[teardown]{reason} {runtime} rm -f {' '.join(containers)}", flush=True)
        try:
            subprocess.run([runtime, "rm", "-f", *containers],
                           capture_output=True, timeout=60)
        except Exception as exc:  # noqa: BLE001
            warn(f"teardown: {exc}")


def _handle_signal(signum, _frame) -> None:
    print(f"\n[signal] caught {signum}; stopping everything this run started", flush=True)
    _do_cleanup(reason=" signal:")
    sys.exit(128 + signum)


def _arm_cleanup(runtime: str) -> None:
    _CLEANUP["runtime"] = runtime
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(_do_cleanup)


def cleanup_stale(runtime: str, enabled: bool = True) -> None:
    """Stop `bench-*` containers left by a previous run before we start."""
    names = []
    for row in list_containers(runtime, needle="bench-"):
        name = row.split("\t", 1)[0].strip()
        if name.startswith("bench-"):
            names.append(name)
    if not names:
        return
    names = sorted(set(names))
    if not enabled:
        print(f"[cleanup] leaving {len(names)} pre-existing bench container(s): "
              f"{' '.join(names)}")
        return
    print(f"[cleanup] stopping pre-existing bench container(s): {' '.join(names)}", flush=True)
    try:
        subprocess.run([runtime, "rm", "-f", *names], capture_output=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        warn(f"cleanup: {exc}")


class ReaperManager:
    """Keeps the mandatory fs-tier eviction policy running for the benchmark window."""

    UNIT = "kvcache-reap.timer"

    def __init__(self, args, host_dir: str):
        self.args = args
        self.root = os.path.join(host_dir, "blocks")
        self.mode = "off"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_systemd = False

    def _env(self) -> dict:
        env = os.environ.copy()
        env["KVCACHE_ROOT"] = self.root
        if self.args.reaper_max_gib:
            env["KVCACHE_MAX_GIB"] = str(self.args.reaper_max_gib)
        return env

    def _systemd_available(self) -> bool:
        return shutil.which("systemctl") is not None and subprocess.run(
            ["systemctl", "cat", self.UNIT],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def _systemd_active(self) -> bool:
        return subprocess.run(["systemctl", "is-active", "--quiet", self.UNIT]).returncode == 0

    def start(self) -> None:
        mode = self.args.reaper
        if mode == "auto":
            if self._systemd_available():
                mode = "systemd"
            elif os.path.exists(self.args.reaper_script):
                mode = "loop"
            else:
                mode = "off"
        self.mode = mode
        print(f"[reaper] mode={mode} root={self.root}")
        if mode == "systemd":
            if self._systemd_active():
                print(f"[reaper] {self.UNIT} already active")
                return
            proc = subprocess.run(["systemctl", "start", self.UNIT],
                                  capture_output=True, text=True)
            if proc.returncode == 0 and self._systemd_active():
                self._started_systemd = True
                print(f"[reaper] started {self.UNIT}")
            else:
                reason = (proc.stderr or proc.stdout or "permission denied").strip()
                warn(f"could not start {self.UNIT}: {reason}")
                print(f"[reaper]   run: sudo KVCACHE_ROOT={self.root} systemctl start {self.UNIT}")
        elif mode == "loop":
            script = os.path.abspath(self.args.reaper_script)
            if os.geteuid() != 0:
                warn("loop not running as root; deletions may fail")
            def loop():
                env = self._env()
                while not self._stop.is_set():
                    subprocess.run(["bash", script], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self._stop.wait(self.args.reaper_interval)
            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
            print(f"[reaper] loop started ({script} every {self.args.reaper_interval:.0f}s)")
        else:
            warn("no reaper available; the fs tier is unsafe without it")

    def stop(self) -> None:
        if self.mode == "loop":
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=5.0)
            print("[reaper] loop stopped")
        elif self.mode == "systemd" and self._started_systemd:
            subprocess.run(["systemctl", "stop", self.UNIT],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._started_systemd = False
            print(f"[reaper] stopped {self.UNIT} (restored prior state)")


def bench_target(args, target: dict, phases: list[str]):
    """Measure one target via BetterBench and persist its result. (path, failed, result)."""
    try:
        result = execute_target(args, target, phases)
    except Exception as exc:  # noqa: BLE001
        warn(f"target FAILED: {exc!r}", target=target["name"])
        if getattr(args, "debug", False):
            traceback.print_exc()
        return None, True, None
    path = args.save
    if args.save_dir:
        path = os.path.join(args.save_dir, f"{target['name']}.json")
    elif len(args.target or []) > 1 and args.save:
        path = re.sub(r"\.json$", f".{target['name']}.json", args.save)
    if path:
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"saved {path}")
    return path, False, result


def write_overview(args, phases: list[str], targets: list[dict],
                   results: list, paths: list) -> str:
    """Write ONE JSON covering every target so the per-target files need not be read.

    Structure: metadata + a `targets` list (config, BetterBench env fingerprint,
    GPU summary, aggregated per-phase metrics) + a flat `comparison` list of cells
    keyed by target name.
    """
    path = args.overview or (
        os.path.join(args.save_dir, "overview.json") if args.save_dir
        else (re.sub(r"\.json$", ".overview.json", args.save) if args.save
              else os.path.join(os.getcwd(), "overview.json")))
    entries = []
    for target, result, tpath in zip(targets, results, paths):
        if not result:
            entries.append({"target": target["name"], "model": target.get("model"),
                            "failed": True, "path": tpath})
            continue
        bb = result.get("phases") or {}
        entries.append({
            "target": target["name"],
            "model": result["config"].get("model"),
            "failed": False,
            "path": tpath,
            "betterbench_path": result.get("betterbench_path"),
            "config": result["config"],
            "betterbench_version": bb.get("betterbench_version"),
            "corpus_version": bb.get("corpus_version"),
            "env": bb.get("env"),
            "gpu": result.get("gpu"),
            "phases": summarize_result(result),
        })
    live = [e for e in entries if not e["failed"]]

    comparison: list[dict] = []
    for ph in ALL_PHASES:
        es = [e for e in live if ph in e["phases"]]
        if not es:
            continue
        if ph == "decode":
            cats = sorted({c for e in es for c in e["phases"]["decode"]["categories"]})
            for c in cats:
                comparison.append({
                    "phase": "decode", "scope": c, "metric": "decode_tps_median",
                    "v": {e["target"]: e["phases"]["decode"]["categories"].get(c, {})
                          .get("decode_tps_median") for e in es}})
            for m in ("combined_decode_tps", "ttft_p50_ms", "update_p50_ms", "n_ok"):
                comparison.append({
                    "phase": "decode", "scope": "all", "metric": m,
                    "v": {e["target"]: e["phases"]["decode"].get(m) for e in es}})
        elif ph == "prefill":
            depths = sorted({r["target_depth"] for e in es
                             for r in e["phases"]["prefill"]})
            for d in depths:
                comparison.append({
                    "phase": "prefill", "scope": str(d), "metric": "pp_tps_median",
                    "v": {e["target"]: next((r.get("pp_tps_median")
                          for r in e["phases"]["prefill"]
                          if r["target_depth"] == d and not r.get("skipped")), None)
                          for e in es}})
        elif ph == "concurrency":
            levels = sorted({r["level"] for e in es
                             for r in e["phases"]["concurrency"]})
            for lvl in levels:
                for m in ("aggregate_tps", "ttft_p50_ms", "decode_tps_median"):
                    comparison.append({
                        "phase": "concurrency", "scope": str(lvl), "metric": m,
                        "v": {e["target"]: next((r.get(m)
                              for r in e["phases"]["concurrency"]
                              if r["level"] == lvl), None) for e in es}})

    workload = next((r["workload"] for r in results if r), None) or {
        "phases": phases, "betterbench": True}
    overview = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": workload,
        "targets": entries,
        "comparison": comparison,
    }
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(overview, fh, indent=2)
    failed = [e["target"] for e in entries if e["failed"]]
    print(f"saved overview {path}"
          + (f" ({len(failed)} failed: {', '.join(failed)})" if failed else ""))
    return path


def _fmt(v, unit=""):
    if v is None or v != v:
        return "      n/a"
    if unit == "ms":
        return f"{v:8.1f}ms"
    return f"{v:10.4f}{unit}"


def _row(name, va, vb, unit="", good="none"):
    da = vb - va if va is not None and vb is not None and va == va and vb == vb \
        else float("nan")
    tag = ""
    if good in ("up", "down") and da == da and da != 0:
        tag = "better" if ((da > 0) == (good == "up")) else "worse"
    print(f"  {name:<26} A={_fmt(va, unit)}  B={_fmt(vb, unit)}  d={_fmt(da, unit)}  {tag}")


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    ta = (a.get("target") or {}).get("name", "A")
    tb = (b.get("target") or {}).get("name", "B")
    print(f"A = {path_a} [{ta}] label={a.get('label')}")
    print(f"B = {path_b} [{tb}] label={b.get('label')}")

    ca, cb = summarize_result(a), summarize_result(b)
    diffs = {k: (a["config"].get(k), b["config"].get(k))
             for k in set(a["config"]) | set(b["config"])
             if a["config"].get(k) != b["config"].get(k)}
    print("config diff (A -> B): " + (json.dumps(diffs) if diffs else "none"))
    for key in ("betterbench_version", "corpus_version"):
        va, vb = a.get(key), b.get(key)
        if va != vb:
            print(f"{key}: A={va} B={vb} (results are only comparable within a version)")

    if "decode" in ca and "decode" in cb:
        print("\n== decode (single stream) ==")
        _row("combined_decode_tps", ca["decode"].get("combined_decode_tps"),
             cb["decode"].get("combined_decode_tps"), "", "up")
        _row("ttft_p50_ms", ca["decode"].get("ttft_p50_ms"),
             cb["decode"].get("ttft_p50_ms"), "ms", "down")
        _row("ttft_p99_ms", ca["decode"].get("ttft_p99_ms"),
             cb["decode"].get("ttft_p99_ms"), "ms", "down")
        _row("update_p50_ms", ca["decode"].get("update_p50_ms"),
             cb["decode"].get("update_p50_ms"), "ms", "down")
        _row("update_p99_ms", ca["decode"].get("update_p99_ms"),
             cb["decode"].get("update_p99_ms"), "ms", "down")
        cats = sorted(set(ca["decode"]["categories"]) | set(cb["decode"]["categories"]))
        for c in cats:
            _row(f"decode_tps[{c}]",
                 ca["decode"]["categories"].get(c, {}).get("decode_tps_median"),
                 cb["decode"]["categories"].get(c, {}).get("decode_tps_median"),
                 " t/s", "up")
    if "prefill" in ca and "prefill" in cb:
        print("\n== prefill (pp t/s median) ==")
        depths = sorted({r["target_depth"] for r in ca["prefill"]} |
                        {r["target_depth"] for r in cb["prefill"]})
        for d in depths:
            va = next((r.get("pp_tps_median") for r in ca["prefill"]
                       if r["target_depth"] == d and not r.get("skipped")), None)
            vb = next((r.get("pp_tps_median") for r in cb["prefill"]
                       if r["target_depth"] == d and not r.get("skipped")), None)
            _row(f"pp_tps[{d}]", va, vb, " t/s", "up")
    if "concurrency" in ca and "concurrency" in cb:
        print("\n== concurrency ==")
        levels = sorted({r["level"] for r in ca["concurrency"]} |
                        {r["level"] for r in cb["concurrency"]})
        for lvl in levels:
            va = next((r for r in ca["concurrency"] if r["level"] == lvl), {})
            vb = next((r for r in cb["concurrency"] if r["level"] == lvl), {})
            _row(f"aggregate_tps[{lvl}]", va.get("aggregate_tps"),
                 vb.get("aggregate_tps"), " t/s", "up")
            _row(f"ttft_p50_ms[{lvl}]", va.get("ttft_p50_ms"),
                 vb.get("ttft_p50_ms"), "ms", "down")

    print("\nnote: this is an offline, unpaired comparison. For interleaved, "
          "drift-cancelled A/B use `betterbench ab`.")
    return 0


def build_and_run(args) -> int:
    phases = resolve_phases(args.phases)

    if args.target:
        targets = [parse_target(t, args.model, args.launch) for t in args.target]
    else:
        targets = [{"name": "default", "base": args.base.rstrip("/"), "model": args.model,
                    "prom_filter": args.prom_filter, "instance": str(args.instance or "")}]

    if args.offload_gib and len(targets) != 1:
        raise SystemExit("offload KV state is model-specific: use exactly one --target "
                         "when --offload-gib is set (disable offload to compare two models)")

    if args.debug:
        args.verbose_logs = True
        debug(args, f"debug on: targets={[t['name'] for t in targets]} phases={phases}")

    runtime = detect_runtime(args.runtime)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    warn_path = args.warnings_log or (
        os.path.join(args.save_dir, "warnings.log") if args.save_dir
        else os.path.join(os.getcwd(), "warnings.log"))
    warn_log_init(warn_path)
    print(f"[warn] logging warnings -> {warn_path}")
    _arm_cleanup(runtime)

    # One card only: pick the optimal (widest host PCIe link) GPU and pin every
    # target to it. Two models cannot share a 32 GiB card, so each target is
    # launched, benchmarked and torn down before the next one starts.
    if args.launch:
        best, ranking = pick_best_card(args.card or os.environ.get("BENCH_CARD") or None)
        if best is None:
            raise SystemExit("could not detect GPUs; pass --card <hip_index>")
        report_card(best, ranking)
        for t in targets:
            if t.get("gpu") not in (None, best):
                print(f"[card] target {t['name']}: overriding gpu={t['gpu']} -> {best}")
            t["gpu"] = best

    overall = Progress(len(targets), "benchmark")
    launched = False
    failures = 0
    paths: list[str | None] = []
    results: list[dict | None] = []
    try:
        if args.launch:
            # Assign ports up front so a conflict is caught before anything starts.
            for i, target in enumerate(targets):
                target["port"] = str(target.get("port") or 8000 + i)
            cleanup_stale(runtime, enabled=args.cleanup)
            if not preflight_ports(runtime, targets):
                return 2

            if args.offload_disk_host and args.reaper != "off":
                reaper = ReaperManager(args, args.offload_disk_host)
                _track_reaper(reaper)
                reaper.start()

            launched = True
            width = max(len(t["name"]) for t in targets)
            for i, target in enumerate(targets):
                name = f"bench-{target['name']}"
                # Track before the launch: a signal during a slow start must
                # still be able to remove the container it may have created.
                _track_container(name)
                try:
                    launch_instance(args, target, i)
                except subprocess.CalledProcessError as exc:
                    print(f"[launch {target['name']}] {args.serve_script} failed "
                          f"(exit {exc.returncode}); tearing down this run")
                    _do_cleanup(reason=" failed-launch:")
                    return exc.returncode or 1
                tailer = None
                if args.logs:
                    tailer = ContainerLogTailer(runtime, name, target["name"],
                                                width=width, verbose=args.verbose_logs)
                    tailer.start()
                    _track_tailer(tailer)
                try:
                    wait_healthy(args, target)
                except SystemExit as exc:
                    warn(str(exc), target=target["name"])
                    paths.append(None)
                    results.append(None)
                    failures += 1
                else:
                    path, failed, result = bench_target(args, target, phases)
                    paths.append(path)
                    results.append(result)
                    failures += 1 if failed else 0
                overall.step(target["name"])
                # Tear this target down before the next launch shares the card.
                keep_this = args.keep and i == len(targets) - 1
                if tailer is not None:
                    tailer.stop()
                    _untrack_tailer(tailer)
                if keep_this:
                    _untrack_container(name)   # leave it running on purpose
                else:
                    _stop_container(runtime, name)
                    _untrack_container(name)
        else:
            for target in targets:
                path, failed, result = bench_target(args, target, phases)
                paths.append(path)
                results.append(result)
                failures += 1 if failed else 0
                overall.step(target["name"])

        write_overview(args, phases, targets, results, paths)
        if len(targets) == 2 and all(paths):
            print("\n=== target comparison ===")
            compare(paths[0], paths[1])
        if failures:
            print(f"\n{failures} target(s) failed; see the messages above", file=sys.stderr)
    finally:
        overall.close()
        _do_cleanup()
        warn_log_close()
    return 1 if failures else 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--target", action="append", default=[],
                   help="JSON target spec; repeat to benchmark several endpoints/models")
    p.add_argument("--launch", action="store_true",
                   help="start one plain (offload-free) vLLM per target via --serve-script, then tear down")
    p.add_argument("--serve-script", default="./serve-mxfp4.sh",
                   help="host launcher used by --launch")
    p.add_argument("--models", default="", help="MODELS dir passed to --serve-script (host checkpoints)")
    p.add_argument("--runtime", default="", help="container runtime for --launch/teardown (auto: podman, docker)")
    p.add_argument("--card", default="",
                   help="HIP index to force as the benchmark card (default: auto-pick the widest "
                        "host PCIe link; e.g. 1 is the x16 card here, 0 is the x1 one)")
    p.add_argument("--keep", action="store_true", help="do not tear down launched instances")
    p.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True,
                   help="stop pre-existing bench-* containers before launching (default on)")
    p.add_argument("--logs", action=argparse.BooleanOptionalAction, default=True,
                   help="stream each launched container's logs into the bench output, prefixed by target")
    p.add_argument("--verbose-logs", action="store_true",
                   help="include tqdm progress / weight-loading lines in the streamed logs")
    p.add_argument("--debug", action="store_true",
                   help="verbose everything: stream all container log lines and print raw target info")
    p.add_argument("--launch-timeout", type=float, default=1800.0, help="seconds to wait for /health")

    # BetterBench passthrough.
    p.add_argument("--betterbench", default="",
                   help="betterbench command to run (default: $BETTERBENCH_BIN, else `betterbench` "
                        "on PATH, else ~/betterbench/.venv/bin/betterbench, else `python3 -m betterbench.cli`)")
    p.add_argument("--passes", type=int, default=None,
                   help="measured passes per category passed to betterbench (default: its own 20)")
    p.add_argument("--warmup", type=int, default=None,
                   help="discarded warmup passes per category passed to betterbench (default: its own 3)")
    p.add_argument("--quick", action="store_true",
                   help="betterbench smoke run: 5 passes after 1 warmup")
    p.add_argument("--bb-config", default="", help="betterbench --config JSON file")
    p.add_argument("--bb-corpus", default="", help="betterbench --corpus directory")
    p.add_argument("--bb-categories", nargs="*", default=None,
                   help="betterbench --categories (subset of its corpus)")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="model context window passed to betterbench (skips prefill depths that do not fit)")
    p.add_argument("--note", action="append", default=[],
                   help="extra betterbench metadata as KEY=VALUE; repeatable")
    p.add_argument("--bb-name", default="", help="betterbench run label (default: target name)")
    p.add_argument("--bb-out", default="", help="explicit results.json path for a single target")
    p.add_argument("--no-html", action="store_true", help="skip betterbench's HTML report")

    # Offload / fs-tier reaper (host launch side).
    p.add_argument("--offload-gib", default="",
                   help="enable KV offload with this CPU-tier size in GiB (default off = plain)")
    p.add_argument("--offload-disk-host", default="",
                   help="host dir for the shared fs secondary tier (mounts /kvcache; implies a reaper)")
    p.add_argument("--offload-policy", default="", help="primary-tier eviction policy: lru or arc")
    p.add_argument("--offload-read-threads", default="", help="fs tier read threads")
    p.add_argument("--offload-write-threads", default="", help="fs tier write threads")
    p.add_argument("--offload-head-cap", default="", help="max_offload_tokens: auto | <N> | off")
    p.add_argument("--offload-fanout-max", default="", help="fs tier fan-out batch cap (RADIANCE_FS_FANOUT_MAX)")
    p.add_argument("--offload-fanout-target-mb", default="", help="fs tier fan-out budget MiB (RADIANCE_FS_FANOUT_TARGET_MB)")
    p.add_argument("--reaper", choices=("auto", "systemd", "loop", "off"), default="auto",
                   help="fs-tier eviction policy for --offload-disk-host (auto: systemd timer, else loop)")
    p.add_argument("--reaper-script",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops", "kvcache-reap.sh"),
                   help="reaper script used in loop mode")
    p.add_argument("--reaper-interval", type=float, default=60.0,
                   help="seconds between reaper runs in loop mode")
    p.add_argument("--reaper-max-gib", default="", help="KVCACHE_MAX_GIB passed to the reaper")

    # Endpoint / sampling.
    p.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://llm-lb:8100/v1"))
    p.add_argument("--prom", default=os.environ.get("BENCH_PROM", "http://prometheus:9090"),
                   help="Prometheus base for radiance_serve_info config enrichment")
    p.add_argument("--gpu", default=os.environ.get("BENCH_GPU", "http://gpu-exporter:9101/metrics"),
                   help="gpu-exporter /metrics URL when --gpu-source prom")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "juup-123"))
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", "Qwen3.8-MXFP4"))
    p.add_argument("--prom-filter", default="", help="Prometheus label selector to scope counters")
    p.add_argument("--instance", default="", help="radiance_serve_info instance label for this target")
    p.add_argument("--phases", default="all",
                   help="all or comma list of " + ",".join(ALL_PHASES)
                        + " (`pressure` aliases concurrency)")
    p.add_argument("--seed", type=int, default=None, help="sampling seed passed to betterbench")
    p.add_argument("--gpu-source", choices=("auto", "prom", "sysfs"), default="auto",
                   help="GPU busy source: local sysfs (default) or Prometheus gpu-exporter")
    p.add_argument("--gpu-interval", type=float, default=0.5)
    p.add_argument("--label", default="")
    p.add_argument("--save", default="")
    p.add_argument("--save-dir", default="")
    p.add_argument("--overview", default="",
                   help="override the path of the combined all-targets summary JSON "
                        "(written by default to <save-dir>/overview.json, else alongside "
                        "--save, else ./overview.json)")
    p.add_argument("--warnings-log", default="",
                   help="file for all warnings (default: <save-dir>/warnings.log, else ./warnings.log)")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    return build_and_run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
