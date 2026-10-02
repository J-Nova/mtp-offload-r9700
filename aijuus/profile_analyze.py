#!/usr/bin/env python3
"""Aggregate a torch-profiler (kineto) chrome trace for the MTP decode step.

Reads one or more chr*.json(.gz) traces, aggregates GPU kernel time by name, and reports total
kernel-busy vs trace wall so the decode gap can be split into (a) kernel/clock time and (b) host /
launch bubbles. This is the N7 readout.

Usage:
  ./profile_analyze.py /tmp/kilo/prof/*.json.gz [--top 30]
"""
import argparse
import glob
import gzip
import json
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files += glob.glob(p)
    if not files:
        print("no trace files matched", file=sys.stderr)
        return 1

    for path in files:
        ev = load(path)
        # GPU work: kernel / memcpy / memset have a 'dur'; cpu_op/flow do not.
        kern = defaultdict(lambda: [0.0, 0])      # name -> [sum_us, count]
        gpu_classes = defaultdict(float)
        busy = 0.0
        t0 = t1 = None
        for e in ev:
            if "ts" in e:
                ts, dur = e["ts"], e.get("dur", 0)
                if t0 is None or ts < t0:
                    t0 = ts
                if ts + dur > (t1 or 0):
                    t1 = ts + dur
            cat = e.get("cat", "")
            dur = e.get("dur")
            if dur is None:
                continue
            if cat in ("kernel", "Kernel"):
                kern[e.get("name", "?")][0] += dur
                kern[e.get("name", "?")][1] += 1
                busy += dur
                gpu_classes["kernel"] += dur
            elif cat == "gpu_memcpy":
                busy += dur
                gpu_classes["memcpy"] += dur
            elif cat == "gpu_memset":
                busy += dur
                gpu_classes["memset"] += dur
        wall = (t1 - t0) if (t0 is not None and t1 is not None) else 0.0
        print(f"\n=== {path} ===")
        print(f"events={len(ev)}  trace wall={wall/1000:.1f} ms  gpu_busy={busy/1000:.1f} ms  "
              f"idle/gap={max(wall-busy,0)/1000:.1f} ms  ({100*busy/wall:.0f}% busy)" if wall else
              "  (no timestamps)")
        print("  gpu classes: " + ", ".join(f"{k} {v/1000:.1f}ms" for k, v in sorted(gpu_classes.items(), key=lambda x: -x[1])))
        print(f"  top {a.top} kernels by total time:")
        for name, (su, c) in sorted(kern.items(), key=lambda x: -x[1][0])[:a.top]:
            print(f"    {su/1000:9.2f} ms  x{c:<6} {su/c/1000:7.3f} ms avg  {name[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
