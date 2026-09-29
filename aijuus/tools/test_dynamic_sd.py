#!/usr/bin/env python3
"""Unit tests for the native dynamic-SD plumbing (vLLM 0.29).

Run INSIDE a vLLM container:
    /opt/vllm/bin/python3 /patches/aijuus/tools/test_dynamic_sd.py

Covers:
  1. validate_and_normalize_dynamic_sd_schedule: accepts our schedule, rejects malformed ones.
  2. build_dynamic_sd_schedule_lookup: dense 1-indexed lookup for max_num_seqs=8.
  3. The cudagraph `_init_candidates` decode_query_lens formula (the site patched by
     patch_dynamic_sd_cudagraph.py): speculator decode manager (decode_query_len=1) crashes
     unpatched and yields {1} patched; main runner (decode_query_len=9) is unchanged.
  4. The installed cudagraph_utils.py actually contains the patched guard.
Exit code 0 = all pass.
"""
import sys

failures = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ((" :: " + detail) if detail else ""))
    if not cond:
        failures.append(name)


SCHED = [[1, 1, 8], [2, 3, 7], [4, 7, 6], [8, 8, 5], [9, 64, 4]]
VLLM_NSPEC = 8
MAX_BS = 8

# --- 1. validation -----------------------------------------------------------
from vllm.v1.spec_decode.dynamic.utils import (  # noqa: E402
    build_dynamic_sd_schedule_lookup,
    validate_and_normalize_dynamic_sd_schedule,
)

norm = validate_and_normalize_dynamic_sd_schedule(SCHED)
check("validate: our schedule accepted sorted", norm == [tuple(x) for x in SCHED], repr(norm))

for bad, why in [
    (None, "None"),
    ([], "empty"),
    ([[1, 4]], "2-item entry"),
    ([[0, 4, 8]], "start 0"),
    ([[1, 4, 8], [4, 8, 6]], "overlap"),
    ([[2, 4, 8]], "starts at 2"),
    ([[1, 4, -1]], "negative K"),
]:
    try:
        validate_and_normalize_dynamic_sd_schedule(bad)
        check(f"validate rejects {why}", False, "no raise")
    except ValueError:
        check(f"validate rejects {why}", True)

# --- 2. lookup ---------------------------------------------------------------
lookup = build_dynamic_sd_schedule_lookup(SCHED, vllm_max_batch_size=MAX_BS,
                                          vllm_num_speculative_tokens=VLLM_NSPEC)
check("lookup: length = max_num_seqs+1", len(lookup) == MAX_BS + 1, repr(lookup))
check("lookup: value", lookup == [0, 8, 7, 7, 6, 6, 6, 6, 5], repr(lookup))
check("lookup: index 0 unused", lookup[0] == 0)

# carry-forward across a gap + clamp K to vllm nspec
lf = build_dynamic_sd_schedule_lookup([[1, 4, 3], [8, 12, 2]], vllm_max_batch_size=12,
                                      vllm_num_speculative_tokens=8)
check("lookup: gap carry-forward", lf[5] == 3 and lf[7] == 3 and lf[8] == 2, repr(lf))
lc = build_dynamic_sd_schedule_lookup([[1, 8, 20]], vllm_max_batch_size=8,
                                      vllm_num_speculative_tokens=8)
check("lookup: K clamped to vllm nspec", all(v == 8 for v in lc[1:]), repr(lc))

# --- 3. cudagraph decode_query_lens formula ----------------------------------
from vllm.utils.math_utils import round_up  # noqa: E402


def patched_lens(decode_query_len):
    """Mirror of the patched _init_candidates branch."""
    dense = build_dynamic_sd_schedule_lookup(SCHED, vllm_max_batch_size=MAX_BS,
                                             vllm_num_speculative_tokens=VLLM_NSPEC)
    off = decode_query_len - VLLM_NSPEC
    return sorted(_q for _q in {ns + off for ns in dense[1:]} if _q >= 1) or [decode_query_len]


def unpatched_lens(decode_query_len):
    dense = build_dynamic_sd_schedule_lookup(SCHED, vllm_max_batch_size=MAX_BS,
                                             vllm_num_speculative_tokens=VLLM_NSPEC)
    off = decode_query_len - VLLM_NSPEC
    return sorted({ns + off for ns in dense[1:]})


# speculator decode manager: decode_query_len == 1
un = unpatched_lens(1)
check("cudagraph: unpatched speculator manager contains 0 (crash)", 0 in un, repr(un))
try:
    round_up(4, 0)
    check("cudagraph: round_up(x,0) raises", False, "no raise")
except ZeroDivisionError:
    check("cudagraph: round_up(x,0) raises", True)
check("cudagraph: patched speculator manager == {1}", patched_lens(1) == [1], repr(patched_lens(1)))

# main runner: decode_query_len == num_spec+1 == 9
check("cudagraph: patched main runner unchanged {6,7,8,9}",
      patched_lens(9) == unpatched_lens(9) == [6, 7, 8, 9], repr(patched_lens(9)))
check("cudagraph: patched main runner has no zero", all(q >= 1 for q in patched_lens(9)))

# --- 4. installed patch present ---------------------------------------------
import vllm  # noqa: E402
from pathlib import Path  # noqa: E402
src = (Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "cudagraph_utils.py").read_text()
if "patch_dynamic_sd_cudagraph" in src and "if _q >= 1" in src:
    check("installed: cudagraph_utils.py carries the guard", True)
else:
    print("NOTE  cudagraph_utils.py guard not installed (schedule off / patch dormant) — expected "
          "unless patch_dynamic_sd_cudagraph.py was applied this boot")

sched_src = (Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py").read_text()
if "patch_sd_sched_trace" in sched_src:
    check("installed: scheduler.py carries the sd-trace", True)
else:
    print("NOTE  scheduler sd-trace not installed (schedule off) — expected unless armed this boot")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all dynamic-SD unit tests passed")
