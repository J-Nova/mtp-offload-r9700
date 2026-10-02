#!/usr/bin/env python3
"""Apply the MTP acceptance gate to an ALREADY-patched speculator in a running container.

WHY. The overlay patches are marker-idempotent, so `docker start` (which reuses the container
filesystem) re-applies nothing -- patch_dynamic_depth prints "already applied" and the old speculator
stays. A changed patch normally needs a container RECREATE. This applies just the gate delta to the
installed file so a single extra kill/start activates it without a rebuild.

Idempotent: skips if RADIANCE_ACC_GATE is already present. ast.parse before writing.
"""
import ast
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"

ANCHOR = (
    "        self._radiance_eff_k = _rd_k\n"
    "        self._radiance_dyn_k = _rd_k\n"
)

GATE = '''        # RADIANCE acceptance gate (patch_dynamic_depth): at small batch the serial draft loop
        # dominates -- each of the K-1 draft forwards re-streams the ~850MB bf16 MTP block -- so
        # shrink K when the observed accepted length is low. max-across-requests (num_reqs <=
        # RADIANCE_ACC_GATE_BATCH) means a good co-scheduled request is never dragged down; a
        # batch above the threshold keeps the size-only schedule (weight-stream flat zone).
        import os as _rd_ae_os
        if (
            not dummy_run
            and not is_profile
            and _rd_ae_os.environ.get("RADIANCE_ACC_GATE", "0") == "1"
            and num_reqs <= int(_rd_ae_os.environ.get("RADIANCE_ACC_GATE_BATCH", "2"))
        ):
            import torch as _rd_ae_torch
            _rd_ae_alpha = float(_rd_ae_os.environ.get("RADIANCE_ACC_GATE_ALPHA", "0.35"))
            _rd_ae_margin = int(_rd_ae_os.environ.get("RADIANCE_ACC_GATE_MARGIN", "1"))
            with _rd_ae_torch.no_grad():
                _rd_ae_acc = (num_sampled - 1).clamp(min=0)
                _rd_ae_max = float(_rd_ae_acc.max().item()) if _rd_ae_acc.numel() else 0.0
            # accepting the full previous width is evidence the cap binds; observe one above it so
            # the EMA can climb back out (same trick as patch_dynwidth).
            _rd_ae_prev = getattr(self, "_radiance_gate_k", _rd_k)
            _rd_ae_obs = _rd_ae_max + (1.0 if _rd_ae_max >= _rd_ae_prev else 0.0)
            _rd_ae_ema = getattr(self, "_radiance_gate_ema", None)
            _rd_ae_ema = (
                _rd_ae_obs
                if _rd_ae_ema is None
                else _rd_ae_alpha * _rd_ae_obs + (1.0 - _rd_ae_alpha) * _rd_ae_ema
            )
            self._radiance_gate_ema = _rd_ae_ema
            _rd_ae_ceil = int(_rd_ae_ema) + (0 if float(_rd_ae_ema).is_integer() else 1)
            _rd_ae_des = max(1, _rd_ae_ceil + _rd_ae_margin)
            if _rd_ae_des < _rd_k:
                _rd_k = _rd_ae_des
            if _rd_ae_os.environ.get("RADIANCE_ACC_GATE_DIAG", "0") == "1":
                import sys as _rd_ae_sys
                print(f"[acc-gate] bs={num_reqs} accmax={_rd_ae_max:.0f} ema={_rd_ae_ema:.2f} "
                      f"k={_rd_k}", file=_rd_ae_sys.stderr)
        self._radiance_gate_k = _rd_k
'''


def main():
    src = TARGET.read_text()
    if "RADIANCE_ACC_GATE" in src:
        print("  NOOP  acceptance gate already present")
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print(f"  FAIL  anchor matched {n}x, expected 1")
        return 1
    out = src.replace(ANCHOR, GATE + ANCHOR, 1)
    ast.parse(out)  # never write a file that would not parse
    TARGET.write_text(out)
    print(f"  OK    gate inserted into {TARGET.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
