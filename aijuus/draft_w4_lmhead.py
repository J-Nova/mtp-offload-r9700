"""draft_w4_lmhead: int4 group-128 lm_head for the MTP DRAFT loop only, on gfx1201.

Two ways in: a checkpoint that ships ``mtp.lm_head.weight_q4`` / ``weight_scale`` /
``weight_zero`` (TP-agnostic, packed here per rank at load with no quantization, see
``pack_from_codes``), or DAVETHA_DRAFTER_QUANT=1 which quantizes the shared bf16 head
online at warmup. Neither present, the draft head stays bf16.

Wraps libr4d's ``r4d_gemm_w4a16_nt_m64``: asymmetric 4-bit weight, per output channel per
group of 128 contiguous K, f16 activation, bf16 output, fp32 accumulate, split-K in the
workgroup, HIP-graph capturable.

WHY IT IS SAFE HERE AND NOWHERE ELSE.  The draft proposes tokens the target then verifies;
a worse draft only lowers the MTP acceptance rate and can never change what the target
emits.  So the draft's lm_head -- 124160 x 2560 per rank, 636 MB of bf16 read once per
draft step -- can be quantized where the target's cannot.  This module is deliberately
NOT registered anywhere: the MTP model code asks for it by name.

MEASURED by the original author (R9700 gfx1201, N=124160 K=2560, cold, in HIP graph):
  us/call    M=1     M=2     M=4     M=5     M=20
  bf16       1006.0  1006.2  1018.6  1021.9  1059.2   (vllm rocm_unquantized_gemm)
  this        261.3   263.7   266.9   268.1   306.8   -> 3.85x 3.82x 3.82x 3.81x 3.45x
Both are DRAM-bound and the speedup is the byte ratio.  Packing takes ~0.75 s on GPU at
load.  Their production arm: step 41.7 -> 38.9 ms, acceptance -3..-9% relative, net
+4..+8% tok/s.  Re-measure on this fork before trusting any of those numbers.

QUALITY: 9.99% relative Frobenius error at g128 with the clip search (10.52% on plain
min/max).  That is the 4-bit grid, not the scale.  A draft is the one place that is
affordable.

ACTIVATION IS f16, NOT bf16.  The kernel widens the 4-bit code to f16, so its A operand is
f16; ``gemm_w4a16`` converts the bf16 hidden state.  f16 has more mantissa than bf16, so
the risk is RANGE: a hidden state past 65504 becomes inf.  Nothing clamps it, deliberately:
a draft that produces inf costs an acceptance, never a wrong target token.

LAYOUT (r4d_gemm_w4a16_nt_m64.hip):
  wq   uint8  [N/16, K/64, 32, 16] -- one uint4 (16 B) per lane per four k steps, in WMMA
         fragment order; nibble is the TWO'S COMPLEMENT code q ^ 8.
  wsz  int32  [N/16, K/128, 16] -- per (row, group): f16 scale in the low half, f16 of
         -(1024 + zero) in the high half.
Dequantized value is ``scale * (q - zero)`` with q in 0..15 and an INTEGER zero.

Ported from davetha/r9700-lru-expert-cache (Apache-2.0) with the gate renamed and the
hidden-state dump tooling removed. Adapted for radiance-vllm-mxfp4.
"""

from __future__ import annotations

import ctypes
import functools
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# Import our local r4d_lib module (sibling to this file in site-packages)
try:
    import r4d_lib
except ImportError:
    r4d_lib = None

logger = init_logger(__name__)

ENV_FLAG = "DAVETHA_DRAFTER_QUANT"

GROUP = 128          # K per (scale, zero) pair; re-read from the library in _fns()
MAX_M = 64           # R4D_GEMM_W4_MAX_M; re-read from the library in _fns()
_WAVE32 = 32
_KPB = 64            # K per packed block (4 k steps, 16 B per lane)
_MAX_THREADS = 1024  # WV*SK*32 <= 1024
_MAX_SMEM = 64 * 1024  # WV*NPW*SK*256 floats

# Rows of N dequantized at a time in the M > MAX_M fallback: 8192 * 2560 * 2 B = 42 MB.
_DEQUANT_CHUNK = 8192


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


@dataclass(frozen=True)
class PackedW4:
    """Everything ``gemm_w4a16`` needs; hold this on the layer."""
    wq: torch.Tensor     # uint8  [N_pad/16, K/64, 32, 16]  (= N_pad*K/2 bytes)
    wsz: torch.Tensor    # int32  [N_pad/16, K/128, 16]
    n: int               # logical output features (before padding)
    n_pad: int           # rows actually packed (multiple of 16)
    k: int

    @property
    def nbytes(self) -> int:
        return self.wq.numel() + self.wsz.numel() * 4


# ---------------------------------------------------------------------------------------
# platform / library gate
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fns():
    """The bound C entry point, or None. ctypes caches the _FuncPtr on the CDLL, so
    binding argtypes here is enough -- r4d_lib.py needs no prototype of its own."""
    global GROUP, MAX_M
    if r4d_lib is None:
        logger.warning_once("draft_w4_lmhead: r4d_lib module not found; unavailable")
        return None
    if not current_platform.is_rocm():
        return None
    try:
        from vllm.platforms.rocm import _GCN_ARCH
    except ImportError:
        return None
    if "gfx1201" not in _GCN_ARCH and "gfx1200" not in _GCN_ARCH:
        return None
    if not r4d_lib.available():
        logger.warning_once("draft_w4_lmhead: libr4d missing (R4D_LIB); unavailable")
        return None
    try:
        lib = r4d_lib.lib()
        fn = lib.r4d_gemm_w4a16_nt_m64
        fn.restype = None
        # (a, wq, wsz, c, M, K, N, WV, SK, MB, NPW, NT, stream)
        fn.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int] * 8 + [ctypes.c_long]
        for name, tgt in (("max_m", "MAX_M"), ("group", "GROUP")):
            g = getattr(lib, f"r4d_gemm_w4a16_nt_m64_{name}")
            g.restype = ctypes.c_int
            g.argtypes = []
            globals()[tgt] = int(g())
    except (OSError, AttributeError) as e:
        logger.warning_once("draft_w4_lmhead: libr4d has no r4d_gemm_w4a16_nt_m64 (%s)", e)
        return None
    return fn


def available() -> bool:
    """True when pack_w4a16 / gemm_w4a16 can use the kernel on this machine."""
    return _fns() is not None


# ---------------------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------------------

# Fractions of the min/max range the clip search tries. Straight min/max is the wrong
# default: over a group of 128 roughly-Gaussian weights the range is ~5.4 sigma, so a
# 16-level grid quantizes at 0.36 sigma. Clipping the tails trades a few saturated
# outliers for a finer grid everywhere else, per (row, group), chosen by squared error.
_CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def quantize_w4(w: torch.Tensor, group: int = 0,
                clip_grid: tuple[float, ...] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[N, K] -> (q uint8 in 0..15, scale f16 [N, K/group], zero int16 [N, K/group]).

    Asymmetric per (row, group), with the f16 rounding of the scale folded in BEFORE the
    codes are chosen so the packed weight is exactly what the kernel will read, and with
    the clip fraction picked per (row, group) by squared error. ``zero`` is an integer,
    not restricted to 0..15: the kernel stores it as the f16 of -(1024 + zero), exact for
    any |zero| <= 1024, and an all-positive group needs a negative one.
    """
    group = group or GROUP
    n, k = w.shape
    assert k % group == 0, f"K={k} must be a multiple of the group {group}"
    wf = w.float().view(n, k // group, group)
    lo, hi = wf.amin(-1), wf.amax(-1)
    mid, half = (hi + lo) / 2, (hi - lo) / 2

    best_err = torch.full_like(lo, float("inf"))
    best_scale = torch.zeros_like(lo, dtype=torch.float16)
    best_zero = torch.zeros_like(lo)
    for c in (clip_grid if clip_grid is not None else _CLIP_GRID):
        lo_c, hi_c = mid - c * half, mid + c * half
        scale = ((hi_c - lo_c) / 15.0).clamp(min=0.0).half()
        # A constant group has no range to quantize: centre it on code 8 so the single
        # value is exact. (An all-zero group -- the N padding -- keeps scale 0, code 0.)
        flat = (scale.float() == 0) & (lo != 0)
        scale = torch.where(flat, (lo.abs() / 7.0).half(), scale)
        sf = scale.float()
        inv = torch.where(sf > 0, 1.0 / sf.clamp(min=1e-30), torch.zeros_like(sf))
        zero = torch.where(flat, torch.full_like(lo, 8.0),
                           torch.round(-lo_c * inv)).clamp_(-1024, 1024)
        q = torch.round(wf * inv.unsqueeze(-1) + zero.unsqueeze(-1)).clamp_(0, 15)
        err = ((q - zero.unsqueeze(-1)) * sf.unsqueeze(-1) - wf).square().sum(-1)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_scale = torch.where(take, scale, best_scale)
        best_zero = torch.where(take, zero, best_zero)
        del scale, sf, inv, zero, q, err

    sf = best_scale.float()
    inv = torch.where(sf > 0, 1.0 / sf.clamp(min=1e-30), torch.zeros_like(sf))
    q = torch.round(wf * inv.unsqueeze(-1) + best_zero.unsqueeze(-1)).clamp_(0, 15)
    return (q.to(torch.uint8).view(n, k), best_scale, best_zero.to(torch.int16))


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 codes (already two's complement) -> [N/16, K/64, 32, 16] uint8.

    Byte j of dword s of lane l is row (l&15)'s k = 16*s + 4*(l>>4) + j in the low nibble
    and + 8 in the high nibble, so the permutation is a reshape and a transpose.
    """
    n, k = codes.shape
    nt, kb = n // 16, k // _KPB
    g = codes.view(nt, 16, kb, 4, 2, 2, 4)          # nt, r, kb, s, half, h, j
    byte = g[:, :, :, :, 0] | (g[:, :, :, :, 1] << 4)   # nt, r, kb, s, h, j
    return byte.permute(0, 2, 4, 1, 3, 5).contiguous().view(nt, kb, 32, 16)


def _pack_sz(scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """(f16 scale, integer zero) [N, K/GROUP] -> int32 [N/16, K/GROUP, 16]."""
    n, ng = scale.shape
    sb = scale.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    nz = (-(1024.0 + zero.float())).half().view(torch.int16).to(torch.int64) & 0xFFFF
    dw = sb | (nz << 16)
    dw = torch.where(dw >= 2 ** 31, dw - 2 ** 32, dw).to(torch.int32)
    return dw.view(n // 16, 16, ng).permute(0, 2, 1).contiguous()


def pack_w4a16(weight: torch.Tensor, group: int = 0, chunk_tiles: int = 512,
               clip_grid: tuple[float, ...] | None = None) -> PackedW4:
    """[N, K] bf16/f32 lm_head weight -> PackedW4 in r4d fragment order.

    Runs once at load. N is padded up to a multiple of 16 with zero rows (the kernel
    requires N % 16 == 0); ``gemm_w4a16`` slices them off. Packing is chunked over
    n-tiles so the transient never exceeds ``chunk_tiles * 16 * K`` bytes.
    """
    assert weight.dim() == 2, f"expected [N, K], got {tuple(weight.shape)}"
    _fns()                                    # binds GROUP / MAX_M from the library
    group = group or GROUP
    n, k = weight.shape
    if k % group:
        raise ValueError(f"draft_w4_lmhead: K={k} must be a multiple of {group}")
    n_pad = (n + 15) // 16 * 16
    dev = weight.device

    wq_out = torch.empty((n_pad // 16, k // _KPB, 32, 16), dtype=torch.uint8, device=dev)
    sz_out = torch.empty((n_pad // 16, k // group, 16), dtype=torch.int32, device=dev)

    rows = chunk_tiles * 16
    for r0 in range(0, n_pad, rows):
        r1 = min(r0 + rows, n_pad)
        w = weight[r0:min(r1, n)]
        if w.shape[0] < r1 - r0:                    # pad tail with zero rows
            w = torch.cat([w, w.new_zeros(r1 - r0 - w.shape[0], k)], 0)
        q, scale, zero = quantize_w4(w, group, clip_grid)
        wq_out[r0 // 16:r1 // 16] = _pack_codes(q ^ 8)   # offset binary -> two's complement
        sz_out[r0 // 16:r1 // 16] = _pack_sz(scale, zero)
        del q, scale, zero, w

    return PackedW4(wq=wq_out, wsz=sz_out, n=n, n_pad=n_pad, k=k)


def unpack_nibbles(q4: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 (even k in the low nibble) -> [N, K] uint8 codes 0..15."""
    n, k2 = q4.shape
    codes = torch.empty((n, k2 * 2), dtype=torch.uint8, device=q4.device)
    codes[:, 0::2] = q4 & 0x0F
    codes[:, 1::2] = q4 >> 4
    return codes


def pack_from_codes(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> PackedW4:
    """Checkpoint-stored int4 (codes 0..15 [N, K], f16 scale + int16 zero [N, K/GROUP]) ->
    PackedW4, the same bytes ``pack_w4a16`` produces from the bf16 weight. No quantization."""
    _fns()
    n, k = codes.shape
    if k % GROUP or scale.shape != (n, k // GROUP) or zero.shape != (n, k // GROUP):
        raise ValueError(f"draft_w4_lmhead: int4 head {tuple(codes.shape)} / {tuple(scale.shape)} "
                         f"/ {tuple(zero.shape)} is not group-{GROUP} consistent")
    n_pad = (n + 15) // 16 * 16
    if n_pad != n:
        codes = torch.cat([codes, codes.new_zeros(n_pad - n, k)], 0)
        scale = torch.cat([scale, scale.new_zeros(n_pad - n, k // GROUP)], 0)
        zero = torch.cat([zero, zero.new_zeros(n_pad - n, k // GROUP)], 0)
    wq = _pack_codes(codes ^ 8)
    wsz = _pack_sz(scale.to(torch.float16), zero.to(torch.int16))
    return PackedW4(wq=wq, wsz=wsz, n=n, n_pad=n_pad, k=k)


def unpack_w4(packed: PackedW4, dtype: torch.dtype = torch.bfloat16,
              rows: slice | None = None) -> torch.Tensor:
    """Dequantize back to [N, K] -- the inverse of pack_w4a16, for the M > MAX_M fallback
    and for numeric references. ``rows`` selects a slice of n-tiles' worth of rows."""
    nt_all, kb, _, _ = packed.wq.shape
    ng = packed.wsz.shape[1]
    k, group = packed.k, packed.k // ng
    t0, t1 = (0, nt_all) if rows is None else (rows.start // 16, -(-rows.stop // 16))
    t1 = min(t1, nt_all)

    byte = packed.wq[t0:t1].view(t1 - t0, kb, 2, 16, 4, 4)   # nt, kb, h, r, s, j
    byte = byte.permute(0, 3, 1, 4, 2, 5)                    # nt, r, kb, s, h, j
    q = torch.stack([byte & 0xF, (byte >> 4) & 0xF], dim=4)  # nt, r, kb, s, half, h, j
    q = (q.reshape(t1 - t0, 16, k) ^ 8).float()              # two's complement -> 0..15

    sz = packed.wsz[t0:t1].permute(0, 2, 1).reshape((t1 - t0) * 16, ng)
    scale = (sz & 0xFFFF).to(torch.int16).view(torch.float16).float()
    zero = -(sz >> 16).to(torch.int16).view(torch.float16).float() - 1024.0

    w = (q.reshape((t1 - t0) * 16, ng, group) - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    w = w.view((t1 - t0) * 16, k).to(dtype)
    if rows is not None:
        w = w[rows.start - t0 * 16:rows.stop - t0 * 16]
    return w


# ---------------------------------------------------------------------------------------
# launch configuration
# ---------------------------------------------------------------------------------------

def _legal(k: int, ntiles: int, wv: int, sk: int, npw: int) -> bool:
    return (wv >= 1 and sk >= 1 and npw in (1, 4)
            and k % (sk * GROUP) == 0
            and wv * sk * _WAVE32 <= _MAX_THREADS
            and wv * npw * sk * 256 * 4 <= _MAX_SMEM
            and wv * npw <= max(1, ntiles))


def _band(m: int) -> int:
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64


# Measured by the original author on the R9700 at N=124160 K=2560, cold, in-graph. The
# whole legal set spans <0.5% at M<=5 -- the kernel is at the DRAM roofline there -- so
# these are picked for being near the top at every M in the band. NT=1 (non-temporal
# weight read) is ahead at every M measured. First legal entry wins.
_CFG_TABLE: dict[int, tuple[tuple[int, int, int, int], ...]] = {
    8: ((1, 10, 1, 1), (1, 4, 1, 1), (1, 2, 1, 1), (1, 1, 1, 1)),
    16: ((1, 10, 1, 1), (1, 4, 1, 1), (1, 2, 1, 1), (1, 1, 1, 1)),
    32: ((1, 4, 4, 1), (1, 2, 4, 1), (1, 1, 4, 1), (1, 4, 1, 1), (1, 1, 1, 1)),
    64: ((1, 4, 4, 1), (1, 2, 4, 1), (1, 1, 4, 1), (1, 4, 1, 1), (1, 1, 1, 1)),
}


@functools.lru_cache(maxsize=None)
def pick_cfg(m: int, n_pad: int, k: int) -> tuple[int, int, int, int]:
    """(WV, SK, NPW, NT) for this M band. Each band's list is tried in order so a
    different shape still gets something legal, and anything illegal falls through to
    the heuristic rather than tripping the C launcher (which throws -> std::terminate
    through ctypes)."""
    ntiles = n_pad // 16
    for wv, sk, npw, nt in _CFG_TABLE.get(_band(m), ()):
        if _legal(k, ntiles, wv, sk, npw):
            return wv, sk, npw, nt
    sk = next((s for s in (4, 2, 1) if k % (s * GROUP) == 0), 1)
    wv = max(1, min(8, _MAX_THREADS // (sk * _WAVE32), _MAX_SMEM // (sk * 1024), ntiles))
    return wv, sk, 1, 1


# ---------------------------------------------------------------------------------------
# the op
# ---------------------------------------------------------------------------------------

def _draft_w4_lmhead_gemm(x: torch.Tensor, wq: torch.Tensor, wsz: torch.Tensor,
                          n: int, k: int, wv: int, sk: int, npw: int,
                          nt: int) -> torch.Tensor:
    """One opaque op: the M <= MAX_M / M > MAX_M branch resolves inside it, so a HIP-graph
    capture freezes the kernel launch and torch.compile never sees the seam."""
    m = x.shape[0]
    n_pad = wq.shape[0] * 16
    if 1 <= m <= MAX_M:
        # Contracts the C launcher would otherwise throw on (-> process abort).
        assert n_pad % 16 == 0 and k % (sk * GROUP) == 0 \
            and wv * sk * _WAVE32 <= _MAX_THREADS and npw in (1, 4) \
            and wv * npw * sk * 256 * 4 <= _MAX_SMEM
        a = x.to(torch.float16).contiguous()
        y = torch.empty((m, n_pad), dtype=torch.bfloat16, device=x.device)
        mb = min(4, (m + 15) // 16)
        _fns()(a.data_ptr(), wq.data_ptr(), wsz.data_ptr(), y.data_ptr(),
               m, k, n_pad, wv, sk, mb, npw, nt,
               torch.cuda.current_stream().cuda_stream)
        return y if n_pad == n else y[:, :n].contiguous()
    # Prefill / wide batch: dequantize in N chunks and use the dense path. The draft never
    # gets here in production (its M is the speculation depth); this is correctness cover.
    packed = PackedW4(wq=wq, wsz=wsz, n=n, n_pad=n_pad, k=k)
    outs = []
    for r0 in range(0, n, _DEQUANT_CHUNK):
        r1 = min(r0 + _DEQUANT_CHUNK, n)
        outs.append(F.linear(x, unpack_w4(packed, x.dtype, slice(r0, r1))))
    return torch.cat(outs, dim=1).to(torch.bfloat16)


def _draft_w4_lmhead_gemm_fake(x: torch.Tensor, wq: torch.Tensor, wsz: torch.Tensor,
                               n: int, k: int, wv: int, sk: int, npw: int,
                               nt: int) -> torch.Tensor:
    return torch.empty((x.size(0), n), dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="draft_w4_lmhead_gemm",
    op_func=_draft_w4_lmhead_gemm,
    fake_impl=_draft_w4_lmhead_gemm_fake,
)


def gemm_w4a16(x: torch.Tensor, packed: PackedW4) -> torch.Tensor:
    """[n_tokens, K] bf16 hidden states -> [n_tokens, N] bf16 logits.

    Safe to call inside a HIP-graph capture: no host sync, no allocation outside the pool."""
    _fns()                                    # binds GROUP / MAX_M from the library
    assert x.dim() == 2 and x.shape[1] == packed.k, \
        f"expected [*, {packed.k}], got {tuple(x.shape)}"
    wv, sk, npw, nt = pick_cfg(min(x.shape[0], MAX_M + 1), packed.n_pad, packed.k)
    return torch.ops.vllm.draft_w4_lmhead_gemm(
        x, packed.wq, packed.wsz, packed.n, packed.k, wv, sk, npw, nt)


def install():
    """Install the W4 draft head into the MTP drafter when DAVETHA_DRAFTER_QUANT=1."""
    if not enabled():
        return
    if not available():
        logger.warning_once("draft_w4_lmhead: not available on this platform; install skipped")
        return

    # (module, class, which LogitsProcessor that drafter scores candidates with)
    targets = []
    for mod_name, cls_name, lp_attr in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP", "logits_processor"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP", "logits_processor"),
        ("vllm.model_executor.models.qwen3_dflash2", "DFlash2Qwen3ForCausalLM",
         "candidate_logits_processor"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            targets.append((getattr(mod, cls_name), lp_attr))
        except Exception:
            continue
    if not targets:
        logger.warning_once("draft_w4_lmhead: no drafter class found; install skipped")
        return

    # Patch LogitsProcessor._apply_head to use the W4 head when available
    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        if not hasattr(LogitsProcessor, "_apply_head_orig"):
            LogitsProcessor._apply_head_orig = LogitsProcessor._apply_head

        def _apply_head_w4(self, lm_head, hidden_states, embedding_bias):
            """Drop-in for LogitsProcessor._apply_head against the int4 W4 draft head."""
            packed = getattr(lm_head, "packed_w4", None)
            if packed is None:
                return self._apply_head_orig(lm_head, hidden_states, embedding_bias)
            return gemm_w4a16(hidden_states, packed)

        LogitsProcessor._apply_head = _apply_head_w4
        logger.info("draft_w4_lmhead: W4 _apply_head installed into LogitsProcessor")
    except Exception as e:
        logger.warning_once("draft_w4_lmhead: LogitsProcessor patch failed: %s", e)

    # Wrap load_weights to quantize the lm_head after loading
    for cls, lp_attr in targets:
        if getattr(cls, "_davetha_w4_head_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig, _lp=lp_attr):
            loaded = _orig(self, weights)
            try:
                lp = getattr(self, _lp, None)
                if lp is None:
                    logger.warning_once("draft_w4_lmhead: no %s on %s", _lp, cls.__name__)
                    return loaded
                lm_head = getattr(self, "lm_head", None)
                if lm_head is None:
                    logger.warning_once("draft_w4_lmhead: no lm_head on %s", cls.__name__)
                    return loaded
                w = lm_head.weight
                if w.dim() != 2:
                    logger.warning_once("draft_w4_lmhead: lm_head weight not 2D on %s", cls.__name__)
                    return loaded
                packed = pack_w4a16(w)
                lm_head.packed_w4 = packed
                logger.info("draft_w4_lmhead: lm_head quantized to W4 (%s)", packed)
            except Exception as e:
                logger.warning_once("draft_w4_lmhead: quantization failed: %s", e)
            return loaded

        cls.load_weights = wrapped
        cls._davetha_w4_head_wrapped = True
    logger.info("draft_w4_lmhead: W4 draft head armed (DAVETHA_DRAFTER_QUANT)")
