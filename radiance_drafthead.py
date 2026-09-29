"""2-bit MTP draft head with an exact rerank, behind RADIANCE_FAST_DRAFT.

Off by default, in which case the drafter uses the stock bf16 head that vLLM shares with the target
model (_maybe_share_lm_head does this unconditionally for an MTP drafter, so it costs no extra
memory). That head reads 1.18 GiB/rank on every draft slot and measures ~2002 us per call.
RADIANCE_FAST_DRAFT=1 replaces it with a 2-bit head at ~473 us including the rerank, for 0.167
GiB/rank.

The head is the largest bandwidth consumer in a decode step: per rank it is x[M,5120] @
W[5120,124160], it runs once per draft slot (up to 8 times per engine step) and it is flat in M from
1 to 72, so the only lever is fewer bytes.

Weights are int2 with an asymmetric per-(row, group-of-128) scale. Four properties make that pay,
and narrower weights alone is not one of them: at group 64 the same kernel measures 781 us, slower
than a 4-bit head.

  * Group 128, not 64. The per-group scale and zero-point arithmetic on the [BLOCK_M, BLOCK_N]
    accumulator is the dominant non-memory term, so halving the group count halves it: 781 -> 417 us
    on identical bytes.
  * Quarter-split packing. Byte j carries k = j, K/4+j, K/2+j and 3K/4+j, so all four 2-bit planes
    feed contiguous k ranges and one byte load serves four contiguous dots. A contiguous-4 layout
    (byte j holding k = 4j..4j+3, one dot per group) measures 676 us instead, because four tile rows
    then share a byte and gather rather than stream.
  * Bit-pattern dequant, hoisted. 0x3F80 | ((b << (5-2q)) & 0x60) reads as the bf16 value 1 + v/4,
    one shift and one mask, with no int-to-float convert and no extract-then-reposition; the uint16
    conversion is hoisted out of the quarter loop, one per tile rather than four. 417 -> 350 us. The
    1.0 bias is exact rather than an approximation: the dot returns sum_k x_k + dot(x,v)/4, and the
    kernel already holds sum_k x_k per group for the zero point, so the contribution collapses to
    (4 s)*p - (4 s + z s)*sum_k x_k, the same two accumulator ops against premultiplied scales.
  * The scale is applied to the accumulator, never to the weight tile. Dequantising [G, BLOCK_N]
    elementwise costs ~400 us instead, 8x more elements to touch.

Accuracy comes from reranking rather than from bits. Each program already holds the maximum of its
64 columns, so it emits the top KCAND of them for free; the top RERANK of those are scored exactly
against the bf16 weight (a few hundred KB against the coarse pass's 0.167 GiB) and written back over
the coarse values. On 8192 draft-head inputs captured from a live serve this matches the exact bf16
argmax on every row, against 22 misses for a 4-bit head at KCAND=1.

KCAND is the lever, not RERANK. Selection emits the top K of each block, so at K=1 a winner that
shares a block with a stronger token is never a candidate at any R, and recall saturates. 2 bits
needs K=8; 4 bits is adequate at K=1. Selecting by block max rather than a token-level topk over the
full row is also the faster choice, 65 us against 105.

Model output cannot move as a result: the draft head decides which tokens are proposed, the target
model verifies every proposal with its own untouched bf16 head on a separate LogitsProcessor
instance, and speculative decoding is distribution-preserving, so a worse draft costs acceptance
rather than a different token. mtp.fc is deliberately left alone: the checkpoint lists it in
modules_to_not_convert alongside the norms, gates, lm_head and embed_tokens, and it is worth only
~0.5% of a decode step.
"""
import os
import sys
import types

import torch

try:
    import triton
    import triton.language as tl
except Exception:                       # pragma: no cover - triton always present in the image
    triton = None

# RADIANCE_FAST_DRAFT gates draft-head quantisation entirely.
#   0 (default): nothing here installs. The drafter uses the stock bf16 head -- which vLLM shares
#                with the target model, so it costs no extra memory but reads 1.18 GiB/rank per draft
#                slot and measures ~2002 us per call.
#   1:           2-bit head, 0.167 GiB/rank, ~473 us per call including the rerank. The rerank makes
#                it exact: on 8192 real draft-head inputs it matches the bf16 argmax on every row.
FAST = os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"

# RADIANCE_DRAFT_HEAD_TOP1 gates the fused top-1 path (B2 design, MTP-DECODE-OPT-PLAN.md).
#   0 (default): the head emits the full bf16 logit row Y; the controller captures confidence and
#                argmax separately (2 capture kernels + a 248320-wide argmax + the Y store).
#   1:           the head emits BM/BI (block maxima) and SM (per-block sum-exp) instead of Y; the
#                top-1 id and confidence are recovered from the candidate arrays with no Y read.
#                Removes 2 capture kernels + the full-vocab argmax + the Y store. The exact winner
#                is always a block maximum, so argmax(Y) == idx[argmax(ex)] whenever the winner is
#                among the rescored candidates. No output can change (speculative decoding verifies
#                every proposal); only draft depth and acceptance can move.
TOP1 = os.environ.get("RADIANCE_DRAFT_HEAD_TOP1", "0") == "1"

# --- local (2026-09-29): pruned DRAFT vocabulary (port of the R9700 branch, REVIEW-LOG §2) --------
# RADIANCE_DRAFT_VOCAB=<file, one token id per line>: the int2 head scores ONLY those rows -- the
# coarse pass is run on the index_selected sub-matrix (~1/5 the reads for a 49k list) and every
# other entry of the returned row is -inf. Drafter-side only: the target verifies with its own
# head, so the OUTPUT cannot change; only acceptance/speed can. Unset (default): nothing changes.
# RADIANCE_DRAFT_EXACTSET=1: only the exactly-reranked candidates stay eligible (the rest of the row
# is -inf), so a SAMPLED draft cannot pick a coarse 2-bit entry. Lossless; argmax unaffected.
VOCAB_FILE = os.environ.get("RADIANCE_DRAFT_VOCAB", "")
EXACT_SET = os.environ.get("RADIANCE_DRAFT_EXACTSET", "0") == "1"
# RADIANCE_DRAFT_FUSED=1: for the EXACT-SET vocab path only (masked row), run the draft head in 6
# launches instead of 15 (int2 candidates, topk, sort, gather, one -inf fill, rerank-scatter). It
# discards the coarse scores, so it requires _radiance_topk_only (EXACTSET, or a candidate
# processor) and no embedding bias; otherwise it is a no-op. Ported from the R9700 branch.
FUSED = os.environ.get("RADIANCE_DRAFT_FUSED", "0") == "1"

GROUP = 128        # weight-quantisation group along K; also the kernel's BLOCK_K
BLOCK_N = 64       # do_bench optimum, and the width of one block-max entry
BITS = 2
# Candidates scored exactly per row. For an ARGMAX caller (mtp) this only caps how many of the
# coarse pass's candidates get rescored, and KCAND is the recall lever -- see the docstring. For a
# TOP-K caller (DFlash2, which asks for selector_top_k=16 candidates per position) it is a HARD
# CEILING instead: _radiance_topk_only blanks every entry the rerank did not touch, so a top-16
# request draws from exactly R tokens. Measured on Qwen3.8-27B + DFlash2-FP8 at ctx 0, R=32:
# acc/draft 1.904 -> 1.804 against the bf16 head, i.e. 2R is too tight a pool for K=16.
RERANK = int(os.environ.get("RADIANCE_DRAFT_RERANK", "32"))
KCAND = 8          # candidates emitted per block; R caps the final count, K feeds it
# Launch geometry, per M band. The head changes regime across the batch sizes one serve produces:
# at M=16 it is memory-bound (427 GB/s, 68% of the DRAM roofline on its 152 MiB of int2) and at
# M=64 it is compute-bound (67 TF/s against a 207 TF/s bf16 WMMA ceiling, only 130 GB/s). The warp
# count that suits one end is wrong at the other, and the penalty is not symmetric: warps=2 is the
# optimum at M=16 (1.10x) and 0.38x at M=64, while warps=8 is the optimum at M=64 (1.07x) and 0.78x
# at M=32. num_stages>1 regresses everywhere. BLOCK_N=64 wins or ties at every M measured.
# Isolated, 124160 x 5120 (one rank's vocab shard), CUDA-graph window, us:
#     M=16   warps 2 / 4 / 8 = 372.4 / 408.8 / 425.6
#     M=32   warps 2 / 4 / 8 = 660.4 / 540.0 / 690.3
#     M=64   warps 2 / 4 / 8 = 3428.4 / 1309.4 / 1218.7
# M is padded to a power of two >= 16 before it gets here, so the bands are 16, 32, 64.
_CFG_BY_M = ((16, {"num_warps": 2, "num_stages": 1}),
             (48, {"num_warps": 4, "num_stages": 1}),
             (64, {"num_warps": 8, "num_stages": 1}))
_CFG = {"num_warps": 4, "num_stages": 1}    # fallback for an M past the table


def _cfg_for(m):
    for lim, cfg in _CFG_BY_M:
        if m <= lim:
            return cfg
    return _CFG


if triton is not None:

    @triton.jit
    def _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC: tl.constexpr, BLOCK_N: tl.constexpr):
        """Top-KC of this block, by successive max-and-mask.

        The exact winner is always the maximum of its own block, so block maxima carry the rerank
        candidates at 1/BLOCK_N the selection width of a token-level top-R; a token-level topk over
        the full row measures 105 us against 65 for this, so the cheap selection is also the fast
        one. KC matters where R does not: R caps how many candidates are finally rescored, but at
        KC=1 a winner sharing a block with a stronger token is never a candidate at any R.
        """
        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        for c in tl.static_range(KC):
            mx = tl.max(masked, axis=1)
            am = tl.argmax(masked, axis=1)
            tl.store(BM + offs_m * (NBLK * KC) + (pid * KC + c), mx)
            tl.store(BI + offs_m * (NBLK * KC) + (pid * KC + c),
                     (pid * BLOCK_N + am).to(tl.int32))
            masked = tl.where(tl.arange(0, BLOCK_N)[None, :] == am[:, None], float("-inf"), masked)

    @triton.jit
    def _draft_head_int2(X, XS, Wq, S, ZS, Y, BM, BI, K: tl.constexpr, N, stride_wq, stride_s,
                         stride_xs, NBLK, KC: tl.constexpr, G: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """2 bits/weight. Packing splits K into quarters, byte j carrying k = j, K/4+j, K/2+j and
        3K/4+j, so all four planes feed contiguous k ranges and one byte load serves four contiguous
        dots. Group 128 rather than 64 is what makes it pay: the per-group accumulator work is the
        dominant non-memory term, and halving the group count moves this kernel 781 -> 417 us."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            # hoisted out of the quarter loop: it is the same tile each time, so one convert not four
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                # 0x3F80 | ((b << (5-2q)) & 0x60) is exactly 0x3F80 | (((b >> 2q) & 3) << 5): one
                # shift and one mask instead of extract-then-reposition. Reads as bf16 1 + v/4, and
                # the 1.0 bias divides out through the group's sum of x, as in the 4-bit path.
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :]).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.load(XS + offs_m * stride_xs + gi).to(tl.float32)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mask_n[None, :])
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)

    @triton.jit
    def _draft_head_int2_top1(X, XS, Wq, S, ZS, BM, BI, SM, K: tl.constexpr, N, stride_wq,
                              stride_s, stride_xs, NBLK, KC: tl.constexpr, G: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """Fused top-1 path (B2): like _draft_head_int2 but emits per-block sum-exp SM instead of
        storing the full logit row Y. The top-1 id and confidence are recovered from the candidate
        arrays on the host: the exact winner is always a block maximum, so argmax(Y) ==
        idx[argmax(ex)] whenever the winner is among the rescored candidates. SM is the sum of
        exp(acc - blockmax) over this block's 64 columns; the host combines it across blocks with
        the block maxima to get the full-row logsumexp."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :]).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.load(XS + offs_m * stride_xs + gi).to(tl.float32)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        # Emit block maxima and indices (same as _emit in the non-fused path)
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)
        # Emit per-block sum-exp: sum of exp(acc - blockmax) over this block's columns
        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        blockmax = tl.max(masked, axis=1)
        sm = tl.sum(tl.exp(masked - blockmax[:, None]), axis=1)
        tl.store(SM + offs_m * NBLK + pid, sm)

    @triton.jit
    def _rerank_exact(X, W, S, IDX, OUT, K: tl.constexpr, stride_w, R: tl.constexpr,
                      BLOCK_K: tl.constexpr, FP8: tl.constexpr):
        """Exact logit for R candidate rows per draft row, straight off the head's own weight:
        bf16 rows, or (FP8) e4m3 rows as raw bytes decoded here times the per-row fp32 scale S --
        the compressed-tensors per-channel FP8 lm_head of the NVFP4 checkpoints. The decode is
        exact bit arithmetic (no Triton fp8 cast, whose gfx12 lowering is not relied on)."""
        m = tl.program_id(0)
        j = tl.program_id(1)
        n = tl.load(IDX + m * R + j)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if FP8:
            sc = tl.load(S + n).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            if FP8:
                b = tl.load(W + n * stride_w + offs).to(tl.int32)
                e = (b >> 3) & 15
                mant = (b & 7).to(tl.float32)
                mag = tl.where(e == 0, mant * 0.001953125,
                               (1.0 + mant * 0.125) * tl.exp2((e - 7).to(tl.float32)))
                wv = tl.where((b & 128) != 0, -mag, mag) * sc
            else:
                wv = tl.load(W + n * stride_w + offs).to(tl.float32)
            acc += tl.load(X + m * K + offs).to(tl.float32) * wv
        tl.store(OUT + m * R + j, tl.sum(acc, axis=0))

    # ---- local (2026-09-29): fused draft head for the exact-set vocab path (RADIANCE_DRAFT_FUSED=1) ----
    # Same math as _draft_head_int2, minus the launches around it: rows past the real m are masked
    # instead of zero-padded, the per-group sums of x come from the tile the kernel already loads
    # instead of a separate fp32 cast + reduce, and the coarse scores are not written (the exact set
    # discards them). Ported from the R9700 branch (radiance_drafthead.fused.diff, REVIEW-LOG §2).
    @triton.jit
    def _draft_head_int2_cand(X, Wq, S, ZS, BM, BI, SM, m_rows, K: tl.constexpr, N, stride_wq, stride_s, NBLK,
                              KC: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, G)
        mask_n = offs_n < N
        mask_m = offs_m < m_rows
        Q: tl.constexpr = K // 4
        NG: tl.constexpr = Q // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, NG):
            b16 = tl.load(Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None],
                          mask=mask_n[None, :], other=0).to(tl.uint16)
            for q in tl.static_range(4):
                if q < 3:
                    wv = (((b16 << (5 - 2 * q)) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                else:
                    wv = (((b16 >> 1) & 0x60) | 0x3F80).to(tl.bfloat16, bitcast=True)
                xv = tl.load(X + offs_m[:, None] * K + (q * Q + g * G + offs_k)[None, :],
                             mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
                gi = q * NG + g
                sv = tl.sum(xv.to(tl.float32), axis=1)
                acc += tl.dot(xv, wv) * tl.load(S + offs_n * stride_s + gi,
                                                mask=mask_n, other=0.0).to(tl.float32)[None, :]
                acc -= sv[:, None] * tl.load(ZS + offs_n * stride_s + gi,
                                             mask=mask_n, other=0.0).to(tl.float32)[None, :]
        _emit(acc, mask_n, BM, BI, offs_m, pid, NBLK, KC, BLOCK_N)
        # Emit per-block sum-exp over the COARSE kept-vocab row (same partial as _draft_head_int2_top1).
        # This decouples the tau-gate confidence from the EXACTSET -inf mask: the returned row keeps
        # only the reranked candidates, so a capture over it would be a softmax over RERANK entries.
        # The conf stashed here is instead a COARSE softmax over the whole kept sub-vocabulary, with
        # an exact-reranked numerator (see _apply_vocab_fused). NOTE: that is NOT the same contract as
        # the unfused-EXACTSET path, whose returned row is -inf outside the RERANK candidates and
        # whose capture is therefore a RERANK-only softmax -- tune RADIANCE_DRAFT_TAU per arm.
        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        blockmax = tl.max(masked, axis=1)
        sm = tl.sum(tl.exp(masked - blockmax[:, None]), axis=1)
        tl.store(SM + offs_m * NBLK + pid, sm)

    @triton.jit
    def _rerank_scatter(X, W, S, IDX, IDS, OUT, K: tl.constexpr, stride_w, NFULL, R: tl.constexpr,
                        BLOCK_K: tl.constexpr, FP8: tl.constexpr):
        """_rerank_exact writing its bf16 logit straight into the full-vocab row at IDS[candidate]
        (the rows around it are -inf from one fill): replaces the -inf fill of the sub row, the cast,
        the scatter and the index_put of the unfused path."""
        m = tl.program_id(0)
        j = tl.program_id(1)
        n = tl.load(IDX + m * R + j)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if FP8:
            sc = tl.load(S + n).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            if FP8:
                b = tl.load(W + n * stride_w + offs).to(tl.int32)
                e = (b >> 3) & 15
                mant = (b & 7).to(tl.float32)
                mag = tl.where(e == 0, mant * 0.001953125,
                               (1.0 + mant * 0.125) * tl.exp2((e - 7).to(tl.float32)))
                wv = tl.where((b & 128) != 0, -mag, mag) * sc
            else:
                wv = tl.load(W + n * stride_w + offs).to(tl.float32)
            acc += tl.load(X + m * K + offs).to(tl.float32) * wv
        tl.store(OUT + m.to(tl.int64) * NFULL + tl.load(IDS + n), tl.sum(acc, axis=0).to(tl.bfloat16))


def _head_matrix(lm_head):
    """(rows [N, K], per-row scale [N] fp32 or None) for the head's weight. A compressed-tensors
    FP8 per-channel lm_head keeps its [N, K] e4m3 storage but exposes it TRANSPOSED ([K, N] view)
    after process_weights_after_loading, with weight_scale [N, 1]; undo the view here."""
    w = getattr(lm_head, "weight", None)
    if w is None or w.dim() != 2:
        return w, None
    if w.dtype == torch.float8_e4m3fn:
        sc = getattr(lm_head, "weight_scale", None)
        if sc is None:
            return None, None
        rows = w if w.shape[0] == sc.numel() else w.t()
        if not rows.is_contiguous() or rows.shape[0] != sc.numel():
            return None, None
        return rows, sc.detach().reshape(-1).float()
    return w, None


def _head_is_empty(rows, scale):
    if scale is None:
        return float(rows.data.abs().max()) == 0.0
    for i in range(0, rows.shape[0], 8192):          # fp8 has no abs(); bytes are enough
        if bool((rows.data[i:i + 8192].view(torch.uint8) != 0).any()):
            return False
    return True


def _pow2_at_least(m):
    """tl.arange needs a power-of-two extent, and tl.dot needs at least 16 rows."""
    n = 16
    while n < m:
        n *= 2
    return n


def _apply_head_int2(self, lm_head, hidden_states, embedding_bias):
    """Drop-in for LogitsProcessor._apply_head against the quantised draft head."""
    if TOP1:
        from vllm.distributed import get_tensor_model_parallel_world_size
        tp = get_tensor_model_parallel_world_size()
        if tp == 1:
            return _apply_head_int2_top1(self, lm_head, hidden_states, embedding_bias)
        # TP>1: the fused top-1 path only works for TP=1 (it returns local-shard-only
        # ids/conf without cross-rank all-gather). Fall back to the standard path.
        if not getattr(self, "_radiance_top1_tp_warned", False):
            self._radiance_top1_tp_warned = True
            sys.stderr.write(f"[radiance] RADIANCE_DRAFT_HEAD_TOP1=1 but TP={tp}: "
                             "fused top-1 path requires TP=1, using standard path\n")
            sys.stderr.flush()
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    m, k = x.shape
    M = _pow2_at_least(m)
    if M != m:
        x = torch.cat([x, x.new_zeros(M - m, k)])
    x = x.contiguous()

    n = self._radiance_n
    nblk = self._radiance_nblk
    ng = k // GROUP
    xs = x.reshape(M, ng, GROUP).float().sum(-1).contiguous()
    y = torch.empty(M, n, dtype=torch.bfloat16, device=x.device)
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    _draft_head_int2[(nblk,)](
        x, xs, self._radiance_wq, self._radiance_scale, self._radiance_zs, y, bm, bi,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))

    w, wsc = _head_matrix(lm_head)
    if w is not None and w.shape == (n, k) and \
            (wsc is not None or w.dtype in (torch.bfloat16, torch.float16)):
        idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
        ex = torch.empty(M, RERANK, dtype=torch.float32, device=x.device)
        if wsc is not None:
            _rerank_exact[(M, RERANK)](x, w.view(torch.uint8), wsc, idx, ex, k, w.stride(0),
                                        R=RERANK, BLOCK_K=512, FP8=True, num_warps=4)
        else:
            _rerank_exact[(M, RERANK)](x, w, x, idx, ex, k, w.stride(0), R=RERANK, BLOCK_K=512,
                                        FP8=False, num_warps=4)
        if getattr(self, "_radiance_topk_only", False):
            # A top-k caller ranks the whole row against itself, so the ~124k entries the rerank
            # did NOT touch are still coarse 2-bit values competing with exact ones -- and a
            # spuriously high coarse entry becomes a candidate. An argmax caller never noticed
            # (the true winner is a reranked block maximum); a top-16 caller sees garbage in
            # every slot the coarse pass over-scored. Make the reranked set the only eligible
            # one, so recall is bounded by RERANK rather than by 2-bit noise.
            y.fill_(float("-inf"))
        y.scatter_(1, idx.long(), ex.to(torch.bfloat16))
    elif not self._radiance_warned:
        # Without the exact weight the coarse pass stands on its own; that is a ~5% top-1 change
        # against bf16, so say so rather than let it pass as the reranked path.
        self._radiance_warned = True
        sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: no bf16 lm_head to rerank against "
                         f"({None if w is None else (tuple(w.shape), w.dtype)}); "
                         "running the coarse 2-bit head unreranked\n")
        sys.stderr.flush()

    if M != m:
        y = y[:m]
    if embedding_bias is not None:
        y = y + embedding_bias
    if self.head_dtype is not None and self.head_dtype != y.dtype:
        y = y.to(self.head_dtype)
    return y.reshape(*hidden_states.shape[:-1], -1)


def _apply_head_int2_top1(self, lm_head, hidden_states, embedding_bias):
    """Fused top-1 path (B2): returns (ids, conf) directly instead of the full logit row Y.
    
    The exact winner is always a block maximum, so argmax(Y) == idx[argmax(ex)] whenever the
    winner is among the rescored candidates. Confidence is recovered from the per-block sum-exp
    partials: Mrow = max_b blockmax_b; S = Σ_b SM_b · exp(blockmax_b − Mrow);
    conf = exp(max(ex) − Mrow) / S. Same top-1 softmax prob, from coarse block partials instead
    of the exact Y row.
    
    Returns (ids [M] int64, conf [M] float32) instead of Y [M, N] bf16.
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    m, k = x.shape
    M = _pow2_at_least(m)
    if M != m:
        x = torch.cat([x, x.new_zeros(M - m, k)])
    x = x.contiguous()

    n = self._radiance_n
    nblk = self._radiance_nblk
    ng = k // GROUP
    xs = x.reshape(M, ng, GROUP).float().sum(-1).contiguous()
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    sm = torch.empty(M, nblk, dtype=torch.float32, device=x.device)
    _draft_head_int2_top1[(nblk,)](
        x, xs, self._radiance_wq, self._radiance_scale, self._radiance_zs, bm, bi, sm,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))

    w, wsc = _head_matrix(lm_head)
    if w is not None and w.shape == (n, k) and \
            (wsc is not None or w.dtype in (torch.bfloat16, torch.float16)):
        # Rerank the top RERANK candidates exactly
        idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
        ex = torch.empty(M, RERANK, dtype=torch.float32, device=x.device)
        if wsc is not None:
            _rerank_exact[(M, RERANK)](x, w.view(torch.uint8), wsc, idx, ex, k, w.stride(0),
                                        R=RERANK, BLOCK_K=512, FP8=True, num_warps=4)
        else:
            _rerank_exact[(M, RERANK)](x, w, x, idx, ex, k, w.stride(0), R=RERANK, BLOCK_K=512,
                                        FP8=False, num_warps=4)
        
        # Recover top-1 id from the reranked candidates
        # The exact winner is always a block maximum, so argmax(Y) == idx[argmax(ex)]
        # Mask out padding candidates (idx >= n - npad)
        npad = getattr(getattr(lm_head, "shard_indices", None), "num_org_vocab_padding", 0) or 0
        valid = idx < (n - npad)
        ex_masked = torch.where(valid, ex, float("-inf"))
        argmax_ex = ex_masked.argmax(dim=1)
        ids = idx.gather(1, argmax_ex.unsqueeze(1)).squeeze(1)
        
        # Recover confidence from per-block sum-exp partials
        # Mrow = max_b blockmax_b (the max of the first KCAND entry per block, which is the block max)
        # SM is the sum of exp(acc - blockmax) over each block's 64 columns
        # S = Σ_b SM_b · exp(blockmax_b − Mrow)
        # conf = exp(max(ex) − Mrow) / S
        blockmax = bm.view(M, nblk, KCAND)[:, :, 0]  # first entry per block is the block max
        Mrow = blockmax.max(dim=1).values
        S = (sm * torch.exp(blockmax - Mrow.unsqueeze(1))).sum(dim=1)
        max_ex = ex_masked.max(dim=1).values
        conf = torch.exp(max_ex - Mrow) / S
    else:
        # Without the exact weight the coarse pass stands on its own. The warn line is emitted
        # once, but the coarse ids/conf MUST be recomputed on every call: _radiance_warned stays
        # True after the first, and returning unbound locals would raise UnboundLocalError.
        if not self._radiance_warned:
            self._radiance_warned = True
            sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: no bf16 lm_head to rerank against "
                             f"({None if w is None else (tuple(w.shape), w.dtype)}); "
                             "running the coarse 2-bit head unreranked\n")
            sys.stderr.flush()
        # Fall back to coarse argmax
        npad = getattr(getattr(lm_head, "shard_indices", None), "num_org_vocab_padding", 0) or 0
        valid = bi < (n - npad)
        bm_masked = torch.where(valid, bm, float("-inf"))
        argmax_bm = bm_masked.argmax(dim=1)
        ids = bi.gather(1, argmax_bm.unsqueeze(1)).squeeze(1)
        blockmax = bm.view(M, nblk, KCAND)[:, :, 0]
        Mrow = blockmax.max(dim=1).values
        S = (sm * torch.exp(blockmax - Mrow.unsqueeze(1))).sum(dim=1)
        conf = torch.exp(bm_masked.max(dim=1).values - Mrow) / S

    if M != m:
        ids = ids[:m]
        conf = conf[:m]
    
    start = getattr(getattr(lm_head, "shard_indices", None), "org_vocab_start_index", 0) or 0
    return ids + start, conf


def _apply_head_lazy(self, lm_head, hidden_states, embedding_bias):
    """First real call quantises, then hands over to the quantised path for good.

    Needed because the weight a drafter scores against may not exist yet when load_weights
    returns; here it is the argument, so it is guaranteed populated.
    """
    _rows, _sc = _head_matrix(lm_head)
    if _rows is None or _head_is_empty(_rows, _sc):
        # Still empty on a real call: something is wrong, but a coarse head would be silently
        # catastrophic, so fall back to the stock GEMM rather than guess.
        return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
    status = _quantize_head_now(self, lm_head)
    sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD (lazy): {status}\n")
    sys.stderr.flush()
    return self._apply_head(lm_head, hidden_states, embedding_bias)


class _SubHead:
    """Kept rows, in the shape _head_matrix reads: .weight [n_sub, K] (+ optional .weight_scale)."""

    def __init__(self, weight, weight_scale=None):
        self.weight = weight
        if weight_scale is not None:
            self.weight_scale = weight_scale


def _apply_head_vocab(self, lm_head, hidden_states, embedding_bias):
    """RADIANCE_DRAFT_VOCAB: build the sub-head once, then score only its rows.

    The sub-matrix is index_selected from the SAME lm_head the full head would use, quantised by
    _quantize_head_now, and the returned row is expanded to the full vocabulary with -inf outside
    the kept ids. Drafter-side only -- the target still verifies with its own head. Port of the
    R9700 branch's _apply_head_vocab (REVIEW-LOG §2).
    """
    sub = getattr(self, "_dv_sub", None)
    if sub is None:
        rows, rsc = _head_matrix(lm_head)
        if rows is None or _head_is_empty(rows, rsc):
            return type(self)._apply_head(self, lm_head, hidden_states, embedding_bias)
        ids = self._dv_ids.to(rows.device)
        w_sub = rows.index_select(0, ids).contiguous()
        s_sub = rsc.index_select(0, ids).reshape(-1, 1).contiguous() if rsc is not None else None
        sub = _SubHead(w_sub, s_sub)
        status = _quantize_head_now(self, sub)   # rebinds _apply_head to the int2 path; take it back
        self._apply_head = types.MethodType(_apply_head_vocab, self)
        self._dv_sub, self._dv_ids_dev, self._dv_nfull = sub, ids, rows.shape[0]
        sys.stderr.write(f"[radiance] DRAFT_VOCAB: {ids.numel()} of {rows.shape[0]} rows -> {status}\n")
        sys.stderr.flush()
    if FUSED and getattr(self, "_radiance_topk_only", False) and embedding_bias is None:
        if not getattr(self, "_radiance_path_logged", False):
            self._radiance_path_logged = True
            sys.stderr.write("[radiance] draft head path: FUSED (EXACTSET)\n")
            sys.stderr.flush()
        # Exact-set path only: the fused kernels write just the reranked candidates, so the coarse
        # scores (which the unfused path keeps for argmax recall) are discarded. Guarded above.
        return _apply_vocab_fused(self, sub, hidden_states)
    if not getattr(self, "_radiance_path_logged", False):
        self._radiance_path_logged = True
        _p = "unfused-EXACTSET" if getattr(self, "_radiance_topk_only", False) else "unfused"
        sys.stderr.write(f"[radiance] draft head path: {_p} "
                         f"(preconf={getattr(self, '_radiance_conf_precomputed', False)})\n")
        sys.stderr.flush()
    eb = embedding_bias.index_select(0, self._dv_ids_dev) if embedding_bias is not None else None
    y_sub = _apply_head_int2(self, sub, hidden_states, eb)
    y = torch.full((*y_sub.shape[:-1], self._dv_nfull), float("-inf"),
                   dtype=y_sub.dtype, device=y_sub.device)
    y[..., self._dv_ids_dev] = y_sub
    return y


def _apply_vocab_fused(self, sub, hidden_states):
    """RADIANCE_DRAFT_FUSED=1: the exact-set vocab path in 6 launches instead of 15.

    Output is identical to the unfused path's: -inf everywhere except the RERANK exactly-scored
    candidates (bf16). See _draft_head_int2_cand / _rerank_scatter. Ported from the R9700 branch.
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    m, k = x.shape
    M = _pow2_at_least(m)
    n, nblk = self._radiance_n, self._radiance_nblk
    bm = torch.empty(M, nblk * KCAND, dtype=torch.float32, device=x.device)
    bi = torch.empty(M, nblk * KCAND, dtype=torch.int32, device=x.device)
    sm = torch.empty(M, nblk, dtype=torch.float32, device=x.device)
    _draft_head_int2_cand[(nblk,)](
        x, self._radiance_wq, self._radiance_scale, self._radiance_zs, bm, bi, sm, m,
        k, n, self._radiance_wq.stride(0), self._radiance_scale.stride(0), nblk, KCAND,
        G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))
    idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
    w, wsc = _head_matrix(sub)
    nfull = self._dv_nfull
    out = torch.full((m, nfull), float("-inf"), dtype=torch.bfloat16, device=x.device)
    if wsc is not None:
        _rerank_scatter[(m, RERANK)](x, w.view(torch.uint8), wsc, idx, self._dv_ids_dev, out, k,
                                     w.stride(0), nfull, R=RERANK, BLOCK_K=512, FP8=True, num_warps=4)
    else:
        _rerank_scatter[(m, RERANK)](x, w, x, idx, self._dv_ids_dev, out, k, w.stride(0), nfull,
                                     R=RERANK, BLOCK_K=512, FP8=False, num_warps=4)
    if self.head_dtype is not None and self.head_dtype != out.dtype:
        out = out.to(self.head_dtype)
    # Tau-gate confidence decoupled from the EXACTSET mask (see the kernel comment): recover the
    # top-1 softmax prob from the COARSE per-block partials, as _apply_head_int2_top1 does, and stash
    # it for _local_draft. The returned row is unchanged (-inf outside the reranked set). Contract
    # note: this is a COARSE kept-vocab softmax with an exact-reranked numerator, so it is NOT the
    # same as capturing the returned row (that would be a RERANK-only softmax, and unfused-EXACTSET
    # does exactly that); RADIANCE_DRAFT_TAU must be tuned per arm.
    blockmax = bm.view(M, nblk, KCAND)[:, :, 0]
    brow = blockmax.max(dim=1).values
    S = (sm * torch.exp(blockmax - brow.unsqueeze(1))).sum(dim=1)
    max_ex = out.max(dim=1).values.float()          # max over the R reranked (finite) candidates
    conf = torch.exp(max_ex - brow[:m]) / S[:m]
    self._radiance_last_conf = torch.nan_to_num(conf, nan=0.0, posinf=0.0).clamp_(0.0, 1.0)
    if not getattr(self, "_radiance_fused_logged", False):
        self._radiance_fused_logged = True
        sys.stderr.write(f"[radiance] fused conf stashed: min={float(self._radiance_last_conf.min()):.4g} "
                         f"max={float(self._radiance_last_conf.max()):.4g} n={int(self._radiance_last_conf.numel())}\n")
        sys.stderr.flush()
    return out.reshape(*hidden_states.shape[:-1], -1)


def _quantize_draft_head(mtp, lp_attr="logits_processor"):
    """Quantise the head a drafter scores with, and rebind that LogitsProcessor's _apply_head.

    lp_attr names which LogitsProcessor to hook. MTP has exactly one; DFlash2 keeps a separate
    `candidate_logits_processor` for candidate generation, and hooking THAT is what confines the
    approximation to the draft path -- the target samples through its own instance and is
    untouched. Both share the same lm_head weight, which is why the bf16 copy has to stay: it is
    the target's, and it is also what the rerank scores against.
    """
    lm_head = getattr(mtp, "lm_head", None)
    lp = getattr(mtp, lp_attr, None)
    if lm_head is None or lp is None or not hasattr(lm_head, "weight"):
        return f"no lm_head/{lp_attr}"
    w = lm_head.weight
    rows, rsc = _head_matrix(lm_head)
    if rows is None or w.dim() != 2 or \
            (rsc is None and w.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return f"unsupported draft-head weight {tuple(w.shape)} {w.dtype}"
    lp._radiance_topk_only = (lp_attr == "candidate_logits_processor") or EXACT_SET
    # RADIANCE_DRAFT_VOCAB: score only the kept rows. Built lazily on the first real call (the
    # tensor is only guaranteed populated then), exactly like the lazy full-head path below.
    if VOCAB_FILE:
        if TOP1:
            return "DRAFT_VOCAB set but RADIANCE_DRAFT_HEAD_TOP1=1 (fused top-1 returns ids/conf); vocab needs it off -- full head kept"
        try:
            ids = sorted({int(t) for t in open(VOCAB_FILE).read().split()})
        except Exception as e:
            return f"DRAFT_VOCAB unreadable ({e!r}); full head kept"
        if not ids:
            return "DRAFT_VOCAB empty; full head kept"
        bad = [i for i in ids if i < 0 or i >= rows.shape[0]]
        if bad:
            return f"DRAFT_VOCAB has {len(bad)} ids outside [0,{rows.shape[0]}); full head kept"
        nblk_sub = (len(ids) + BLOCK_N - 1) // BLOCK_N
        if nblk_sub * KCAND < RERANK:
            return (f"DRAFT_VOCAB too small ({len(ids)} ids -> {nblk_sub * KCAND} candidates "
                    f"< RERANK {RERANK}); full head kept")
        lp._dv_ids = torch.tensor(ids, dtype=torch.long)
        lp._dv_sub = None
        lp._apply_head = types.MethodType(_apply_head_vocab, lp)
        # FUSED+EXACTSET returns a -inf-masked row; the conf it stashes is computed from the coarse
        # kept-vocab partials so the tau gate keeps working. Tell _local_draft to use it.
        lp._radiance_conf_precomputed = bool(FUSED and EXACT_SET)
        lp._radiance_last_conf = None
        return (f"DRAFT_VOCAB: {len(ids)} of {rows.shape[0]} rows (EXACTSET={EXACT_SET}); "
                f"quantised on first call")
    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target's tensor shared in
    # AFTER load_weights returns, so at this point the parameter is not yet the weight we want.
    # Quantising it yields a garbage head, and the failure is silent and total: the serve comes up,
    # text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --
    # which reads as a plausible accuracy verdict on the quantisation.
    #
    # This deferred only when the parameter sniffed as all-zero, which was load-bearing and wrong:
    # it held only because a fresh torch.empty() happened to hand back zeroed pages. Measured
    # 2026-09-18 -- a loader that allocated and freed fp32 temporaries before the drafter loaded
    # dirtied the caching allocator, the not-yet-shared parameter came back non-zero, the sniff
    # said "populated", and acceptance fell 7.63 -> 1.01 (draft accept 94.7% -> 0.1%, decode
    # 320 -> 47 tok/s) with GSM8K unmoved at 97.2%, i.e. invisible to every accuracy gate.
    #
    # So defer unconditionally. _apply_head_lazy takes the head as an ARGUMENT and is therefore
    # guaranteed the populated tensor; it already falls back to the stock GEMM if that is somehow
    # still empty. Costs one stock-GEMM call on the first draft. This is the branch every dflash
    # serve already took anyway -- the sniff returned True in production -- so it narrows to the
    # exercised path rather than adding a new one.
    lp._apply_head = types.MethodType(_apply_head_lazy, lp)
    return "deferred to first use (the weight a drafter scores against is shared in later)"


# int2 buffers keyed by the bf16 weight they were derived from. DFlash2 shares ONE lm_head between
# the drafter's candidate_logits_processor and the target's logits_processor, so when both are armed
# the second one must reuse the first's packing rather than spend another 0.167 GiB/rank -- at
# GPU_UTIL 0.98 that second copy comes straight out of the KV cache. Keyed by data_ptr because the
# two LogitsProcessors hold the same tensor object anyway; a weight that moved would get a new
# pointer and be repacked, which is the safe direction.
_HEAD_CACHE: dict = {}


def _quantize_head_now(lp, lm_head):
    w, wsc = _head_matrix(lm_head)          # [N, K] rows (bf16, or e4m3 + per-row scale)
    n, k = w.shape
    if k % (2 * GROUP):
        return f"hidden size {k} not a multiple of {2 * GROUP}"

    cached = _HEAD_CACHE.get((w.data_ptr(), n, k))
    if cached is not None:
        (lp._radiance_wq, lp._radiance_scale, lp._radiance_zs,
         lp._radiance_n, lp._radiance_nblk) = cached
        lp._radiance_warned = False
        lp._apply_head = types.MethodType(_apply_head_int2, lp)
        return (f"draft head ({n}, {k}) reusing the int{BITS} packing already built for this "
                f"lm_head, {KCAND} cand/block, rerank top-{RERANK} exact")

    # Asymmetric min/max at BITS, quarter-split packing (see the module docstring).
    # Quantise in row chunks: a whole-tensor fp32 intermediate is 2.5 GiB here, and the caching
    # allocator keeps that reservation for the rest of the process, which comes straight out of
    # the VRAM the KV cache could have used.
    per_byte = 8 // BITS
    lv = (1 << BITS) - 1
    packed = torch.empty(n, k // per_byte, dtype=torch.uint8, device=w.device)
    scale = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    zs = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=w.device)
    CH = 8192
    for i in range(0, n, CH):
        j = min(i + CH, n)
        wg = w.data[i:j].to(torch.float32)
        if wsc is not None:
            wg = wg * wsc[i:j, None]
        wg = wg.reshape(j - i, k // GROUP, GROUP)
        lo, hi = wg.amin(dim=2), wg.amax(dim=2)
        sc = ((hi - lo) / lv).clamp(min=1e-8)
        zp = torch.round(-lo / sc).clamp(0, lv)
        q = torch.round(wg / sc[:, :, None] + zp[:, :, None]).clamp(0, lv).to(torch.uint8)
        q = q.reshape(j - i, k)
        if BITS == 2:
            # quarter-split: byte j carries k = j, K/4+j, K/2+j, 3K/4+j
            Q = k // 4
            packed[i:j] = (q[:, :Q] | (q[:, Q:2 * Q] << 2)
                           | (q[:, 2 * Q:3 * Q] << 4) | (q[:, 3 * Q:] << 6))
        else:
            packed[i:j] = q[:, : k // 2] | (q[:, k // 2:] << 4)
        # The kernel's dot returns sum_k x_k + dot(x,v)/16 because every weight carries the bf16
        # 1.0 bias, so s*dot(x,q) - zp*s*sum_k x_k becomes (16 s)*p - (16 s + zp s)*sum_k x_k.
        # Both premultiplied here, which also saves a load in the inner loop.
        # the bf16 bias is 1 + v/(2^m) with m the mantissa slot used, so the premultiplier is
        # 2^m: 16 for the 4-bit path (v << 3), 4 for the 2-bit one (v << 5)
        bias = 4.0 if BITS == 2 else 16.0
        scale[i:j] = (bias * sc).to(torch.bfloat16)
        zs[i:j] = (bias * sc + zp * sc).to(torch.bfloat16)
        del wg, lo, hi, sc, zp, q
    lp._radiance_wq = packed
    lp._radiance_scale = scale
    lp._radiance_zs = zs
    lp._radiance_n = n
    lp._radiance_nblk = (n + BLOCK_N - 1) // BLOCK_N
    lp._radiance_warned = False
    lp._apply_head = types.MethodType(_apply_head_int2, lp)
    _HEAD_CACHE[(w.data_ptr(), n, k)] = (packed, scale, zs, n, lp._radiance_nblk)
    torch.cuda.empty_cache()

    stored = (lp._radiance_wq.numel()
              + (lp._radiance_scale.numel() + lp._radiance_zs.numel()) * 2)
    # The bf16 weight is deliberately left in place: the rerank scores against it, and leaving it
    # lets vLLM's _maybe_share_lm_head point the MTP at the target's copy instead of keeping a
    # second one. Blanking it here (as the fp8 head did) defeats that share.
    return (f"draft head ({n}, {k}) {'fp8' if wsc is not None else 'bf16'} -> int{BITS} g{GROUP} asym "
            f"({stored / 2**30:.2f} GiB/rank), {KCAND} cand/block, rerank top-{RERANK} exact")


def install():
    if not FAST:
        # stock bf16 head; nothing patched, nothing quantised
        return
    if triton is None:
        sys.stderr.write("[radiance] quantised draft head off: no triton\n")
        return

    # (module, class, which LogitsProcessor that drafter scores candidates with)
    targets = []
    for mod_name, cls_name, lp_attr in (
        ("vllm.model_executor.models.qwen3_5_mtp", "Qwen3_5MTP", "logits_processor"),
        ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP", "logits_processor"),
        # DFlash2 reaches the head through get_top_k_tokens, which calls _apply_head like
        # everything else. Its lm_head is not in the drafter checkpoint -- it is shared in from
        # the target after load_weights -- so this one always takes the lazy path.
        ("vllm.model_executor.models.qwen3_dflash2", "DFlash2Qwen3ForCausalLM",
         "candidate_logits_processor"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            targets.append((getattr(mod, cls_name), lp_attr))
        except Exception:
            continue
    if not targets:
        sys.stderr.write("[radiance] quantised draft head off: no drafter class found\n")
        return

    for cls, lp_attr in targets:
        if getattr(cls, "_radiance_quant_head_wrapped", False):
            continue
        orig = cls.load_weights

        def wrapped(self, weights, _orig=orig, _lp=lp_attr):
            loaded = _orig(self, weights)
            try:
                status = _quantize_draft_head(self, _lp)
            except Exception as e:
                status = f"FAILED, bf16 head kept: {e!r}"
            sys.stderr.write(f"[radiance] INT{BITS}_DRAFT_HEAD: {status}\n")
            sys.stderr.flush()
            return loaded

        cls.load_weights = wrapped
        cls._radiance_quant_head_wrapped = True
    sys.stderr.write(f"[radiance] int{BITS} draft head armed (RADIANCE_FAST_DRAFT)\n")
    sys.stderr.flush()
