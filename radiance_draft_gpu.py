#!/usr/bin/env python3
"""GPU kernels for the RADIANCE draft controller (Triton, gfx1201).

Three device kernels feed the per-slot decision that gates the MTP draft loop, keeping its inputs off
the full-vocab softmax path:

  capture_gpu : per draft slot, one split-V online logsumexp over the vocab logits -> the top-1
                confidence, with NO full-vocab softmax and NO topk. Result stays on device. The
                confidence is the softmax maximum, exp(max - M) / sum(exp(l - M)) = 1 / S; bit-exact vs
                the torch reduction.

  match_gpu   : batched longest-suffix n-gram match over a GPU context mirror -> the TOP-2 verbatim
                continuations and their match lengths, all on device. Full context by default; an
                optional per-row search window bounds the scan at long context, and a cross-request
                mode searches every active row rather than only the request's own row. Bit identical
                to a CPU longest-suffix matcher (see ngram_draft_selftest.py for the reference).

The tiny per-slot threshold decision runs on the host (top-1 confidence, match length, and the
drafted token id, one coalesced copy per SLOT and one per STEP), which is what lets the controller
short-circuit the host-launched forward loop. Both kernels compute exactly the quantities the policy
uses, so drafting output is unchanged; the target's rejection sampler still verifies every proposal.

Encoded candidate key (int64):  [ match_len | row_id | end_pos ]
  end_pos   in bits [0, _SH)          (positions within a context row, < 2**20)
  row_id    in bits [_SH, _SH+_RH)    (which context row the continuation is read from)
  match_len in bits [_SH+_RH, 64)     (longest suffix match length)
A zero key means "no match". Ties on length break toward the more recent end position because
end_pos is in the low bits and the compare is on the whole packed integer.
"""
import torch
import triton
import triton.language as tl

_MAXL, _MIN, _SH, _RH = 32, 3, 20, 7     # default match horizon; end_pos occupies _SH bits, row _RH bits
_LSH = _SH + _RH                         # match length starts above the (row, end_pos) payload
_META = 7                                # pack tail: clen1, mlen1, clen2, mlen2, window_miss, rec1, rec2
_BLOCK = 512                             # scan positions per program
_T2_BLOCK = 256                          # top-2 reduction chunk
_BIG = 1 << 30                           # recency sentinel for a cross-request match (gate disabled)


# ----- n-gram matcher ---------------------------------------------------------
@triton.jit
def _match_scan(ctx, nA, ML, cand, base, NBLK,
                MAXL: tl.constexpr, MIN: tl.constexpr, SH: tl.constexpr, RH: tl.constexpr,
                BLOCK: tl.constexpr):
    """Self match: longest suffix of row i that ended earlier within the same row.

    Grid (B, NBLK). `base[i]` is the window lower bound (0 disables it). Writes the block's top-2
    keys into cand[i*(2*NBLK) + 2*blk + {0,1}]; _match_top2 then reduces the per-block top-2s to the
    row's top-2. Two slots per block are required for exactness: the row's 2nd best can sit in the
    same block as its 1st, so a single per-block max would discard it."""
    i = tl.program_id(0)
    blk = tl.program_id(1)
    n = tl.load(nA + i)
    row = ctx + i * ML
    b = tl.load(base + i)
    q = b + blk * BLOCK + tl.arange(0, BLOCK)
    alive = q < (n - 1)
    L = tl.zeros((BLOCK,), tl.int32)
    for k in range(MAXL):
        sk = n - 1 - k                       # suffix position; guard >=0 for short contexts (n<MAXL)
        sfx = tl.load(row + sk, mask=sk >= 0, other=-2)   # else row[negative] page-faults the GPU
        qi = q - k
        m = alive & (qi >= b) & (sk >= 0)
        c = tl.load(row + qi, mask=m, other=-1)
        alive = m & (c == sfx)
        L = L + alive.to(tl.int32)
    key = tl.where(L >= MIN,
                   (L.to(tl.int64) << (SH + RH)) | (i.to(tl.int64) << SH) | q.to(tl.int64),
                   0)
    m1 = tl.max(key, 0)
    m2 = tl.max(tl.where(key == m1, 0, key), 0)   # keys are unique per q, so this is the 2nd best
    o = i * (2 * NBLK) + 2 * blk
    tl.store(cand + o, m1)
    tl.store(cand + o + 1, m2)


@triton.jit
def _match_scan_x(ctx, nA, ML, cand, NBLK, NROWS,
                  MAXL: tl.constexpr, MIN: tl.constexpr, SH: tl.constexpr, RH: tl.constexpr,
                  BLOCK: tl.constexpr):
    """Cross-request match: the suffix of row i searched against row j (any active row).

    Grid (B, B, NBLK). Full context (no window). Writes the block's top-2 keys into
    cand[i*(2*NROWS*NBLK) + j*(2*NBLK) + 2*blk + {0,1}]. The continuation is read from row j in
    _match_gather, so it can never cross into another row."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    blk = tl.program_id(2)
    n = tl.load(nA + i)
    nj = tl.load(nA + j)
    rowi = ctx + i * ML
    rowj = ctx + j * ML
    q = blk * BLOCK + tl.arange(0, BLOCK)
    alive = q < (nj - 1)
    L = tl.zeros((BLOCK,), tl.int32)
    for k in range(MAXL):
        sk = n - 1 - k
        sfx = tl.load(rowi + sk, mask=sk >= 0, other=-2)
        qi = q - k
        m = alive & (qi >= 0) & (sk >= 0)
        c = tl.load(rowj + qi, mask=m, other=-1)
        alive = m & (c == sfx)
        L = L + alive.to(tl.int32)
    key = tl.where(L >= MIN,
                   (L.to(tl.int64) << (SH + RH)) | (j.to(tl.int64) << SH) | q.to(tl.int64),
                   0)
    m1 = tl.max(key, 0)
    m2 = tl.max(tl.where(key == m1, 0, key), 0)
    o = i * (2 * NROWS * NBLK) + j * (2 * NBLK) + 2 * blk
    tl.store(cand + o, m1)
    tl.store(cand + o + 1, m2)


@triton.jit
def _match_top2(cand, key1, key2, NC, BN: tl.constexpr):
    """Per row, the top-2 distinct keys over NC candidate slots, into key1/key2.

    The candidates are the per-block top-2s emitted by _match_scan/_match_scan_x, so this reduction
    is exact: the row's 1st and 2nd best each appear in some block's top-2. The first chunk is peeled
    so k1/k2 start as scalars from tl.max rather than a 0-d initialiser."""
    i = tl.program_id(0)
    r0 = tl.arange(0, BN)
    v0 = tl.load(cand + i * NC + r0, mask=r0 < NC, other=0)
    k1 = tl.max(v0, 0)
    k2 = tl.max(tl.where((v0 == k1) & (r0 < NC), 0, v0), 0)
    for off in tl.range(BN, NC, BN):
        r = off + tl.arange(0, BN)
        mask = r < NC
        v = tl.load(cand + i * NC + r, mask=mask, other=0)
        b1 = tl.max(v, 0)
        b2 = tl.max(tl.where((v == b1) & mask, 0, v), 0)
        n1 = tl.maximum(k1, b1)
        n2 = tl.maximum(tl.minimum(k1, b1), tl.maximum(k2, b2))
        k1 = n1
        k2 = n2
    tl.store(key1 + i, k1)
    tl.store(key2 + i, k2)


@triton.jit
def _match_gather(ctx, nA, ML, key1, key2, pack, base,
                  SH: tl.constexpr, RH: tl.constexpr, NSPEC: tl.constexpr):
    """Decode the two keys per row into [cont1 | cont2 | meta] in `pack`.

    pack layout (int32), stride 2*NSPEC + 7:
      [0, NSPEC)          cont1 continuation tokens (-1 beyond clen1)
      [NSPEC, 2*NSPEC)    cont2 continuation tokens
      [2*NSPEC + 0..6]    clen1, mlen1, clen2, mlen2, window_miss, rec1, rec2
    A continuation is read from the key's row j, bounded by nA[j], so it never crosses a row
    boundary. `window_miss` is 1 when a positive window was used for the row and no candidate was
    found (so the host can fall back to a full scan without a second sync). `rec` is the match's
    distance back from THIS row's end (nA[i]-1-q) for a self match, or the sentinel _BIG for a
    cross-request match, where recency is not meaningful."""
    i = tl.program_id(0)
    t = tl.arange(0, NSPEC)
    stride = 2 * NSPEC + 7
    out = pack + i * stride
    qmask = (1 << SH) - 1
    rmask = (1 << RH) - 1
    ni = tl.load(nA + i)
    k1 = tl.load(key1 + i)
    k2 = tl.load(key2 + i)
    for c in range(2):
        k = k1 if c == 0 else k2
        has = k != 0
        q = (k & qmask).to(tl.int32)
        j = ((k >> SH) & rmask).to(tl.int32)
        L = (k >> (SH + RH)).to(tl.int32)
        start = q + 1
        nj = tl.load(nA + j)
        valid = has & (start + t < nj)
        row = ctx + j * ML
        tok = tl.load(row + start + t, mask=valid, other=-1)
        tl.store(out + c * NSPEC + t, tl.where(valid, tok, -1))
        base_off = 2 * NSPEC + 2 * c
        tl.store(out + base_off + 0, tl.sum(valid.to(tl.int32), 0))
        tl.store(out + base_off + 1, tl.where(has, L, 0))
        rec = tl.where((j == i) & has, ni - 1 - q, 1 << 30)
        tl.store(out + 2 * NSPEC + 5 + c, rec)
    win_miss = (tl.load(base + i) > 0) & (k1 == 0)
    tl.store(out + 2 * NSPEC + 4, win_miss.to(tl.int32))


@triton.jit
def _match_count(ctx, nA, key1, base, occ_out, agree_out, ML, NBLK,
                 MAXL: tl.constexpr, SH: tl.constexpr, RH: tl.constexpr, BLOCK: tl.constexpr):
    """Frequency of the top-1 matched suffix's continuation (F6, cont.79).

    For each row, `key1` holds the top-1 match (length Lref, end position qref). This kernel scans the
    row's own context once and, for every OTHER occurrence of the Lref-length suffix (L >= Lref), counts
    how many share the reference continuation's first token. Arctic Suffix Decoding gates on this
    continuation probability (`min_token_prob`); here it lets a strongly repeated continuation override
    an MTP disagreement while a boilerplate suffix (many occurrences, divergent continuations) is
    rejected. Deterministic (no atomics): one program per row loops the blocks. occ/agree EXCLUDE the
    reference occurrence (the host adds 1 for it when forming the probability)."""
    i = tl.program_id(0)
    n = tl.load(nA + i)
    row = ctx + i * ML
    b = tl.load(base + i)
    k = tl.load(key1 + i)
    has = k != 0
    Lref = (k >> (SH + RH)).to(tl.int32)
    qref = (k & ((1 << SH) - 1)).to(tl.int32)
    cref = tl.load(row + qref + 1, mask=has & (qref + 1 < n), other=-1)
    occ = 0
    agree = 0
    for blk in tl.range(0, NBLK):
        q = b + blk * BLOCK + tl.arange(0, BLOCK)
        alive = q < (n - 1)
        L = tl.zeros((BLOCK,), tl.int32)
        for kk in range(MAXL):
            sk = n - 1 - kk
            sfx = tl.load(row + sk, mask=sk >= 0, other=-2)
            qi = q - kk
            m = alive & (qi >= b) & (sk >= 0)
            c = tl.load(row + qi, mask=m, other=-1)
            alive = m & (c == sfx)
            L = L + alive.to(tl.int32)
        hit = has & (q < (n - 1)) & (q >= b) & (L >= Lref) & (q != qref)
        cn = tl.load(row + q + 1, mask=hit, other=-1)
        occ += tl.sum(hit.to(tl.int32), 0)
        agree += tl.sum((hit & (cn == cref)).to(tl.int32), 0)
    tl.store(occ_out + i, occ)
    tl.store(agree_out + i, agree)


def _nblk(nmax, window):
    span = nmax if not window else min(nmax, window)
    return max(1, triton.cdiv(span, _BLOCK))


def make_match_buffers(B, nspec, nc, device):
    """Scratch for the matcher, sized for `nc` candidate SLOTS per row (two per 512-position block)
    and reused across steps. Caller keeps the dict and passes it back to match_gpu."""
    return {
        "nc": nc,
        "cand": torch.empty(B * nc, dtype=torch.int64, device=device),
        "key1": torch.empty(B, dtype=torch.int64, device=device),
        "key2": torch.empty(B, dtype=torch.int64, device=device),
        "base": torch.zeros(B, dtype=torch.int32, device=device),
        "pack": torch.empty(B, 2 * nspec + _META, dtype=torch.int32, device=device),
        "occ": torch.empty(B, dtype=torch.int32, device=device),
        "agree": torch.empty(B, dtype=torch.int32, device=device),
    }


def match_gpu(ctx, n_arr, nspec, nmax, base, active_nc, cross, bufs, maxl=_MAXL):
    """Top-2 longest-suffix n-gram match.

    ctx [B, maxlen] int32 GPU context mirror; n_arr [B] int32 lengths; nmax = max length (from the
    host-side num_tokens_no_spec, so no D2H); base [B] int32 window lower bounds; `active_nc` is the
    number of candidate SLOTS to reduce (two per block, so `2*nblk`; bufs may be sized larger for a
    full-scan fallback); cross selects the cross-request scan; `maxl` bounds the compared suffix
    length (the continuation is always capped at nspec). Returns `pack` [B, 2*nspec+7] int32 on
    device (one .cpu() gives the continuations AND the metadata)."""
    B, ML = ctx.shape
    if cross:
        nblk = active_nc // (2 * B)
        _match_scan_x[(B, B, nblk)](ctx, n_arr, ML, bufs["cand"], nblk, B,
                                    maxl, _MIN, _SH, _RH, BLOCK=_BLOCK)
    else:
        nblk = active_nc // 2
        _match_scan[(B, nblk)](ctx, n_arr, ML, bufs["cand"], base, nblk,
                               maxl, _MIN, _SH, _RH, BLOCK=_BLOCK)
    _match_top2[(B,)](bufs["cand"], bufs["key1"], bufs["key2"], active_nc, BN=_T2_BLOCK)
    _match_gather[(B,)](ctx, n_arr, ML, bufs["key1"], bufs["key2"], bufs["pack"], base,
                        _SH, _RH, nspec)
    return bufs["pack"]


def match_count(ctx, n_arr, key1, base, occ, agree, active_nc, maxl=_MAXL):
    """Count occurrences of the top-1 matched suffix and how many share its continuation token.

    `key1` is `bufs["key1"]` after `match_gpu`; `occ`/`agree` are `bufs["occ"]`/`bufs["agree"]`
    (written here, so zero them first). `active_nc = 2 * nblk` as in `match_gpu`."""
    B, ML = ctx.shape
    nblk = active_nc // 2
    _match_count[(B,)](ctx, n_arr, key1, base, occ, agree, ML, nblk,
                       maxl, _SH, _RH, BLOCK=_BLOCK)


# ----- confidence capture -----------------------------------------------------
# capture split-V reduction: NSPLIT chunks of CHUNK, stepped BLOCK at a time (CHUNK/BLOCK static iters)
_NSPLIT, _CBLOCK, _CHUNK = 64, 512, 4096


@triton.jit
def _cap_s1(logits, V, pm, ps, NSPLIT, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    sp = tl.program_id(1)
    row = logits + b * V
    base = sp * CHUNK
    m = -float("inf")
    s = 0.0
    for k in range(CHUNK // BLOCK):
        idx = base + k * BLOCK + tl.arange(0, BLOCK)
        mask = idx < V
        l = tl.load(row + idx, mask=mask, other=-float("inf")).to(tl.float32)
        nm = tl.maximum(m, tl.max(l, 0))
        e = tl.where(mask, tl.exp(l - nm), 0.0)
        sc = tl.where(m == float("-inf"), 0.0, tl.exp(m - nm))
        s = s * sc + tl.sum(e, 0)
        m = nm
    o = b * NSPLIT + sp
    tl.store(pm + o, m)
    tl.store(ps + o, s)


@triton.jit
def _cap_s2(logits, V, pm, ps, conf, NSPLIT, BN: tl.constexpr):
    b = tl.program_id(0)
    r = tl.arange(0, BN)
    mask = r < NSPLIT
    o = b * NSPLIT + r
    m = tl.load(pm + o, mask=mask, other=-float("inf"))
    s = tl.load(ps + o, mask=mask, other=0.0)
    M = tl.max(m, 0)
    sc = tl.exp(m - M)
    S = tl.sum(s * sc, 0)
    tl.store(conf + b, 1.0 / S)                             # top-1 softmax prob = exp(max-M)/S = 1/S


def capture_gpu(logits, out_conf, scratch):
    """logits [B,V] (draft-head). Writes the per-row top-1 confidence into out_conf [B] in-place.
    scratch holds the [B*NSPLIT] partials (reused across slots)."""
    B, V = logits.shape
    pm, ps = scratch
    _cap_s1[(B, _NSPLIT)](logits, V, pm, ps, _NSPLIT, _CHUNK, _CBLOCK)
    _cap_s2[(B,)](logits, V, pm, ps, out_conf, _NSPLIT, 64)


@triton.jit
def _cap_s2_local(pm, ps, om, os_, NSPLIT, BN: tl.constexpr):
    """Same combine as _cap_s2, but emits the shard's (max, sum-exp) instead of a finished
    confidence -- the cross-rank logsumexp finishes it."""
    b = tl.program_id(0)
    r = tl.arange(0, BN)
    mask = r < NSPLIT
    o = b * NSPLIT + r
    m = tl.load(pm + o, mask=mask, other=-float("inf"))
    s = tl.load(ps + o, mask=mask, other=0.0)
    M = tl.max(m, 0)
    sc = tl.exp(m - M)
    tl.store(om + b, M)
    tl.store(os_ + b, tl.sum(s * sc, 0))


def capture_local(logits, out_max, out_sum, scratch):
    """logits [B,Vlocal] (this rank's vocabulary shard only). Writes the shard's running max into
    out_max [B] and its sum of exp(l - max) into out_sum [B]. Combining these across ranks with a
    logsumexp gives exactly the number capture_gpu returns for the gathered row, while moving three
    floats per row instead of the whole 248320-wide logit row."""
    B, V = logits.shape
    pm, ps = scratch
    _cap_s1[(B, _NSPLIT)](logits, V, pm, ps, _NSPLIT, _CHUNK, _CBLOCK)
    _cap_s2_local[(B,)](pm, ps, out_max, out_sum, _NSPLIT, 64)


def make_scratch(B, device):
    z = lambda: torch.empty(B * _NSPLIT, device=device)
    return z(), z()
