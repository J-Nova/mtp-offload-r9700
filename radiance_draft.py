#!/usr/bin/env python3
"""RADIANCE dynamic MTP draft-depth controller.

A per-request, per-slot controller for MTP self-speculation. At each draft slot it decides one of:
keep drafting (run another MTP forward), take a verbatim n-gram continuation, or stop and verify -- so
the server drafts deep on repetitive/high-acceptance content (code, JSON) and stays shallow on
prose/chat, and never runs deep serial forwards at concurrency.

Lossless: it only changes how many tokens are drafted (and MTP vs a verbatim copy of earlier text).
Every drafted token verifies identically through the unchanged rejection sampler, so outputs cannot
change -- this is a throughput optimization only.

Rule (evaluated per slot): take the n-gram tail when its next token equals the drafter's own top guess
(the original "free win"), OR when the match is long enough that a verbatim continuation is the better
bet than MTP's single top-1 (RADIANCE_DRAFT_NGRAM_STRONG, default 8 -- a long exact suffix repeat on
code/JSON/echo); otherwise draft while the running product of the drafter's top-1 confidences stays
>= RADIANCE_DRAFT_TAU, else stop and verify. A batch-size schedule caps how many serial MTP forwards
run at a given concurrency; the free verbatim n-gram tail can still fill the remaining draft width.

The matcher returns the top TWO candidates per row; if the longest disagrees and is not long enough to
be trusted, the second is tried before falling back to the confidence gate.

All hot-path work is on-device (Triton confidence capture + n-gram matcher, radiance_draft_gpu.py).
One tiny per-slot device->host copy (top-1 confidence + the drafted token id) lets the host
short-circuit the draft-forward loop (patch_mtp_loopbreak.py), plus one per-step copy of the match
metadata. If the GPU kernels are unavailable the controller stays out of the way and the server runs
stock MTP.

Env (the knobs):
  RADIANCE_DYNAMIC_DRAFT    1=on (default), 0=off -> byte-identical stock MTP
  RADIANCE_DRAFT_SCHEDULE   "bs:max_depth,..." batch-size MTP-forward ceiling (default 1:8,2:7,4:6,8:5,16:4)
  RADIANCE_DRAFT_TAU        confidence-product stop threshold (default 0.20)
  RADIANCE_DRAFT_NGRAM_STRONG  min match length to take a tail without MTP agreement (default 8; 0=off)
  RADIANCE_DRAFT_NGRAM_RECENT  take a short match (>=3) whose occurrence is within N tokens (default 0=off)
  RADIANCE_DRAFT_NGRAM_MAXL    suffix comparison horizon (default 32)
  RADIANCE_DRAFT_NGRAM_WINDOW  self-match window: "auto" (default), 0=full, or tokens
  RADIANCE_DRAFT_NGRAM_AUTO_FULL / _WINDOW_TOKENS  auto-window thresholds (default 32768 / 16384)
  RADIANCE_DRAFT_CROSS_REQ  search all active rows: "auto" (default, small B + short ctx), 1, 0
  RADIANCE_DRAFT_CROSS_MAX / _CROSS_CTX_MAX  auto cross-request bounds (default 8 / 16384)
  RADIANCE_DRAFT_STATS      periodic controller counters on stderr (default 1)
  RADIANCE_DRAFT_STATS_EVERY  steps between stats lines (default 200)
  RADIANCE_DRAFT_V2_CONF    V2 speculator confidence capture (default 0=off; see _install_v2_hooks)
"""
import os
import sys
import json
import time as _time
import numpy as np

# ----- config -----------------------------------------------------------------
DYNAMIC = os.environ.get("RADIANCE_DYNAMIC_DRAFT", "1") == "1"
# Confidence-product stop threshold. The shipped serve used 0.20; the image/docs said 0.35 and the
# code fallback 0.28. Unified on the production value here (see NGRAM-MTP-PLAN.md, P9). The 2-bit
# head with FAST_DRAFT measured 0.28 as +5.3% over 0.35, and 0.20 is the deeper value production runs.
TAU = float(os.environ.get("RADIANCE_DRAFT_TAU") or "0.20")
# Match length at which a verbatim n-gram tail is taken WITHOUT requiring MTP agreement. Long exact
# suffix repeats (code, JSON, file edits, echo) are where prompt-lookup beats a single top-1; short
# matches keep the strict agreement rule so prose is unaffected. 0 restores agree-only behaviour.
STRONG = int(os.environ.get("RADIANCE_DRAFT_NGRAM_STRONG") or "8")
_MINLEN = 3  # must match radiance_draft_gpu._MIN (the matcher's minimum reported match length)
# Self-match search window in tokens. Most deployments get "auto": full context up to
# RADIANCE_DRAFT_NGRAM_AUTO_FULL (default 32768), then the last RADIANCE_DRAFT_NGRAM_WINDOW_TOKENS
# (default 16384) to bound the per-step scan at long context, with a full-scan fallback when a
# windowed row finds nothing. A positive integer sets a fixed window; 0 forces full context always.
_WINDOW_ENV = (os.environ.get("RADIANCE_DRAFT_NGRAM_WINDOW") or "auto").strip().lower()
WINDOW = int(_WINDOW_ENV) if _WINDOW_ENV.isdigit() else 0
AUTO_WINDOW_TOKENS = int(os.environ.get("RADIANCE_DRAFT_NGRAM_WINDOW_TOKENS") or "16384")
AUTO_FULL = int(os.environ.get("RADIANCE_DRAFT_NGRAM_AUTO_FULL") or "32768")
# Match horizon: how far back the suffix comparison runs. Match length is the confidence signal, so
# a longer horizon sharpens the gate; the continuation is capped at nspec regardless.
MAXL = int(os.environ.get("RADIANCE_DRAFT_NGRAM_MAXL") or "32")
# Recency gate: accept a SHORT match (>= the matcher's MIN of 3) without MTP agreement when its
# occurrence is within this many tokens of the current end -- a recent repeat is a much stronger
# signal than a flat length threshold alone. DEFAULT 0 (off): measured dormant on the blend-OCP probe
# (fired 0-2 times per ~400 steps), so the length gate already covers it; raise it for corpora that
# actually produce short near-term repeats.
RECENT = int(os.environ.get("RADIANCE_DRAFT_NGRAM_RECENT") or "0")
# Cross-request matching: search every active row, not only the request's own. "auto" (default)
# enables it only for small batches and short contexts (O(B^2)); "1" forces it; "0" disables.
_CROSS_ENV = (os.environ.get("RADIANCE_DRAFT_CROSS_REQ") or "auto").strip().lower()
CROSS_MAX = int(os.environ.get("RADIANCE_DRAFT_CROSS_MAX") or "8")
CROSS_CTX_MAX = int(os.environ.get("RADIANCE_DRAFT_CROSS_CTX_MAX") or "16384")
# Controller counters (opt-out). Cheap: a few ints per step.
STATS = os.environ.get("RADIANCE_DRAFT_STATS", "1") == "1"
STATS_EVERY = int(os.environ.get("RADIANCE_DRAFT_STATS_EVERY") or "200")
# V2 speculator confidence capture (opt-in, default OFF). The served draft decode loop is recorded
# into FULL CUDA graphs (cudagraph_utils / autoregressive.speculator), so the Python
# `_greedy_sample_draft`/`_apply_head` bodies do NOT run per served decode step -- only eager
# (PIECEWISE) draft-prefill steps reach them. Turning this on therefore captures at most the
# position-0 confidence, not a per-step one, and nothing consumes it yet (stage 2). Keep it off on
# the served path to avoid a misleading signal and per-eager-step softmax work.
V2_CONF = os.environ.get("RADIANCE_DRAFT_V2_CONF", "0") == "1"
# batch-size MTP-forward ceiling "bs:max_depth,..." (carry-forward). Caps how deep the drafter forwards
# by running batch size, so deep serial drafts do not run at concurrency. The per-slot rule still stops
# earlier within it, and the free n-gram tail is unaffected. Empty string disables the cap.
SCHEDULE = (os.environ.get("RADIANCE_DRAFT_SCHEDULE") or "1:8,2:7,4:6,8:5,16:4").strip()
# Per-phase host timers (diagnostic, default off). RADIANCE_DRAFT_PHASE_TIMERS=1 accumulates the
# wall time of each step of the per-slot draft path (_apply_head / padding / capture / argmax /
# packed D2H / slot_decide) and logs cumulative averages every 200 slot calls. This is what
# attributes the ~36 ms MTP-specific sample_tok, which B1 and B3-defer both showed is NOT the
# controller sync/policy. Times are wall, not CPU: a phase that blocks on the GPU shows up here.
PHASE_TIMERS = os.environ.get("RADIANCE_DRAFT_PHASE_TIMERS", "0") == "1"
_PH = {}
_PH_CALLS = [0]
_PROP_CALLS = [0]
_WRAP_CALLS = [0]
# Matcher stride (host-cost lever): run the on-device n-gram matcher AND its per-step pack .cpu()
# sync only every K-th step; the other steps draft on the confidence gate alone (a zeroed candidate
# pack, so slot_decide degrades to the confidence-only rule). The context mirror append still runs
# every step, so no matcher step ever sees stale text. The n-gram tail is a bonus, not a correctness
# input -- skipping it only changes how many tokens are drafted, never the verified output.
# 1 = every step (default, stock). The step trace prices the matcher+sync at ~13 ms of host time
# per step at 205k, and it fires on a minority of rows, so a stride is nearly free on novel text.
MATCH_EVERY = max(1, int(os.environ.get("RADIANCE_DRAFT_MATCH_EVERY") or "1"))
_match_ctr = [0]

# Runtime state, not knobs.
_LOCAL_OK = [True]      # latches off if the shard-local draft path ever raises
_GPU = [None]           # the radiance_draft_gpu kernels module (set at install if it imports)
# Controller counters (P8). `steps` drives the periodic log; the rest are cumulative.
_STATS = {"steps": 0, "rows": 0, "match": 0, "strong": 0, "recent": 0, "tail1": 0, "tail2": 0,
          "mtp_tokens": 0, "ngram_tokens": 0}


def _log(msg):
    sys.stderr.write(f"[radiance.draft] {msg}\n")
    sys.stderr.flush()


def _phase(name, t0):
    """Accumulate one phase's wall time (no-op unless RADIANCE_DRAFT_PHASE_TIMERS=1)."""
    if PHASE_TIMERS:
        _PH[name] = _PH.get(name, 0.0) + (_time.perf_counter() - t0)


def _phase_log():
    if not PHASE_TIMERS:
        return
    _PH_CALLS[0] += 1
    if _PH_CALLS[0] % 200 == 0:
        _log("phase/slot-call n=%d  %s" % (_PH_CALLS[0], "  ".join(
            "%s=%.3f" % (k, 1000.0 * v / _PH_CALLS[0]) for k, v in sorted(_PH.items()))))


def _stats_tick():
    if not STATS:
        return
    _STATS["steps"] += 1
    steps = _STATS["steps"]
    if STATS_EVERY > 0 and steps % STATS_EVERY == 0:
        s = _STATS
        tails = s["tail1"] + s["tail2"]
        drafted = s["mtp_tokens"] + s["ngram_tokens"]
        share = (100.0 * s["ngram_tokens"] / drafted) if drafted else 0.0
        _log(f"stats steps={steps} rows={s['rows']} matched={s['match']} strong={s['strong']} "
             f"recent={s['recent']} tail={tails} (c1={s['tail1']} c2={s['tail2']}) "
             f"tokens: mtp={s['mtp_tokens']} ngram={s['ngram_tokens']} "
             f"({share:.1f}% ngram, {drafted / steps:.2f} tok/draft)")


def _parse_schedule(spec):
    pairs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            bs, d = tok.split(":")
            pairs.append((int(bs), int(d)))
    return sorted(pairs)


_SCHED = _parse_schedule(SCHEDULE)

# Live knob file: when RADIANCE_DRAFT_KNOB_FILE points at a JSON object, the policy globals are
# re-read from it once per engine step (cheap mtime check). This lets ONE server launch A/B the
# policy knobs -- {"strong":8,"recent":0,"tau":0.2,"maxl":32,"window":"auto","cross":"auto"} -- which
# the quick harness uses to fit a full sweep in a single boot. SPEC/num_speculative_tokens is a vLLM
# launch arg and still needs a restart.
KNOB_FILE = os.environ.get("RADIANCE_DRAFT_KNOB_FILE", "")
_knob_mtime = [None]


def _reload_knobs():
    if not KNOB_FILE:
        return
    global STRONG, RECENT, TAU, MAXL, WINDOW, _WINDOW_ENV, _CROSS_ENV
    try:
        m = os.stat(KNOB_FILE).st_mtime_ns
        if m == _knob_mtime[0]:
            return
        with open(KNOB_FILE) as f:
            k = json.load(f)
        _knob_mtime[0] = m
    except Exception:
        return
    try:
        if "strong" in k:
            STRONG = int(k["strong"])
        if "recent" in k:
            RECENT = int(k["recent"])
        if "tau" in k:
            TAU = float(k["tau"])
        if "maxl" in k:
            MAXL = int(k["maxl"])
        if "window" in k:
            w = str(k["window"]).lower()
            _WINDOW_ENV = w
            WINDOW = int(w) if w.isdigit() else 0
        if "cross" in k:
            _CROSS_ENV = str(k["cross"]).lower()
        _log(f"knobs reloaded: strong={STRONG} recent={RECENT} tau={TAU} maxl={MAXL} "
             f"window={_WINDOW_ENV} cross={_CROSS_ENV}")
    except Exception as e:
        _log(f"knob file parse failed: {e!r}")


def _batch_ceil(num_reqs):
    """Max MTP forward depth at the current batch size (carry-forward on the schedule); large when unset."""
    if not _SCHED:
        return 1 << 30
    d = _SCHED[0][1]
    for bs, v in _SCHED:
        if num_reqs >= bs:
            d = v
        else:
            break
    return max(1, d)


# ----- local-vocabulary draft sampling (no full-vocab all-gather) -------------
# The gate needs exactly two numbers per row: the drafted token id and the top-1 softmax
# probability. Getting them the obvious way -- compute_logits() -> argmax -- makes vLLM all-gather
# the whole 248320-wide logit row across the TP group on every draft slot, which the decode profile
# prices at ~374 us per slot. Both numbers are recoverable from per-rank partial reductions instead:
# each rank reduces its own vocabulary shard to (max, sum-exp, argmax), the ranks exchange three
# floats per row, and a cross-rank logsumexp finishes the softmax. That is the same arithmetic on
# the same values, up to floating-point associativity -- not an approximation.
#
# vLLM has its own version of half of this (`use_local_argmax_reduction`), but it only produces the
# token id; the confidence gate would still need the gathered logits, which is why this controller
# switches itself off whenever that flag is set.
def _local_draft(proposer, hidden_states):
    """Returns (draft_token_ids [B] int64, conf [B] float32) without gathering the vocabulary.
    
    When RADIANCE_DRAFT_HEAD_TOP1=1, the head returns (ids, conf) directly (fused B2 path),
    skipping the capture and argmax steps. Otherwise, it returns the full logit row Y and
    we capture confidence and argmax separately."""
    import torch
    from vllm.distributed import (get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)
    from radiance_drafthead import TOP1
    gpu = _GPU[0]
    model = proposer.model
    lp = model.logits_processor
    lm_head = model.lm_head
    _t = _time.perf_counter() if PHASE_TIMERS else 0.0
    local = lp._apply_head(lm_head, hidden_states, None)
    if PHASE_TIMERS:
        _phase("head", _t)
        _t = _time.perf_counter()

    # Fused top-1 path: _apply_head returns (ids, conf) directly.
    # Only valid for TP=1: the fused path returns local-shard-only ids/conf without
    # cross-rank all-gather, so it would produce incorrect results under TP>1.
    tp = get_tensor_model_parallel_world_size()
    if TOP1 and isinstance(local, tuple) and tp == 1:
        ids, conf = local
        if PHASE_TIMERS:
            _phase("fused_top1", _t)
        return ids.to(torch.int64), conf

    # Standard path: _apply_head returns the full logit row Y
    if local.dim() > 2:
        local = local.reshape(-1, local.shape[-1])
    npad = getattr(getattr(lm_head, "shard_indices", None), "num_org_vocab_padding", 0) or 0
    if npad > 0:
        local[..., -npad:] = float("-inf")      # padding entries must not win the argmax
    if PHASE_TIMERS:
        _phase("pad", _t)
        _t = _time.perf_counter()

    B = local.shape[0]
    start = getattr(getattr(lm_head, "shard_indices", None), "org_vocab_start_index", 0) or 0
    # FUSED+EXACTSET: the head stashed a confidence from the COARSE kept-vocab partials. The
    # returned row is -inf outside the reranked set, so capture_local would read ~1.0 always and
    # defeat the tau gate. Consume the stashed value and skip capture (see radiance_drafthead).
    preconf = None
    if getattr(lp, "_radiance_conf_precomputed", False):
        preconf = getattr(lp, "_radiance_last_conf", None)
        lp._radiance_last_conf = None
    if not getattr(lp, "_radiance_ld_logged", False):
        lp._radiance_ld_logged = True
        _log(f"_local_draft: preconf_flag={getattr(lp, '_radiance_conf_precomputed', False)} "
             f"last_conf={'set' if preconf is not None else 'none'} tp={tp}")
    lidx = local.argmax(dim=-1)
    if PHASE_TIMERS:
        _phase("argmax", _t)
        _t = _time.perf_counter()
    if preconf is not None and tp == 1:
        if not getattr(lp, "_radiance_preconf_logged", False):
            lp._radiance_preconf_logged = True
            _log("precomputed tau confidence active (FUSED+EXACTSET)")
        return (lidx + start).to(torch.int64), preconf[:B].to(torch.float32)
    sc = getattr(proposer, "_radiance_lscratch", None)
    if sc is None or sc[0].shape[0] != B * gpu._NSPLIT:
        sc = proposer._radiance_lscratch = gpu.make_scratch(B, local.device)
    lmax = torch.empty(B, device=local.device)
    lsum = torch.empty(B, device=local.device)
    gpu.capture_local(local, lmax, lsum, sc)
    if PHASE_TIMERS:
        _phase("capture", _t)
        _t = _time.perf_counter()
    tp = get_tensor_model_parallel_world_size()
    if tp == 1:
        return (lidx + start).to(torch.int64), 1.0 / lsum
    stats = torch.stack([lmax, lsum, (lidx + start).to(torch.float32)], dim=-1)   # [B,3]
    g = tensor_model_parallel_all_gather(stats, dim=-1).view(B, tp, 3)
    gm = g[:, :, 0]
    M, which = gm.max(dim=1)
    S = (g[:, :, 1] * torch.exp(gm - M[:, None])).sum(dim=1)
    ids = g[:, :, 2].gather(1, which[:, None]).squeeze(1).to(torch.int64)
    return ids, 1.0 / S


# ----- per-slot policy (pure, unit-tested in ngram_draft_selftest.py) ---------
def slot_decide(j, cont1, cont2, clen1, mlen1, clen2, mlen2, mtpn, cum,
                rec1=None, rec2=None, strong=None, tau=None, recent=None):
    """Per-slot decision, the single home of the policy (unit-tested in ngram_draft_selftest.py).

    Inputs are per-row numpy arrays (or 0-d scalars) for the batch:
      cont1/cont2 [B,nspec] int  candidate continuations
      clen1/clen2 [B] int        candidate continuation lengths (<= nspec)
      mlen1/mlen2 [B] int        candidate match lengths (0 = no match)
      mtpn [B] int64             this slot's drafted token from MTP
      cum [B] float              running product of MTP top-1 confidences
      rec1/rec2 [B] int          match distance back from this row's end (self) or a big sentinel
    Returns (action [B] int64 in {0,1,2}, cand [B] int64 in {0,1,2}) where action 1 = take the n-gram
    tail (cand says which candidate), 0 = keep drafting, 2 = stop and verify.

    A candidate is used when its next token equals MTP's (the original free win), OR its match is
    long enough to trust verbatim (strong), OR it is a short match that recurred very recently
    (recency). Candidate 2 is only considered when candidate 1 is not used. With no candidate,
    drafting continues while `cum >= tau` and otherwise stops."""
    strong = STRONG if strong is None else strong
    tau = TAU if tau is None else tau
    recent = RECENT if recent is None else recent
    B = mtpn.shape[0]
    ns = cont1.shape[1] if cont1.ndim == 2 else 0
    agree1 = np.zeros(B, bool)
    agree2 = np.zeros(B, bool)
    if ns and j < ns:
        agree1 = (j < clen1) & (cont1[:, j] == mtpn)
        agree2 = (j < clen2) & (cont2[:, j] == mtpn)
    if strong > 0:
        strong1 = (mlen1 >= strong) & (j < clen1)
        strong2 = (mlen2 >= strong) & (j < clen2)
    else:
        strong1 = np.zeros(B, bool)
        strong2 = np.zeros(B, bool)
    if recent > 0 and rec1 is not None:
        recent1 = (mlen1 >= _MINLEN) & (j < clen1) & (rec1 <= recent)
        recent2 = (mlen2 >= _MINLEN) & (j < clen2) & (rec2 <= recent)
    else:
        recent1 = np.zeros(B, bool)
        recent2 = np.zeros(B, bool)
    use1 = agree1 | strong1 | recent1
    use2 = (~use1) & (agree2 | strong2 | recent2)
    cand = np.where(use1, 1, np.where(use2, 2, 0)).astype(np.int64)
    action = np.where(cand != 0, 1, np.where(cum >= tau, 0, 2)).astype(np.int64)
    return action, cand


# ----- drafter hooks: per-slot confidence capture + A/B/C decision ------------
def _install_drafter_hooks():
    import torch
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    if getattr(SpecDecodeBaseProposer, "_radiance_wrapped", False):
        return
    orig_greedy = SpecDecodeBaseProposer._greedy_sample
    orig_propose = SpecDecodeBaseProposer.propose

    def greedy_sample(self, hidden_states):
        if not getattr(self, "_radiance_gs_logged", False):
            self._radiance_gs_logged = True
            _log(f"greedy_sample ENTERED active={getattr(self, '_radiance_active', False)} "
                 f"cls={type(self).__name__}")
        if not getattr(self, "_radiance_active", False):
            return orig_greedy(self, hidden_states)
        _t0 = _time.perf_counter() if PHASE_TIMERS else 0.0
        gpu = _GPU[0]
        logits = None
        if _LOCAL_OK[0]:
            try:
                draft_token_ids, conf_dev = _local_draft(self, hidden_states)
            except Exception as e:
                _log(f"local-vocab draft failed, using the gathered path: {e!r}")
                _LOCAL_OK[0] = False
        if not _LOCAL_OK[0]:
            logits = self.model.compute_logits(hidden_states)
            draft_token_ids = logits.argmax(dim=-1)
        # Per-slot decision (confidence-gated depth + length-gated n-gram tail). One tiny per-slot
        # D2H (conf + drafted token); the match metadata was already copied to the host once per step.
        j = self._radiance_slot
        self._radiance_slot = j + 1
        B = draft_token_ids.shape[0]
        N = self._radiance_nspec
        pk = getattr(self, "_radiance_pack_cpu", None)
        if pk is None or pk.shape[0] != B or pk.shape[1] != 2 * N + gpu._META:
            if PHASE_TIMERS:
                _phase("slot_total", _t0)
            return draft_token_ids                       # matcher didn't run -> full native draft (safe)
        sc = getattr(self, "_radiance_scratch", None)
        if sc is None or sc[0].shape[0] != B * gpu._NSPLIT:
            sc = self._radiance_scratch = gpu.make_scratch(B, draft_token_ids.device)
        # pack conf + drafted token id into one device tensor so the whole slot is a SINGLE D2H
        # (token ids < 2^24 are exact in fp32).
        packed = torch.empty(2, B, device=draft_token_ids.device)
        if logits is None:
            packed[0] = conf_dev
        else:
            gpu.capture_gpu(logits, packed[0], sc)
        packed[1] = draft_token_ids.to(torch.float32)
        _t = _time.perf_counter() if PHASE_TIMERS else 0.0
        arr = packed.cpu().numpy()
        if PHASE_TIMERS:
            _phase("d2h", _t)
            _t = _time.perf_counter()
        cfn = arr[0]
        mtpn = arr[1].astype(np.int64)
        cont1 = pk[:, :N]
        cont2 = pk[:, N:2 * N]
        meta = pk[:, 2 * N:]
        clen1 = meta[:, 0].astype(np.int64)
        mlen1 = meta[:, 1].astype(np.int64)
        clen2 = meta[:, 2].astype(np.int64)
        mlen2 = meta[:, 3].astype(np.int64)
        rec1 = meta[:, 5].astype(np.int64)
        rec2 = meta[:, 6].astype(np.int64)
        st = self._radiance_gate
        if j == 0:
            st.update(cum=np.ones(B, np.float32), stopped=np.zeros(B, bool),
                      sslot=np.full(B, N, np.int64), saction=np.full(B, 2, np.int64),
                      scand=np.zeros(B, np.int64))
        action, cand = slot_decide(j, cont1, cont2, clen1, mlen1, clen2, mlen2, mtpn, st["cum"],
                                   rec1=rec1, rec2=rec2)
        if PHASE_TIMERS:
            _phase("decide", _t)
            _t = _time.perf_counter()
        if RECENT > 0:
            agree = ((j < clen1) & (cont1[:, j] == mtpn)) | ((j < clen2) & (cont2[:, j] == mtpn))
            ronly = ((cand != 0) & ~agree & (mlen1 < STRONG) & (mlen1 >= _MINLEN)
                     & (rec1 <= RECENT))
            _STATS["recent"] += int(ronly.sum())
        # action at slot j is the decision FOR slot j: 1 = take the n-gram tail (keep MTP[:j], then
        # the verbatim continuation from j), 0 = keep drafting, 2 = stop and verify. Slot j's own
        # forward ran only to produce this decision.
        newly = (~st["stopped"]) & (action != 0)
        st["sslot"] = np.where(newly, j, st["sslot"])
        st["saction"] = np.where(newly, action, st["saction"])
        st["scand"] = np.where(newly, cand, st["scand"])
        st["stopped"] = st["stopped"] | (action != 0)
        st["cum"] = st["cum"] * cfn
        fc = self._radiance_fwd_cap
        if fc and (j + 1) >= fc:
            # Forward cap hit: stop the remaining MTP forwards. The capped rows just verify their MTP
            # prefix; the free n-gram tail is delivered earlier by the length gate, so the old
            # allagree-at-cap branch (which was unreachable) is gone.
            cap_new = ~st["stopped"]
            st["sslot"] = np.where(cap_new, j + 1, st["sslot"])
            st["stopped"] = st["stopped"] | cap_new
        if st["stopped"].all():
            self._radiance_stop = True
        if PHASE_TIMERS:
            _phase("gate", _t)
            _phase("slot_total", _t0)
        _phase_log()
        return draft_token_ids

    def propose(self, num_speculative_tokens, *args, **kwargs):
        _reload_knobs()
        bc = getattr(self, "_radiance_batch_ceil", 0)
        # gate only on the standard full-vocab argmax path the capture kernel assumes
        skip = self.use_local_argmax_reduction or self.use_heterogeneous_vocab
        if not getattr(self, "_radiance_prop_logged", False):
            self._radiance_prop_logged = True
            _log(f"propose ENTERED cls={type(self).__name__} skip={skip} "
                 f"local_argmax={self.use_local_argmax_reduction} "
                 f"hetero={self.use_heterogeneous_vocab} nspec={num_speculative_tokens}")
        if skip:
            self._radiance_active = False
            if 0 < bc < num_speculative_tokens:          # still honor the concurrency cap by clamping
                num_speculative_tokens = bc
            return orig_propose(self, num_speculative_tokens, *args, **kwargs)
        # Keep num_speculative_tokens at full width so vLLM allocates the full draft; cap the MTP FORWARD
        # count via the loop-break instead (fwd_cap = batch_ceil), so the free n-gram tail can still fill
        # the remaining draft positions at concurrency instead of being truncated.
        self._radiance_fwd_cap = bc if (0 < bc < num_speculative_tokens) else 0
        self._radiance_active = True
        self._radiance_slot = 0
        self._radiance_stop = False
        self._radiance_nspec = num_speculative_tokens
        self._radiance_gate = {}                          # per-slot gating state (filled at j==0)
        _tp = _time.perf_counter() if PHASE_TIMERS else 0.0
        r = orig_propose(self, num_speculative_tokens, *args, **kwargs)
        if PHASE_TIMERS:
            _PH["propose_total"] = _PH.get("propose_total", 0.0) + (_time.perf_counter() - _tp)
            _PROP_CALLS[0] += 1
            if _PROP_CALLS[0] % 100 == 0:
                _log("phase/step n=%d  propose_total=%.2fms  slot_total=%.2fms  slack(loop+fwd)=%.2fms"
                     % (_PROP_CALLS[0], 1000.0 * _PH["propose_total"] / _PROP_CALLS[0],
                        1000.0 * _PH.get("slot_total", 0.0) / _PROP_CALLS[0],
                        1000.0 * (_PH["propose_total"] - _PH.get("slot_total", 0.0)) / _PROP_CALLS[0]))
        return r

    SpecDecodeBaseProposer._greedy_sample = greedy_sample
    SpecDecodeBaseProposer.propose = propose
    SpecDecodeBaseProposer._radiance_wrapped = True


# ----- V2 speculator hooks (vLLM 0.29 default runner) ------------------------
# 0.29 runs the NEW model runner (vllm/v1/worker/gpu/model_runner.py) whose MTP drafter is
# MTPSpeculator -> AutoRegressiveSpeculator -> DraftModelSpeculator, NOT the legacy
# SpecDecodeBaseProposer the hooks above target. These hook the V2 speculator for liveness; the
# per-step confidence capture is opt-in (RADIANCE_DRAFT_V2_CONF=1) because the served decode loop is
# a replayed FULL CUDA graph, so the Python sampling body only runs on eager (PIECEWISE) draft steps
# (see the V2_CONF comment). Gate/tail assembly lands next.
def _install_v2_hooks():
    try:
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
        from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
            AutoRegressiveSpeculator,
        )
    except Exception as e:
        _log(f"V2 speculator hooks unavailable ({e!r}); legacy hooks only")
        return
    if getattr(DraftModelSpeculator, "_radiance_v2_wrapped", False):
        return

    orig_greedy = DraftModelSpeculator._greedy_sample_draft

    def greedy_draft(self, hidden_states):
        if not getattr(self, "_radiance_active", False):
            return orig_greedy(self, hidden_states)
        logits = self.model.compute_logits(hidden_states)
        ids = logits.argmax(dim=-1)
        lp = getattr(self.model, "logits_processor", None)
        conf = None
        if lp is not None and getattr(lp, "_radiance_conf_precomputed", False):
            conf = getattr(lp, "_radiance_last_conf", None)
            lp._radiance_last_conf = None
        if conf is None:
            f = logits.float()
            mx = f.max(dim=-1).values
            conf = 1.0 / (f - mx[..., None]).exp().sum(dim=-1)
        if conf.dim() > 1:
            conf = conf.reshape(-1)
        self._radiance_conf_hist = conf
        if not getattr(self, "_radiance_v2_logged", False):
            self._radiance_v2_logged = True
            _log(f"V2 greedy_sample_draft ENTERED ids={tuple(ids.shape)} "
                 f"conf min={float(conf.min()):.3g} max={float(conf.max()):.3g} "
                 f"preconf={getattr(lp, '_radiance_conf_precomputed', False)}")
        return ids

    # Capture is opt-in: off by default it is pure overhead with no consumer, and on the served
    # path it cannot see the replayed decode steps anyway.
    if V2_CONF:
        DraftModelSpeculator._greedy_sample_draft = greedy_draft

    orig_propose = AutoRegressiveSpeculator.propose

    def propose_v2(self, *a, **k):
        self._radiance_active = True
        if V2_CONF:
            self._radiance_conf_hist = None
        r = orig_propose(self, *a, **k)
        if not getattr(self, "_radiance_v2_prop_logged", False):
            self._radiance_v2_prop_logged = True
            h = getattr(self, "_radiance_conf_hist", None)
            _log(f"V2 propose ENTERED cls={type(self).__name__} "
                 f"conf_hist={'set' if h is not None else 'none'} v2_conf={int(V2_CONF)} "
                 f"fused={getattr(self, 'use_fused_multi_step_decode', None)} "
                 f"steps={getattr(self, 'num_speculative_steps', None)} "
                 f"adv={getattr(self, 'advance_draft_positions', None)}")
        return r

    AutoRegressiveSpeculator.propose = propose_v2
    DraftModelSpeculator._radiance_v2_wrapped = True
    _log(f"V2 speculator hooks installed (liveness; confidence capture {'ON' if V2_CONF else 'off'})")
    sys.stderr.flush()


# ----- runner wrap: run the matcher, then assemble the gated draft ------------
def _install_runner_wrap():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_radiance_draft_wrapped", False):
        return
    orig = GPUModelRunner.propose_draft_token_ids

    def wrapped(self, *args, **kwargs):
        _tw = _time.perf_counter() if PHASE_TIMERS else 0.0
        if not getattr(self, "_radiance_wrap_logged", False):
            self._radiance_wrap_logged = True
            try:
                _dn = type(self.drafter).__name__
            except Exception:
                _dn = "?"
            _log(f"runner wrap ENTERED drafter={_dn} "
                 f"method={getattr(getattr(self, 'speculative_config', None), 'method', None)}")
        try:
            self.drafter._radiance_batch_ceil = _batch_ceil(len(self.input_batch.req_ids))
        except Exception:
            pass
        try:
            _prepare_match_gpu(self)          # ctx mirror + matcher -> drafter._radiance_pack_cpu
        except Exception as e:
            _log(f"gpu match prep failed (native draft used): {e!r}")
            try:
                self.drafter._radiance_pack_cpu = None
            except Exception:
                pass
        if PHASE_TIMERS:
            _phase("prepare", _tw)
            _tw = _time.perf_counter()
        draft = orig(self, *args, **kwargs)
        if PHASE_TIMERS:
            _phase("propose_all", _tw)
            _tw = _time.perf_counter()
        try:
            r = _postprocess_gpu(self, draft)
        except Exception as e:
            _log(f"postprocess error (native draft used): {e!r}")
            r = draft
        if PHASE_TIMERS:
            _phase("postprocess", _tw)
            _WRAP_CALLS[0] += 1
            if _WRAP_CALLS[0] % 100 == 0:
                _log("phase/wrap n=%d  sample=%.2f  bookkeep=%.2f  upd=%.2f  copy_draft=%.2f | "
                     "prepare=%.2f  propose_all=%.2f  postprocess=%.2f ms/step"
                     % (_WRAP_CALLS[0], 1000.0 * _PH.get("sample", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("_bookkeeping_sync", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("_update_states_after_model_execute", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("_copy_draft_token_ids_to_cpu", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("prepare", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("propose_all", 0.0) / _WRAP_CALLS[0],
                        1000.0 * _PH.get("postprocess", 0.0) / _WRAP_CALLS[0]))
        return r

    GPUModelRunner.propose_draft_token_ids = wrapped
    GPUModelRunner._radiance_draft_wrapped = True

    if PHASE_TIMERS and not getattr(GPUModelRunner, "_radiance_sample_wrapped", False):
        _orig_sample = GPUModelRunner._sample

        def _sample_wrapped(self, *args, **kwargs):
            _ts = _time.perf_counter()
            out = _orig_sample(self, *args, **kwargs)
            _phase("sample", _ts)
            return out

        GPUModelRunner._sample = _sample_wrapped
        GPUModelRunner._radiance_sample_wrapped = True

    if PHASE_TIMERS and not getattr(GPUModelRunner, "_radiance_book_wrapped", False):
        for _m in ("_bookkeeping_sync", "_update_states_after_model_execute",
                   "_copy_draft_token_ids_to_cpu"):
            _om = getattr(GPUModelRunner, _m, None)
            if _om is None:
                continue

            def _mk(om, key):
                def _tw(self, *args, **kwargs):
                    _ta = _time.perf_counter()
                    out = om(self, *args, **kwargs)
                    _phase(key, _ta)
                    return out
                return _tw

            setattr(GPUModelRunner, _m, _mk(_om, _m))
        GPUModelRunner._radiance_book_wrapped = True


def _prepare_match_gpu(runner):
    """Update the GPU context mirror with newly generated tokens, then run the on-device matcher.
    Stores `_radiance_pack_cpu` (numpy [B, 2*nspec+7]) on the drafter: one host copy carrying both
    continuations and the match metadata. The mirror append is O(new tokens) per request, the only
    host->device movement; matcher scratch is reused across steps."""
    import torch
    gpu = _GPU[0]
    ib = runner.input_batch
    req_ids = ib.req_ids
    d = runner.drafter
    d._radiance_pack_cpu = None
    B = len(req_ids)
    if B == 0:
        return
    dev = runner.device
    src = ib.token_ids_cpu_tensor            # cpu int tensor [max_reqs, max_len]
    nts = ib.num_tokens_no_spec              # numpy [max_reqs]
    ctxg = getattr(runner, "_radiance_ctx_gpu", None)
    if ctxg is None:
        ctxg = runner._radiance_ctx_gpu = torch.zeros(src.shape[0], src.shape[1], dtype=torch.int32, device=dev)
        runner._radiance_seen = {}
    seen = runner._radiance_seen
    n_list = []
    for i, rid in enumerate(req_ids):
        n = int(nts[i])
        n_list.append(n)
        # `seen` records (row, tokens_mirrored). The ROW matters: vLLM reuses and reshuffles
        # input-batch slots as requests come and go, so a request can land on a row that still
        # holds the previous occupant's tokens. Keyed by request id alone, the stale tail stayed
        # and the matcher searched another request's text.
        prev = seen.get(rid)
        s = prev[1] if (prev is not None and prev[0] == i) else 0
        if n > s:
            # NOT non_blocking: the source is the temporary from .to(int32), freed as soon as this
            # statement ends. An async copy from pageable memory can still be reading it, which put
            # nondeterministic garbage in the mirror.
            ctxg[i, s:n].copy_(src[i, s:n].to(torch.int32))
        seen[rid] = (i, n)
    live = set(req_ids)
    for rid in [r for r in seen if r not in live]:
        seen.pop(rid, None)
    nmax = max(n_list)
    if nmax < 3:
        return
    nspec = runner.num_spec_tokens
    if MATCH_EVERY > 1:
        _match_ctr[0] += 1
        if (_match_ctr[0] % MATCH_EVERY) != 0:
            # confidence-only step: no matcher launch, no pack D2H. Slot decisions read clen/mlen=0
            # so the n-gram gate can never fire this step; the confidence gate is unchanged.
            d._radiance_pack_cpu = np.zeros((B, 2 * nspec + gpu._META), dtype=np.int32)
            _STATS["rows"] += B
            return
    n_arr = torch.tensor(n_list, dtype=torch.int32, device=dev)
    n_np = np.asarray(n_list, dtype=np.int64)
    if _CROSS_ENV == "1":
        cross = True
    elif _CROSS_ENV == "auto":
        cross = 2 <= B <= CROSS_MAX and nmax <= CROSS_CTX_MAX
    else:
        cross = False
    if WINDOW > 0:
        window = WINDOW
    elif _WINDOW_ENV == "auto" and nmax > AUTO_FULL:
        window = AUTO_WINDOW_TOKENS
    else:
        window = 0
    # Scratch is sized for the FULL scan (two candidate slots per 512-position block) so an in-place
    # window fallback fits. Candidate slots are counted doubled everywhere (see radiance_draft_gpu).
    nc_full = (2 * B * gpu._nblk(nmax, 0)) if cross else (2 * gpu._nblk(nmax, 0))
    mb = getattr(runner, "_radiance_mbufs", None)
    if (mb is None or mb.get("nc") != nc_full or mb["pack"].shape[0] != B
            or mb["pack"].shape[1] != 2 * nspec + gpu._META):
        mb = gpu.make_match_buffers(B, nspec, nc_full, dev)
        runner._radiance_mbufs = mb
    base = mb["base"]
    if cross:
        pack = gpu.match_gpu(ctxg[:B], n_arr, nspec, nmax, base, nc_full, True, mb, MAXL)
    else:
        base_np = np.maximum(0, n_np - window) if window > 0 else np.zeros(B, dtype=np.int64)
        base.copy_(torch.from_numpy(base_np.astype(np.int32)))
        pack = gpu.match_gpu(ctxg[:B], n_arr, nspec, nmax, base,
                             2 * gpu._nblk(nmax, window), False, mb, MAXL)
    pk = pack.cpu().numpy()
    miss = pk[:, 2 * nspec + 4] != 0
    if not cross and window > 0 and miss.any():
        # Windowed rows that found nothing are re-scanned with the window removed. Only the miss
        # rows do work (base = the row's own length makes the window empty for the rest) and only
        # their rows are copied back -- not a full-batch rescan plus a second whole-pack sync.
        sel = np.nonzero(miss)[0]
        base.copy_(torch.from_numpy(np.where(miss, 0, n_np).astype(np.int32)))
        pack = gpu.match_gpu(ctxg[:B], n_arr, nspec, nmax, base, nc_full, False, mb, MAXL)
        pk[sel] = pack.index_select(0, torch.from_numpy(sel).to(pack.device)).cpu().numpy()
        base.zero_()
    d._radiance_pack_cpu = pk
    # P8 counters: rows, rows with any match, rows with a strong match.
    meta = pk[:, 2 * nspec:]
    _STATS["rows"] += B
    _STATS["match"] += int((meta[:, 1] >= gpu._MIN).sum())
    if STRONG > 0:
        _STATS["strong"] += int(((meta[:, 1] >= STRONG) | (meta[:, 3] >= STRONG)).sum())


def _postprocess_gpu(runner, draft):
    """Assemble the final draft from the per-slot decisions. The patched loop already ran only the
    forwards up to the batch's stop point; here each request is trimmed to its own stop slot and given
    the verbatim n-gram tail from the chosen candidate where it took one. Returns a ragged
    list[list[int]] (native draft format), so vLLM verifies exactly the tokens the controller kept."""
    import torch
    d = runner.drafter
    gpu = _GPU[0]
    st = getattr(d, "_radiance_gate", None)
    pk = getattr(d, "_radiance_pack_cpu", None)
    N = getattr(d, "_radiance_nspec", 0)
    if not st or "sslot" not in st or pk is None or N <= 0 or pk.shape[1] != 2 * N + gpu._META:
        return draft
    if torch.is_tensor(draft) and draft.dim() == 2:
        mtp = draft.int().cpu().numpy()
        B = mtp.shape[0]
        row = lambda i: mtp[i]
    elif isinstance(draft, list):
        B = len(draft)
        row = lambda i: draft[i]
    else:
        return draft
    if st["sslot"].shape[0] != B or pk.shape[0] != B:
        return draft
    cont1 = pk[:, :N]
    cont2 = pk[:, N:2 * N]
    meta = pk[:, 2 * N:]
    clen1 = meta[:, 0]
    clen2 = meta[:, 2]
    sslot = st["sslot"]
    saction = st["saction"]
    scand = st["scand"]
    out = []
    for i in range(B):
        r = row(i)
        k = int(min(sslot[i], len(r)))
        di = [int(x) for x in r[:k]]
        _STATS["mtp_tokens"] += k
        if saction[i] == 1:
            if int(scand[i]) == 2:
                c = cont2[i]
                cl = int(clen2[i])
                _STATS["tail2"] += 1
            else:
                c = cont1[i]
                cl = int(clen1[i])
                _STATS["tail1"] += 1
            kk = min(k, cl)
            if kk < cl:                               # append the verbatim n-gram continuation
                di += [int(x) for x in c[kk:cl]]
                _STATS["ngram_tokens"] += cl - kk
        if not di:
            di = [int(r[0])] if len(r) else []
        out.append(di[:N])
    _stats_tick()
    return out


# ----- entry ------------------------------------------------------------------
def install():
    """Entry from radiance_kernels.install_all(). Env-gated on RADIANCE_DYNAMIC_DRAFT."""
    if not DYNAMIC:
        _log("RADIANCE_DYNAMIC_DRAFT=OFF (stock MTP)")
        return
    try:
        import radiance_draft_gpu as gpu
        _GPU[0] = gpu
    except Exception as e:
        _log(f"GPU kernels unavailable ({e!r}) -> stock MTP")
        return
    try:
        _install_drafter_hooks()
        _install_v2_hooks()
        _install_runner_wrap()
        _log(f"RADIANCE_DYNAMIC_DRAFT=ON  controller=policy  tau={TAU}  schedule={SCHEDULE or 'off'}  "
             f"ngram_strong={STRONG} recent={RECENT} maxl={MAXL} "
             f"window={WINDOW if WINDOW else ('auto' if _WINDOW_ENV == 'auto' else 'full')}  "
             f"cross={_CROSS_ENV}  stats={STATS}  match_every={MATCH_EVERY}  "
             f"(GPU-resident capture + top-2 n-gram matcher; per-slot confidence gate "
             f"short-circuits the forward loop)")
    except Exception as e:
        _log(f"install failed (dynamic draft disabled): {e!r}")
