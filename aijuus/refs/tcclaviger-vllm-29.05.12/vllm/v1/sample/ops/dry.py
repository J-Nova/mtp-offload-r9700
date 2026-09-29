# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) repetition penalty.

Sequence-level repetition penalty: if emitting token T would extend a token
n-gram that already appears earlier in the context, T is penalized by

    multiplier * base ** (match_len - allowed_length)

where ``match_len`` is the length of the longest suffix of the context that
(a) also occurs earlier in the context and (b) is followed there by T. Unlike
presence/frequency/repetition penalties, which are bag-of-tokens, DRY only
punishes tokens that would continue an actual repeat, so it suppresses looping
without flattening ordinary token reuse.

Implementation notes
--------------------
This is the O(N) formulation from llama.cpp (``src/llama-sampler.cpp``, the
``llama_sampler_dry_apply`` steps 1-4), NOT the O(matches * max_ngram) scan used
by the vLLM PR #11368 prototype (which walked every prior occurrence of the last
token in Python and called ``.item()`` inside the inner loop -- a GPU sync per
position). The match lengths for every suffix come out of a single reverse
Z-algorithm pass, so cost is linear in the scanned window regardless of how
repetitive the context is.

Defaults deliberately mirror llama.cpp (``common/common.h``) so that turning DRY
on with no tuning reproduces llama.cpp behavior: base 1.75, allowed_length 2,
range -1 (whole context), breakers ``\\n``, ``:``, ``"``, ``*``.

The penalty is applied inside ``apply_all_penalties``, which is the single point
reached by BOTH the non-speculative path (``Sampler.apply_penalties``) and the
speculative/MTP path (``RejectionSampler.apply_penalties``). The rejection
sampler pre-expands its per-request rows via ``repeat_indices`` and builds the
cumulative per-draft-position output prefixes, so by the time we are called both
modes look identical and DRY needs no spec-decode special case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# llama.cpp defaults (common/common.h: dry_multiplier / dry_base /
# dry_allowed_length / dry_penalty_last_n / dry_sequence_breakers).
DRY_DEFAULT_MULTIPLIER = 0.0  # 0 => disabled
DRY_DEFAULT_BASE = 1.75
DRY_DEFAULT_ALLOWED_LENGTH = 2
DRY_DEFAULT_RANGE = -1  # -1 => whole context
DRY_DEFAULT_BREAKERS: tuple[str, ...] = ("\n", ":", '"', "*")

# The server-level ramp is expressed as a 0.xx number and maps to the
# exponential base as ``base = 1 + ramp``. 0.75 -> 1.75 == llama.cpp's default,
# so the shipped default is behaviorally identical to llama.cpp.
DRY_DEFAULT_RAMP = DRY_DEFAULT_BASE - 1.0

# llama.cpp clamps breaker tokenization to bound the step-1 restart scan
# (llama-sampler.cpp: MAX_CHAR_LEN / MAX_SEQ_LEN). Without the clamp that scan
# is worst-case O(N^2) for adversarial breaker strings.
DRY_MAX_BREAKER_CHAR_LEN = 40
DRY_MAX_BREAKER_SEQ_LEN = 20

# Prevents overflow in base ** exponent (llama-sampler.cpp uses the same
# constant: log(FLT_MAX)).
_FLOAT_MAX_LOG = 88.7228391


def ramp_to_base(ramp: float) -> float:
    """Server-level 0.xx ramp -> exponential base. See DRY_DEFAULT_RAMP."""
    return 1.0 + ramp


@dataclass
class DryParams:
    """Resolved per-request DRY configuration.

    ``breaker_seqs`` maps a *head* token id to the list of token-id tails that,
    together with the head, form a breaker sequence. A single-token breaker has
    an empty tail. This mirrors llama.cpp's ``dry_processed_breakers`` multimap
    and lets a breaker that tokenizes to several tokens still be recognized.
    """

    multiplier: float = DRY_DEFAULT_MULTIPLIER
    base: float = DRY_DEFAULT_BASE
    allowed_length: int = DRY_DEFAULT_ALLOWED_LENGTH
    # Tokens of trailing context to scan. -1 => everything, 0 => disabled.
    range: int = DRY_DEFAULT_RANGE
    breaker_seqs: dict[int, list[list[int]]] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value) -> "DryParams | None":
        """Coerce a DryParams OR its serialized dict form into a DryParams.

        REQUIRED because SamplingParams is a msgspec.Struct that crosses the
        API-server -> EngineCore -> worker process boundary. The ``_dry_params``
        field is typed ``Any``, so msgspec encodes this dataclass to a plain dict
        and does NOT rebuild it on the far side; the worker therefore sees a dict
        and any attribute access on it raises AttributeError. Normalize once, at
        the point worker-side state is built (gpu_input_batch.add_request), so
        everything downstream holds a real DryParams.

        msgpack preserves int dict keys, but a JSON hop would stringify them, so
        breaker_seqs keys are coerced defensively.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"cannot build DryParams from {type(value).__name__}")
        raw = dict(value)
        breakers = raw.pop("breaker_seqs", None) or {}
        coerced: dict[int, list[list[int]]] = {}
        for k, tails in breakers.items():
            coerced[int(k)] = [[int(t) for t in tail] for tail in (tails or [])]
        return cls(
            multiplier=float(raw.get("multiplier", DRY_DEFAULT_MULTIPLIER)),
            base=float(raw.get("base", DRY_DEFAULT_BASE)),
            allowed_length=int(
                raw.get("allowed_length", DRY_DEFAULT_ALLOWED_LENGTH)
            ),
            range=int(raw.get("range", DRY_DEFAULT_RANGE)),
            breaker_seqs=coerced,
        )

    @property
    def enabled(self) -> bool:
        # Matches llama.cpp's dry_enabled gate: a base below 1.0 would make the
        # penalty shrink as the repeat grows, which is never intended.
        return self.multiplier != 0.0 and self.base >= 1.0 and self.range != 0

    @property
    def max_exponent(self) -> int:
        if self.base > 1.000001:
            import math

            return int(_FLOAT_MAX_LOG / math.log(self.base))
        return 0


def build_breaker_seqs(
    breakers: tuple[str, ...], tokenizer
) -> dict[int, list[list[int]]]:
    """Port of llama.cpp's ``get_overlapping_token_sequences``.

    For each breaker string, find every token that could *begin* an occurrence
    of it, and record what must follow:

    * A token whose text CONTAINS the breaker is itself a breaker -> empty tail.
      These are also exempt from being penalized (a token that ends a repeat
      should not be pushed down, or the model is nudged to keep repeating).
    * A token whose text ENDS WITH a proper prefix of the breaker starts a
      multi-token breaker -> the tail is the tokenization of the remainder.

    This is a full vocabulary scan per breaker set, so the result MUST be cached
    by the caller (see input_processor.py) rather than rebuilt per request.
    Tails are clamped to ``DRY_MAX_BREAKER_SEQ_LEN`` and breakers longer than
    ``DRY_MAX_BREAKER_CHAR_LEN`` are truncated, matching llama.cpp -- the clamp
    is what keeps the step-1 restart scan linear.
    """
    out: dict[int, list[list[int]]] = {}

    def _add(token_id: int, tail: list[int]) -> None:
        tails = out.setdefault(token_id, [])
        if tail not in tails:
            tails.append(tail)

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for raw in breakers:
        if not raw:
            continue
        s = raw[:DRY_MAX_BREAKER_CHAR_LEN]
        str_len = len(s)
        for token_id in range(vocab_size):
            try:
                word = tokenizer.decode([token_id])
            except Exception:
                continue
            if not word:
                continue
            if s in word:
                _add(token_id, [])
                continue
            # Does the token's text end with a proper prefix of the breaker?
            word_len = len(word)
            pos = word.find(s[0])
            while pos != -1:
                i = 1
                match = True
                while i < str_len and i + pos < word_len:
                    if word[pos + i] != s[i]:
                        match = False
                        break
                    i += 1
                if match and i < str_len:
                    # The word ran out mid-breaker: the rest must follow as its
                    # own tokens.
                    tail = tokenizer.encode(s[i:], add_special_tokens=False)
                    _add(token_id, list(tail[:DRY_MAX_BREAKER_SEQ_LEN]))
                pos = word.find(s[0], pos + 1)
    return out


def _restart_limit(tokens: list[int], params: DryParams) -> int:
    """Step 1: longest suffix length that does not cross a breaker sequence.

    Scans backwards for the most recent breaker; the repeat length is capped so
    a match can never span it. Returns the cap (``len(tokens)`` if no breaker is
    in range).
    """
    n = len(tokens)
    rep_limit = n
    for i in range(n):
        # tokens[n - 1 - i] is llama.cpp's rat(i): i tokens back from the end.
        head = tokens[n - 1 - i]
        tails = params.breaker_seqs.get(head)
        if not tails:
            continue
        longest_match = -1
        for tail in tails:
            seq_len = len(tail)
            # The head is already matched, so the tail must fit behind it.
            if seq_len <= longest_match or seq_len > i:
                continue
            if all(
                tail[offset] == tokens[n - 1 - (i - offset - 1)]
                for offset in range(seq_len)
            ):
                longest_match = seq_len
        if longest_match >= 0:
            rep_limit = i - longest_match
            break
    return rep_limit


def _suffix_match_lengths(tokens: list[int], rep_limit: int) -> list[int]:
    """Step 2: reverse Z-algorithm.

    ``out[i]`` is the length of the longest common prefix between the reversed
    context and the reversed context starting at position ``i`` -- i.e. how many
    of the context's trailing tokens also appear ending at ``i``. Linear time:
    the ``lt``/``rt`` Z-box bounds guarantee each token is examined once across
    the whole loop, despite the inner whiles.
    """
    n = len(tokens)
    out = [0] * n
    last = n - 1
    lt = 0
    rt = 0
    for k in range(1, n):
        if k > rt:
            # Outside the current Z-box: compare directly.
            m = 0
            while m + k < n and tokens[last - m] == tokens[last - (m + k)]:
                m += 1
            out[last - k] = min(m, rep_limit)
            if m > 0:
                lt = k
                rt = k + m - 1
        else:
            p = k - lt
            right_part_len = rt - k + 1
            if out[last - p] < right_part_len:
                # Fully determined by the mirrored position; no comparisons.
                out[last - k] = min(out[last - p], rep_limit)
            else:
                # Extend past the Z-box, then re-anchor it.
                i = rt + 1
                while i < n and tokens[last - i] == tokens[last - (i - k)]:
                    i += 1
                out[last - k] = min(i - k, rep_limit)
                lt = k
                rt = i - 1
    return out


def _max_token_repeat(
    tokens: list[int], match_lens: list[int], allowed_length: int
) -> dict[int, int]:
    """Step 3: for each candidate next token, the longest repeat it would extend.

    A non-zero ``match_lens[i]`` means the context's tail recurs ending at ``i``;
    the token at ``i + 1`` is what continued it last time, so emitting that token
    now would extend the repeat. By convention the length excludes the new token.
    """
    out: dict[int, int] = {}
    for i in range(len(tokens) - 1):
        repeat_len = match_lens[i]
        if repeat_len >= allowed_length:
            token = tokens[i + 1]
            if out.get(token, -1) < repeat_len:
                out[token] = repeat_len
    return out


def compute_dry_penalties(
    tokens: list[int], params: DryParams
) -> tuple[list[int], list[float]]:
    """Return ``(token_ids, penalties)`` for one row. Empty if nothing to do."""
    if not params.enabled:
        return [], []

    if params.range > 0:
        tokens = tokens[-params.range :]
    n = len(tokens)
    # Need at least one token to match plus one to extend it.
    if n < 2:
        return [], []

    # A breaker as the very last token means nothing can be extended across it.
    if tokens[-1] in params.breaker_seqs:
        return [], []

    rep_limit = _restart_limit(tokens, params)
    if rep_limit < params.allowed_length:
        return [], []

    match_lens = _suffix_match_lengths(tokens, rep_limit)
    repeats = _max_token_repeat(tokens, match_lens, params.allowed_length)
    if not repeats:
        return [], []

    max_exponent = params.max_exponent
    token_ids: list[int] = []
    penalties: list[float] = []
    for token, repeat_len in repeats.items():
        # A single-token breaker is never penalized: it is the thing that ends a
        # repeat, so penalizing it would push the model to keep going.
        tails = params.breaker_seqs.get(token)
        if tails is not None and any(not tail for tail in tails):
            continue
        exponent = repeat_len - params.allowed_length
        if max_exponent and exponent > max_exponent:
            exponent = max_exponent
        token_ids.append(token)
        penalties.append(params.multiplier * (params.base**exponent))
    return token_ids, penalties


def apply_dry(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: list[list[int]],
    dry_params: dict[int, DryParams],
    vocab_size: int,
) -> torch.Tensor:
    """Apply DRY in-place to the rows named in ``dry_params``.

    ``prompt_token_ids`` is the padded [num_rows, max_prompt_len] tensor the
    penalty path already builds; pad cells hold ``vocab_size`` (and, under async
    scheduling, ``-1``), so both are stripped. ``output_token_ids[row]`` is the
    live decoded-token list for that row -- under speculative decoding the caller
    has already appended the draft tokens, so no spec handling is needed here.

    All rows' penalties are gathered and applied as ONE fused scatter, so the
    number of GPU ops does not grow with batch size.
    """
    if not dry_params:
        return logits

    num_rows = logits.shape[0]
    # Move prompts to host once; the match search is inherently sequential.
    prompts_cpu = prompt_token_ids.tolist()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row, params in dry_params.items():
        if row >= num_rows or not params.enabled:
            continue
        prompt = [t for t in prompts_cpu[row] if 0 <= t < vocab_size]
        output = output_token_ids[row] if row < len(output_token_ids) else []
        tokens = prompt + [t for t in output if 0 <= t < vocab_size]
        token_ids, penalties = compute_dry_penalties(tokens, params)
        if not token_ids:
            continue
        rows.extend([row] * len(token_ids))
        cols.extend(token_ids)
        vals.extend(penalties)

    if not rows:
        return logits

    device = logits.device
    row_t = torch.tensor(rows, dtype=torch.long, device=device)
    col_t = torch.tensor(cols, dtype=torch.long, device=device)
    val_t = torch.tensor(vals, dtype=logits.dtype, device=device)
    # SUBTRACT the penalty. Each (row, token) pair is unique -- `repeats` is keyed
    # by token id per row -- so an in-place indexed subtract needs no accumulate.
    logits[row_t, col_t] -= val_t
    return logits
