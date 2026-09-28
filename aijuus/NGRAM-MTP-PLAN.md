# MTP + n-gram generation-speed plan

Living tracker for the work to make the `mtp` speculative path faster with n-gram drafting.
Source of the findings: a repository study of the existing controller (`radiance_draft.py`,
`radiance_draft_gpu.py`, `patch_mtp_loopbreak.py`) and its config surface.

**Safety invariant for every item below:** these changes only alter which tokens are *proposed*.
vLLM's rejection sampler verifies every proposal with the target's own weights, so a proposal change
cannot change the output distribution. A bug can only make drafting slower or less accurate, never
the emitted text wrong. That is what makes shipping these behind defaults acceptable without a
serving box attached to this worktree.

Status legend: `DONE` (implemented + statically checked), `PENDING-HW` (implemented, needs an A/B on
the R9700 box), `N/A`.

| # | Item | Files | Status |
|---|------|-------|--------|
| P0 | Plan tracker (this doc) | `NGRAM-MTP-PLAN.md` | DONE |
| P1 | Length-gated n-gram tail (take a long verbatim match without MTP agreement) | `radiance_draft_gpu.py`, `radiance_draft.py` | DONE |
| P2 | Prime the draft from a long match at slot 0 (delivered by P1's slot-0 behaviour) | `radiance_draft.py` | DONE |
| P3 | Re-enable mtp `SPEC=8` ceiling + decode-band/capture derivation | `serve-mxfp4.sh`, docs | PENDING-HW |
| P4 | Top-2 candidate match with fallback to a shorter agreeing candidate | `radiance_draft_gpu.py`, `radiance_draft.py` | DONE |
| P5 | Windowed match (`_NGRAM_WINDOW`) + reusable buffers | `radiance_draft_gpu.py`, `radiance_draft.py` | DONE |
| P6 | Single-sync packed metadata ([cont1|cont2|meta] in one D2H) | `radiance_draft_gpu.py`, `radiance_draft.py` | DONE |
| P7 | Cross-request matching (`_CROSS_REQ`, opt-in, bounded batch) | `radiance_draft_gpu.py`, `radiance_draft.py` | DONE |
| P8 | Counters + periodic stats log (`_STATS`) | `radiance_draft.py` | DONE |
| P9 | Unify `RADIANCE_DRAFT_TAU` default across code/config/docs | `radiance_draft.py`, `serve-mxfp4.sh`, `Dockerfile`, `radiance_preamble.py`, docs | DONE |
| T1 | CPU reference selftest (matcher semantics + slot policy) | `ngram_draft_selftest.py` | DONE |
| T2 | Docs (README / DOCKERHUB / preamble banner) | `README.md`, `DOCKERHUB.md`, `radiance_preamble.py` | DONE |

## What the old path did

At each draft slot `j` the controller captured the drafter's top-1 softmax confidence and required
`cont[j] == mtp[j]` (exact `agree`) before taking the verbatim n-gram continuation; otherwise it
drafted while `cum >= TAU`, else stopped. The tail was truncated to `num_speculative_tokens`.

## Changes

### P1 — length-gated tail
A match of length `mlen >= RADIANCE_DRAFT_NGRAM_STRONG` (default 8) is now taken even when MTP's
top-1 disagrees. Long verbatim repeats (code, JSON, file edits, echo) are the highest-value content
for prompt-lookup, and MTP's single top-1 is a weak gate against a long exact suffix match. Strict
agreement still applies to short matches. `_NGRAM_STRONG=0` restores the old agree-only behaviour.

### P2 — slot-0 priming
Because a strong match at `j=0` yields `action=1` with `sslot=0`, the kept MTP prefix is empty and
the draft is the pure n-gram continuation; `_radiance_stop` then skips the remaining serial MTP
forwards. One forward still runs (slot 0 produces the token/confidence before the decision), so the
saving is up to `nspec-1` forwards, not all `nspec`.

### P3 — mtp ceiling
`serve-mxfp4.sh` pinned mtp to `SPEC=4`, truncating the free tail at 4 (`DOCKERHUB.md` already
recommends 8 under the dynamic controller). Raised to 8. `SPEC=8` makes the verify batch reach
`MAXSEQS*(SPEC+1)=72 > 64`, so `RADIANCE_MXFP4_DECODE_MAX_M` is now derived from the batch shape
(128 when the product exceeds 64) and the CUDA-graph capture list already derives from the same
product. **Must be A/B'd** against `SPEC=4` on BetterBench's weighted mix; rollback is `SPEC=4`.

### P4 — top-2 candidates
The matcher now returns the two best matches per row (length first, then recency). If candidate 1
disagrees and is not strong, candidate 2 is tried before falling back to the confidence gate. This
recovers the case where the longest match's continuation diverges but a different (shorter) match
agrees.

### P5 — search window + buffer reuse
`RADIANCE_DRAFT_NGRAM_WINDOW` (tokens, default `0` = full context) bounds the self-match region.
If a windowed row finds no match, the host re-runs the full scan for that step (correctness
preserved; the common case is still bounded). All matcher scratch (`cand`, `key1`, `key2`, `pack`)
is allocated once per batch shape and reused, removing per-step allocations.

### P6 — one sync per step
`cont1`, `cont2` and the four meta ints travel in a single `pack` tensor, so the step needs one
`.cpu()` instead of the previous `cont.cpu()` + `clen.cpu()`.

### P7 — cross-request matching
`RADIANCE_DRAFT_CROSS_REQ=1` searches every active row for the suffix, not just the request's own
row (continuations stay within the matched row, so no cross-request bleed). Bounded to
`RADIANCE_DRAFT_CROSS_MAX` (default 8) requests because the scan is `O(B^2)`. Default off.

### P8 — observability
`RADIANCE_DRAFT_STATS=1` (default on) logs match/tail/strong counters every
`RADIANCE_DRAFT_STATS_EVERY` steps (default 200) to stderr under `[radiance.draft]`.

### P9 — TAU unification
The shipped serve passed `0.20`, the image and docs said `0.35`, and the code fallback was `0.28`.
Unified on **0.20** (the production value) in code, Dockerfile, preamble, docs and serve.

## Rollback / tuning knobs added

| Env var | Default | Meaning |
|---|---|---|
| `RADIANCE_DRAFT_NGRAM_STRONG` | `8` | min match length to take a tail without MTP agreement; `0` disables |
| `RADIANCE_DRAFT_NGRAM_WINDOW` | `0` | self-match search window in tokens; `0` = full context |
| `RADIANCE_DRAFT_CROSS_REQ` | `0` | search all active requests' rows |
| `RADIANCE_DRAFT_CROSS_MAX` | `8` | max batch for cross-request search |
| `RADIANCE_DRAFT_STATS` | `1` | periodic controller counters |
| `RADIANCE_DRAFT_STATS_EVERY` | `200` | steps between stats lines |

`RADIANCE_DYNAMIC_DRAFT=0` remains the master off switch (byte-identical stock MTP).

## Round 2 — recall + cost (requested follow-up)

| # | Item | What landed | Status |
|---|------|-------------|--------|
| R1 | Tune thresholds | `bench-ngram-sweep.sh` runs a STRONG x RECENT grid over the blend-OCP MTP target, one overview per cell | DONE (needs HW run) |
| R2 | Recency-weighted gate | matcher now returns each candidate's distance back (`rec1`/`rec2`); `slot_decide` takes a short match (`>=3`) within `RADIANCE_DRAFT_NGRAM_RECENT`. **Default `0` (off)** — measured dormant on the blend-OCP probe (0-2 hits / ~400 steps); raise it for corpora with short near-term repeats. The length gate (`STRONG`) is what actually fires | DONE |
| R3 | Cross-request matching | `RADIANCE_DRAFT_CROSS_REQ=auto` (default): enabled only for `2<=B<=CROSS_MAX` and `nmax<=CROSS_CTX_MAX`; `1`/`0` override | DONE |
| R4 | Longer match horizon | `RADIANCE_DRAFT_NGRAM_MAXL` (default 32, was 24), passed to the kernels | DONE |
| R5 | Bounded matcher cost at long context | `RADIANCE_DRAFT_NGRAM_WINDOW=auto` (default): full up to `AUTO_FULL` (32768) then last `WINDOW_TOKENS` (16384), full-scan fallback on a miss | DONE |
| R6 | Wider `N` support | capture list extended to 256 so `SPEC>8` (up to ~16) is graphable; decode band already derived from `MAXSEQS*(SPEC+1)` | DONE (default stays 8) |
| R7 | Skip the first MTP forward | **Not applicable**: the baked proposer reuses the target's hidden states for slot 0 (`llm_base_proposer.propose`), so there is no separate first MTP forward to skip. The only serial forwards are slots 1..N-1, already gated by `_radiance_stop` | N/A (verified against source) |

`_META` grew to 7 (`clen1, mlen1, clen2, mlen2, window_miss, rec1, rec2`); the host reads the recency and
the token split, and the selftest's pack-shape guard tracks it.

## Fast A/B harness (~5 min instead of 25-30)

`bench-quick.sh` boots one `serve-mxfp4.sh` instance, waits for health, runs `bench-quick.py` (a short
single-stream decode probe over 4 fixed prompts -- echo/code/json/prose -- reporting TTFT, decode
tok/s, ms/step, acc/draft, dup-8gram and the `vllm:spec_decode_*` deltas), grabs the controller's
`[radiance.draft] stats` lines, and tears down. `bench-quick-ab.sh` runs a baseline and a candidate and
diffs them. `bench-ngram-sweep.sh` is the threshold grid on the same probe.

```bash
bash bench-quick-ab.sh                                    # SPEC=4 vs SPEC=8, n-gram on
A_SPEC=8 B_SPEC=8 A_STRONG=0 B_STRONG=8 bash bench-quick-ab.sh   # isolate the length gate
A_RECENT=0 B_RECENT=16 A_SPEC=8 B_SPEC=8 bash bench-quick-ab.sh  # isolate the recency gate
```

### Thinking-on probe

vLLM 0.27.1 streams qwen3 reasoning under `delta.reasoning` (not `reasoning_content`); the parser timed
on `content` only, so every thinking-on sample looked like one buffered chunk -> `[invalid]` rows and a
fake `1e11 tok/s`. Fixed in `bench-quick.py`: time on `content or reasoning`, and require **two** chunks
(`last > tf`) for a sample to count. Run thinking-on with `BENCH_THINKING=1` (env passes through
`bench-quick.sh`). Matched probe (gen 160, 2 reps, thinking on):

| arm | combined decode | acc/draft | ms/step |
|---|---|---|---|
| mtp `SPEC=4` | 72.6 tok/s | 1.9-2.8 | ~47 |
| mtp `SPEC=8` | **76.0 tok/s** | 2.5-4.1 | ~57 |

`SPEC=8` stays ahead with thinking on (+4.7%), same direction as thinking-off (+15%), so the default
stays 8.

### Policy-knob sweep (STRONG x TAU)

Ran live on ONE `SPEC=8` boot via `RADIANCE_DRAFT_KNOB_FILE` (no relaunch between cells), thinking on,
gen 160 x 2 reps. Combined decode spans 71.7-79.0 tok/s with no monotonic trend in either knob
(per-category `code` alone swings 68-91 across near-identical arms), so the single-stream probe cannot
resolve these gates -- the shipped `STRONG=8` / `TAU=0.20` sit at the top of the spread (non-prose mean
90.5 tok/s, best cell) and no change is justified. Also fixed `bench-quick.sh`: the optional knob file
was passed as `${VAR:+NAME="$VAR"}`, a word that starts with `$` so bash ran it as a command instead of
an env assignment; now built with an `env` array.

## Verification checklist

- [x] `ast.parse` every changed Python file
- [x] `bash -n serve-mxfp4.sh` (and `gpu-detect.sh`, `calibrate-kv.sh`, `bench-ngram-sweep.sh`)
- [x] `python3 ngram_draft_selftest.py` (key round-trip, matcher reference incl. window/cross/top-2/same-block, slot policy)
- [x] Six-track code review of the uncommitted diff; all findings fixed (see above)
- [x] **GPU correctness of the matcher kernels**, on a real R9700 (torch 2.11.0+rocm7.14, triton 3.6.0):
  `docker run --rm --privileged --ipc=host --network=host --device /dev/kfd --device /dev/dri \
  --group-add 993 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 -v <repo>:/patches:z --entrypoint bash \
  stilldeadcode/vllm-radiance:0.9.3 -lc 'cd /patches && python3 ngram_draft_gpu_selftest.py'`
  -> **ALL PASS** (9 cases: self/no-match/same-block top-2/window on+off/cross on+off/heterogeneous batch/708-token block-boundary)
- [ ] HW A/B: mtp `SPEC=8` + P1 vs stock mtp `SPEC=4` on BetterBench weighted mix, single-stream + conc-8 (needs a full serve)
- [ ] HW: confirm `RADIANCE_DRAFT_STATS` shows tail rate climbing on code/json/file_edit
- [x] HW: sweep `_NGRAM_STRONG` (4/6/8/12/16) x `TAU` (0.20/0.28) -> no resolvable difference; defaults stay 8 / 0.20 (see above)
- [ ] HW: verify no regression on prose (strong=8 should be inert there)
- [ ] HW: verify `RADIANCE_DRAFT_NGRAM_WINDOW` and `_CROSS_REQ` (both default off) do not change output

## Files changed

| File | Change |
|---|---|
| `radiance_draft_gpu.py` | matcher rewritten: `_match_scan` (self), `_match_scan_x` (cross), `_match_top2`, `_match_gather`; int64 key `[mlen\|row\|end_pos]`; `make_match_buffers`; `match_gpu` returns a single `pack` tensor. Capture kernels untouched |
| `radiance_draft.py` | `slot_decide` pure policy; length gate (`STRONG`), top-2 fallback, window, cross-request, one-sync pack, counters; TAU default 0.20; docstring rewritten |
| `ngram_draft_selftest.py` | new CPU reference test (matcher semantics + policy + key encoding + pack-shape contract) |
| `ngram_draft_gpu_selftest.py` | new GPU test: runs the real Triton matcher kernels on the R9700 and diffs them to the reference |
| `bench-ngram-sweep.sh` | new: STRONG x RECENT threshold sweep over the blend-OCP MTP target |
| `bench-quick.py`, `bench-quick.sh`, `bench-quick-ab.sh` | new: ~5-min single-stream decode probe + A/B launcher |
| `serve-mxfp4.sh` | mtp `SPEC` default 4 -> 8; `MAXSEQS`/`SPEC` integer validation; decode band derived after the profile block (64-multiple of `MAXSEQS*(SPEC+1)`); KV lookup passes `SPEC`; comments |
| `gpu-detect.sh` | `rad_kv_lookup` gains `method:depth` key matching (bare `method` still allowed for dflash, not mtp) |
| `calibrate-kv.sh` | derives and validates `SPEC`; writes `method:depth` keys |
| `kv-profiles.tsv` | header documents the `method:depth` key |
| `Dockerfile`, `Dockerfile.ggz14` | `RADIANCE_DRAFT_TAU=0.35` -> `0.20` |
| `radiance_preamble.py` | TAU default 0.20; new n-gram knobs in the banner |
| `README.md`, `DOCKERHUB.md` | feature text, new knob rows, mtp `SPEC` guidance |

## Review findings resolved

A six-track review of the first implementation found and this revision fixed:

| Finding | Fix |
|---|---|
| P4 top-2 was per-block maxima, so the true 2nd match was dropped and `key2` was always 0 for contexts <512 | `_match_scan`/`_match_scan_x` now emit a per-block **top-2**; `_match_top2` reduces those (exact). Selftest asserts candidate 2 in the same-block case |
| `slot_decide`'s `cum`/`tau` params were unused; the gate lived in `greedy_sample` | The full decision (including `cum >= tau`) now lives in `slot_decide`; it returns `action`, tested |
| Forward-cap `allagree` branch was unreachable | Removed, along with the `allagree` state |
| Window fallback did a second `.cpu()` and a whole-batch full rescan | A `window_miss` flag rides back in the single `pack` copy; `base` is now a reused buffer |
| Decode band hardcoded 128 while the requirement is `MAXSEQS*(SPEC+1)` | Derived: the product rounded up to a 64-multiple (192 at `MAXSEQS=16`, `SPEC=8`) |
| Decode band read `MAXSEQS` before the TP=1 profile narrowed it to 3 | Moved after the profile block, so it reads the final `MAXSEQS` |
| `$(( ${MAXSEQS:-8} * ... ))` parsed raw env text as an expression | `MAXSEQS` and `SPEC` are validated integers before any arithmetic (serve + calibrate) |
| mtp `SPEC=8` reused a KV pin keyed without `SPEC` | `rad_kv_lookup` keys on `method:depth`; bare-`method` rows still satisfy dflash but not mtp, so an old mtp pin falls back to profiling |
| Host pack-shape checks pinned `2*N + 4` after `_META` became 5 (would silently fall back to native drafts every step) | Both checks now use `gpu._META`; a selftest guard asserts the contract |
| New knobs were baked-only, so the HW tuning checklist could not reach them | `serve-mxfp4.sh` now forwards `RADIANCE_DRAFT_NGRAM_*`, `_CROSS_*`, `_STATS*`, `_SCHEDULE` |

## Notes / known non-issues

- All changes are proposal-only. Even a matcher bug can only cost acceptance, never output correctness.
- The cross-request path is opt-in and batch-bounded; continuations are read from the matched row, so
  no cross-request token bleed is possible.
- The matcher kernels are validated on gfx1201 (`ngram_draft_gpu_selftest.py`, ALL PASS). What remains
  `PENDING-HW` is only the `SPEC=8` mtp default and its decode-band widening (P3): the throughput A/B
  against `SPEC=4` on the weighted mix needs a full serve.


