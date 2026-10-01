# E3 — Lazy GDN snapshots: deep research (2026-10-01)

Status: research only; no code changed. Companion to WORKLOG cont.65-67 and OPEN-TASKS-INDEX §13/OS3.
Sources: `r4d_gdn_lazy_update_k128_v128.h` (from `~/.cache/radiance-libr4d/b9e42ab-rx10/`, i.e. the
rx10 patch applied), the live patched `vllm/v1/worker/mamba_utils.py`,
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`, `vllm/v1/worker/gpu/model_runner.py`,
`radiance_gdn_lazy.py`, `patch_gdn_lazy.py`.

## 1. What lazy changes

Eager GDN spec-decode stores one full `[V,K]` state per candidate token per head (T snapshots/step),
because which candidate survives verification is unknown until after the layer runs. Lazy keeps:

- the **base**: `sidx[n, 0]` — the state after every accepted token;
- a **stash**: `sidx[n, 1]` — the previous step's candidate *inputs* (post-conv q/k/v and gates),
  from which the accepted candidates `1..naccept-1` are replayed on the base.

MambaSpec gets **one** speculative block instead of `num_speculative_tokens`
(`_radiance_lazy_spec_blocks` returns 1), so a request's mamba window is 2 columns
(`state_idx` = running, `state_idx+1` = stash) instead of `1+SPEC`. That is the 9→3 page drop
(`+11%` KV pool measured, 171,320→190,157).

The stash lives in head `hv`'s state region of the stash *slot*: `state + stash_slot*st_slot +
hv*st_head`, `LZ_HEAD_BYTES=16384` (two `LZ_REGION=8192` splits), header 64 B + a,b per candidate
64 B + `LZ_MAXT=8`×(q256|k256|v256). The stash slot is a scratch block; it is not a real state.

## 2. The two GPUkernel entry points

**Update** (`r4d_gdn_lazy_update_kernel`, called by the GDN decode layer):
1. load base `sidx[n,0]`;
2. if `s_hdr.magic==LZ_HDR_MAGIC && s_hdr.base_slot==sidx[n,0]` replay accepted `1..naccept-1`
   from the stash, else replay nothing;
3. run candidate 0, store its state back to `sidx[n,0]`;
4. run candidates 1..T-1, store their inputs into `sidx[n,1]`;
5. write header `{magic, count=T-1, base_slot=sidx[n,0], vs_w}`.

**Materialize** (`r4d_gdn_lazy_materialize_kernel`, called by `radiance_gdn_lazy.materialize` from
the two vLLM align copies; one launch for all 48 layers, driven by GPU-resident inputs, no host sync).
It writes `base + replay(stash[0..r-1])` into a destination slot, where
`r = count` if the stash header matches `base_slot`, else **`r=0` (fail-open → pure base)**:

- **MODE 0** (pre-forward migration, `run_fused_precopy`): `base=bt[src_col]`,
  `stash=bt[src_col+1]`, `count=token_bias` (= pre-reset `num_accepted-1`), `dst=bt[dst_col]`.
  `dst_col == src_col+1` (the window slid by one, so `dst` IS the old stash block — read then
  overwrite with a barrier). Afterwards it **zeroes the header** of `bt[dst_col+1]`, the new stash
  block, "so a stale stash can never replay".
- **MODE 1** (post-step checkpoint, `run_fused_postprocess_align`): `base=bt[state_idx]` (current
  running col), `stash=bt[state_idx+1]`, `count=aligned-running`
  (`running=newc-naccept+1`, `aligned=floor(newc/block_size)*block_size`), `dst=bt[aligned/bs-1]`.
  This writes the block-aligned checkpoint that a future prefix hit resumes from.

## 3. Align plumbing (`mamba_hybrid.preprocess_state` / `postprocess_state`)

- `preprocess_mamba_align_fused_kernel` (every step): `src_col=state_idx`, `dst_col=
  ceil((num_computed+query_len)/MAMBA_BLOCK)-1`, `token_bias=max(naccept-1,0)`; stores src/dst;
  **advances `state_idx` and resets `naccept=1` when a boundary is crossed**.
- `postprocess_state` → `run_fused_postprocess_align(num_accepted, state_idx,
  new_num_computed, idx_mapping)` picks the MODE 1 checkpoint.
- Both kernels run every step and fast-exit per row (`src<0`, `src==dst`, `aligned<running`,
  `dst==base && count==0`, `stash_slot<=0`).

Consumer comparison: the *eager* `_copy_mamba_state_block` for a temporal state copies
`state[bt[src_col + token_bias]] -> state[bt[dst_col]]` — i.e. it reads the **candidate-snapshot
column**. Lazy makes that column not exist, so materialize must synthesise exactly that state from
`base + replay`. The correctness contract: **materialize's output must equal the eager column
`bt[src_col+token_bias]`.**

## 4. Why it corrupts (known + analysis)

Team root cause (WORKLOG cont.42-43, cont.50): *"a libr4d materialize kernel that fails open
(`r=0` stores the base as the checkpoint) whenever a prefix hit invalidates the stash."* The kernel
author's own comment claims fail-open base is right "after a prefill / prefix hit" — the two
disagree, and the multi-turn README failure (empty replies/repeat loops from ~turn 5) says the
team is right.

The header records **only `base_slot`** (a physical slot id), so there are two ways a stale stash is
used:
- **(H1) true mismatch → fail-open.** Stash header invalid (block moved, or zeroed by a MODE 0
  invalidation that the subsequent update did not repair) and `count>0` → materialize stores the
  **base** where the eager path would store the **candidate** state. Wrong whenever `count>0`.
- **(H2) false match → stale replay.** Prefix caching reuses the same physical blocks across turns,
  so a *new* turn's running slot can equal a *previous* turn's stash `base_slot`; header matches,
  contents are the old turn's candidates → wrong replay.

Both produce a wrong MODE 1 checkpoint, which a later prefix hit resumes from — matching the
"symptom on multi-turn, fine on single-shot" signature.

## 5. Consequences for the fix

- **Pure-Python route 3c is infeasible.** The fail-open decision is inside the kernel; Python cannot
  observe it without a per-step readback, and cannot *repair* it: the value needed is the state after
  `count` candidates, whose only source is the stash's candidate inputs. A host ring holding that is
  ~18 MB/req/step.
- **K2 (one extra backup page) may not fix H2**, because a backup keyed the same way (magic+base_slot)
  false-matches identically. It could help H1 *iff* the invalidation only zeroes the primary stash
  header while the data survives in an un-zeroed backup slot — that depends on the exact column
  slide/invalidations, not yet confirmed.
- **A generation key likely fixes both:** store a per-write epoch/frontier in the header (not just
  `base_slot`) so a stash is trusted only for the base it was written against in the same generation;
  a mismatch then fails open to base, which is *correct* exactly in the H1/H2 cases (stale stash) but
  still wrong for a legitimate first-step-after-prefill MODE 1 with `count>0`. Need to confirm that
  MODE 1's `count>0` never legitimately needs a stash the update did not just write.

## 6. The decisive experiment (next step, needs a ~1 min rx10 rebuild)

Instrument, do not guess:
1. Add device counters to the materialize kernel: `n_failopen` (header mismatch && `count>0`),
   `n_stale_replay` candidates, and a small ring dumping `{mode, req, count, base_slot,
   hdr.magic, hdr.base_slot, stash_slot}` on the mismatch path. Expose via a debug env.
2. Repro harness: a tight multi-turn chat with guaranteed prefix hits (turn N+1 reuses turn N),
   temp 0, comparing lazy vs non-lazy greedy tokens; find the first divergent turn and correlate with
   the dumped events. (Faster and more targeted than `turnbench --concurrent`.)
3. Read the counters to decide H1 vs H2, then implement: generation key (if H2) or backup/invalidate
   fix (if H1).

## 7. Open questions
- Does the update kernel always (re)write `sidx[n,1]` before a MODE 1 that needs `count>0`, on the
  step after a prefix hit? (Determines whether H1 is reachable without H2.)
- Is `sidx[n,1]` the same physical block as `bt[state_idx+1]` the materialize reads? (The update uses
  `spec_state_indices_tensor[:,1]`; materialize uses the mamba group block table — confirm identity.)
- Under K2, does the backup column survive the MODE 0 invalidation and hold the *current* base's
  candidates?

## 8. Concrete lead: the intended prefill-invalidation is DEAD

`patch_gdn_lazy.py`'s gdn_attn edits add `radiance_stash_indices` to `GDNAttentionMetadata`
(`gdn_attn.py:341`, `:688`, set to `bt[:bs, 1]` / `block_table_tensor[:, 1]`, "the stash block of
every batch row, for prefill invalidation"). But a tree-wide search shows **nothing consumes it** —
it is set and never read:

```
$ grep -rn radiance_stash_indices vllm/ | grep -v 'None = None' | grep -v ': torch'
gdn_attn.py:341   radiance_stash_indices=(bt[:bs, 1] if self._rad_lazy else None)
gdn_attn.py:688   radiance_stash_indices=(block_table_tensor[:, 1] if self._rad_lazy else None)
```

So the design's "a prefill invalidates its stash so a stale stash can never replay" step was **never
wired**. The only invalidation that exists is MODE 0 zeroing `bt[dst_col+1]` (a migration). A request
that is prefilling / resumed from a prefix hit never has its `sidx[n,1]` header cleared. Combined
with prefix caching reusing the same physical blocks (same `base_slot` across turns), this is exactly
H2 (false match → stale replay) and is a strong candidate for the root cause. It is also **fixable
without a kernel rebuild** (the metadata is already there; wire the zeroing), unlike K2.

Note this also reconciles the kernel author's comment ("header mismatch → base is right after a
prefill"): the design *relied* on the stash being invalidated at prefill so the mismatch would
happen; the python half that performs the invalidation is missing.
