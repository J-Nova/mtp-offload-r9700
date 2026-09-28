# MTP speedup plan — make `SPEC_METHOD=mtp` genuinely fast (from first principles)

Status: **planning only — no code changed.** Written 2026-09-27 after extracting and
reading the served tree (`stilldeadcode/vllm-radiance:0.9.3` = **vLLM 0.27.1**) and
diffing against upstream `main`. Updated same day with no-GPU static findings
(M0 answered from disk; GPUs busy with quantization, so no boots yet).
Further updated same day with a three-way static audit (graph migration, head /
controller contracts, alternative methodologies) plus primary-source checks
(GDN Tree-Scan arXiv:2609.23900, TreeWY arXiv:2608.20961, upstream fused-decode
history). **Corrections in §8 supersede the struck claims elsewhere — read §8
before acting on §§2-4.** No performance numbers below are predictions; every
track keeps its measurement gate.

Ground rule for this doc (per request): **prior WITHDRAWN / deferred / "not viable"
notes are ignored.** Only physics, hardware, and data-dependency limits count.
Rebuilds, rewrites, new files, new speculators, and model work are all in scope.
dflash is out of scope — this doc is MTP-only. V1 vs V2 is a means, not a constraint:
pick whichever runner makes MTP fastest, port what must be ported.

## 1. Objective and baseline

- Target: `Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ`, `method=mtp`, `SPEC=8`,
  `MAXSEQS=8`, single R9700 (gfx1201, TP=1).
- Baseline (plan's `bench/overview.json` + progress log): **~50-56 ms/step,
  tok/update ~4.6, combined ~80-93 t/s**, TTFT p50 ~88 ms.
- Roofline: weights ~14.3 GB/step → ~22.3 ms DRAM floor @640 GB/s; achievable
  streaming ~574-580 GB/s. **~28-34 ms/step is non-weight overhead.** That is the
  whole game — the verify forward is near the floor, the draft loop + host chain
  is not.
- Measured split (PHASE row, mtp 205k): per slot-call head 0.39 / capture 0.07 /
  argmax 0.02 / **d2h 1.82** / decide 0.056 ms; per step sample(rejection) 1.0 /
  **`_bookkeeping_sync` 28.4** / prepare 0.39 / propose_all 16.5 (slot_total 13.3 +
  loop/fwd slack 2.9) / postprocess 0.07 ms. Step traces: MTP host chain ~52 ms =
  step (host-bound); dflash host chain ~6 ms (GPU-bound). Controller host work
  itself is ~0.5 ms — **the controller is not the cost; the 7 serial draft
  forwards + their per-slot op chain are.**

## 2. Architecture reality (0.27.1 served tree — file map)

### 2.1 Which runner runs MTP today, and why

- `vllm/config/vllm.py` (`config/vllm.py` in image):
  `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (69-79) lacks Qwen3.5; `use_v2_model_runner`
  (578-629); `_is_dflash2_draft` (631-640) forces V2 for dflash; `_is_default_v2_...`
  (654-675) ends `return is_default_v2_architecture or not model_config.is_moe`.
- The load-bearing gate for this target is the **hybrid gate** (668-671):
  `if is_hybrid and (not is_default_v2_architecture): return False`.
  Qwen3.5-hybrid + not-in-default-set → **V1**. Arch absence alone is not the
  mechanism (dense non-MoE would return True); hybrid-ness is.
- Override exists: `VLLM_USE_V2_MODEL_RUNNER=1` forces V2 (579-581), and `mtp` is in
  the V2-supported method list (2214-2221), so validation would not reject it.
  Consequence today: the radiance V1 controller (see 2.2) would go silent, because
  V2 MTP is a different class hierarchy. That is a **porting task, not a blocker**.
- V1 drafter selection (`v1/worker/gpu_model_runner.py:595-650`): `method=mtp`
  (no `model` field, not gemma4/step3p5/dflash) → **`EagleProposer`**
  (`v1/spec_decode/eagle.py:10`, `use_eagle()` true for mtp). Native MTP reuses the
  Eagle/MTP proposer loop with the target's MTP layers as the draft model.

### 2.2 The V1 MTP hot path (the loop to kill)

- `SpecDecodeBaseProposer.propose` (`v1/spec_decode/llm_base_proposer.py:502-769`):
  first-pass forward (580-600), then **serial Python loop** (682):
  `for token_index in range(num_speculative_tokens - 1)`, each iteration:
  `_update_positions_dependent_metadata` (Triton, 771-821) → buffer copies
  (715-724) → `self.model(**model_kwargs)` (749) → `_sample_draft_tokens`
  → `_greedy_sample` → `model.compute_logits(hidden).argmax` (428-438).
- `EagleProposer.__init__` passes `pass_hidden_states_to_model=True`; the MTP
  layer consumes target hidden states + its own embed/fc/norm + **one decoder
  layer** + norm per slot (`model_executor/models/qwen3_5_mtp.py:139-177`).
- Sampling is greedy (`_sample_draft_tokens`, 468-487; `all_greedy` fast path).
  The radiance hooks patch `_greedy_sample` + `propose` (`radiance_draft.py:350-484`)
  and wrap `GPUModelRunner.propose_draft_token_ids` (488-536). The baked
  `patch_mtp_loopbreak.py` honors `_radiance_stop` at loop line 682-684.
- V1 drafter cudagraph: **PIECEWISE only**
  (`initialize_cudagraph_keys`, 411-426; `_determine_batch_execution_and_padding`,
  1799-1841). The loop body runs eagerly per slot today.
- Scheduling: for `use_eagle()` methods the drafter runs on GPU sampled tokens
  **before** `_bookkeeping_sync` (`gpu_model_runner.py:4636-4658`,
  `drafter_runs_model_forward`, `use_gpu_toks`). Drafting does NOT wait for the
  28 ms `parse_output` D2H. `draft_after_bookkeeping` stays False for mtp.

### 2.3 The V2 MTP path (the alternative vehicle — already graphed)

- `v1/worker/gpu/spec_decode/__init__.py`: `method=mtp` → `MTPSpeculator`
  (`mtp/speculator.py`, subclasses `AutoRegressiveSpeculator`).
- `autoregressive/speculator.py`: `capture()` graphs prefill (101-109) and the
  **full decode draft routine** `_generate_draft` (118-126); `propose()` runs
  prefill graph then `_multi_step_decode` (374-424) replaying the **FULL decode
  graph per step** (413-415); inputs updated **on device** by Triton
  (`update_draft_inputs`, 746-776; `prepare_decode_inputs`, 653-676).
  Sole `.item()` is on a CPU tensor (line 159). No per-slot host loop, no D2H.
- `model_runner.py` (V2) `sample_tokens` (1457-1600): GPU-only `sample()`
  (1143-1175), `AsyncOutput` on `output_copy_stream` (1527-1535, literal overlap
  comment), `speculator.propose()` inline (1572-1585), GPU-only `postprocess_sampled`
  (1177-1206).
- `multi_module_mtp/speculator.py` exists for multi-layer MTP (Qwen3.5 uses
  `num_nextn_predict_layers`-style depth via `use_multi_module_mtp()`).
- Upstream `main` still has the serial V1 loop (line 689) and the same
  PIECEWISE-only V1 rule — **no upstream release parallelizes the V1 MTP loop.**
  V2 AR loop is equally serial across steps (each step needs the prior token),
  but each step is **one graph replay**, not ~10 eager launches + Python.

### 2.4 True data dependencies (the only "impossible" list — everything else is fair game)

1. **Within-draft serial:** draft slot N+1 takes draft token N as `input_ids`
   (V1: `input_ids = draft_token_ids_list[-1]`, line 688; V2: `update_draft_inputs`
   writes sampled token into next step's inputs). Cannot be parallelized without
   changing the algorithm (masked/parallel drafting needs a retrained head).
   → Graph it, shrink it, skip it — do not try to run slots concurrently.
2. **Cross-step serial:** verify(N+1) takes draft(N) as `input_ids`
   (V2: `req_states.draft_tokens → draft_tokens_handler → take_draft_token_ids →
   scheduler`; V1: `propose_draft_token_ids → _copy_draft_token_ids_to_cpu →
   take_draft_token_ids`). Draft(N) must finish before verify(N+1) starts.
   → P0-as-specified is impossible. Legal overlaps only: sampled-token D2H vs
   draft forward (already ships both runners), host/scheduler work vs GPU,
   bookkeeping vs nothing (it needs verify output).
3. **Acceptance coupling:** fewer/cheaper forwards that lower acceptance can lose
   net t/s. Every track is gated on **tok/update + ms/step jointly**, never one
   number.
4. **Hardware (gfx1201/R9700):** ~640 GB/s DRAM roof; ~210 W cap; no usable
   cross-layer weight-prefetch intrinsic; Triton + ROCm available; single card
   (no DP/EP/DBO — DBO is a multi-GPU MoE-comm overlap, irrelevant here).
   Everything else — runner choice, graphs, kernels, scheduler, depth policy,
   head weights — is mutable.

## 3. Upstream check (what's new, what applies)

- Releases: 0.28.0 on PyPI (2026-08-26); 0.29.0/0.30.0 tags on GitHub. No release
  notes advertise MTP draft-loop parallelization or V1 full-graph drafting.
- **PR #35607** (`fix(qwen3.5-mtp): propagate spec_step_idx`, 2026-02-28):
  Qwen3.5 predictor cycles `layers[spec_step_idx % num_mtp_layers]` but the
  wrapper dropped the index → always `layers[0]` when `mtp_num_hidden_layers>1`.
  Served 0.27.1 `Qwen3_5MTP.forward` (line 281) still drops it → **backport
  candidate T1.** Draft-quality fix (acceptance up, ms/step flat).
- **Dynamic speculative decoding** (`num_speculative_tokens_per_batch_size`):
  batch-size schedule, V1 piecewise-only caveat, DP-incompatible. Weaker than the
  radiance controller; only its cross-step depth-adaptation idea (T5) is worth
  borrowing, driven by acceptance EMA instead of batch size.
- **PARD / parallel_drafting**: one-forward-K-tokens contract; needs a
  masked-slot-trained head (`pard_token`/`ptd_token_id`/`mask_token_id`). Native
  MTP head cannot do it untrained. Model-work track only (T6b).
- **DSpark / DFlash / EAGLE3.1 / speculators-lib MTP finetune**: DSpark is a new
  method (own speculator, own checkpoint) — not MTP acceleration; EAGLE3.1 is
  robustness, not speed; **speculators-lib MTP finetuning (FastMTP-style) is the
  applicable model track** (T6a).
- **Issue #34234 / PR #58356**: Eagle-prefill graphs (V2-only tracking) and
  ngram_gpu trim-loop `.item()` removal — neither is the V1 MTP loop. No free
  upgrade; custom work required.
- **V1 deprecation notice** (0.27.0 notes: "considering V1 deprecated, targeting
  v0.32 for removal"): strategic pressure toward V2 (T2). Do not invest in V1-only
  machinery that cannot be carried to V2; prefer the V2 vehicle long-term.

## 4. Solution tracks (all authorized: rebuild, rewrite, new files, model work)

### T0 — Baseline + instrumentation (1 boot, no rebuild)

- Fix the measurement basis: `SPEC_METHOD=mtp`, SPEC=8, MAXSEQS=8, target ctx;
  `RADIANCE_DRAFT_PHASE_TIMERS=1` + `RADIANCE_STEP_TRACE` + per-slot timers
  inside `_local_draft` (head / capture / argmax / d2h / decide) and around each
  loop-body line (metadata / copy / forward / sample). Record ms/step, tok/upd,
  acc/draft per position 1-8, acceptance by category, GPU busy + clocks.
- Deliverable: slot-cost table that picks T3 vs T4 first. Do not skip — B1/B3
  history shows intuition without timers misfires.

### T1 — `spec_step_idx` multi-layer fix: DEAD for this checkpoint (M0 answered)

- **M0 result (disk, 2026-09-27):** blend target
  `~/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ/config.json` →
  `text_config.mtp_num_hidden_layers = 1`; safetensors index contains exactly
  `mtp.layers.0.*` (15 `mtp.*` tensors, no `layers.1`). Single MTP layer.
- Consequence: `spec_step_idx % 1 == 0` always; PR #35607's bug cannot fire.
  Served 0.27.1 and upstream `main` **both** still drop `spec_step_idx` in the
  `Qwen3_5MTP.forward` wrapper (verified both files) — worth a 3-line hygiene
  backport on the next rebuild so a future multi-layer checkpoint is correct from
  day one, but it buys **zero** on this checkpoint. T1 closed, no further work.
- Related: upstream PR #55369 (n_predict from `text_config` for multimodal MTP)
  postdates 0.27.1's `config/speculative.py` shape only cosmetically; served tree
  resolves n_predict per-family without the text_config fallback in the same spot.
  No action unless a future checkpoint mis-resolves depth.

### T2 — Move MTP to the V2 vehicle (structural win; controller gets ported)

Why: V2's draft path already captures each step as a graph replay with
device-side input updates and AsyncOutput D2H overlap — but see correction C1:
served 0.27.1 V2 keeps a **host loop around per-step replays** (one graph per
slot, not one graph per sequence). The win is launch-overhead-per-step plus
device-side input updates, not loop elimination. Upstream `main` adds the fused
option (C2). The plan's TRACE rows (MTP host chain 52 ms vs dflash host chain
6 ms) compare different draft workloads on different runners — do not attribute
the gap to the runner alone without a matched V1-mtp vs V2-mtp measurement.

No-GPU static pre-work (done, from 0.27.1 sources):

- **V2 MTP draft entry:** `v1/worker/gpu/model_runner.py:202-207`
  (`init_speculator` when `num_speculative_tokens>0`), `:321-346` (load draft
  model through `load_eagle_model`, i.e. embeds + lm_head shared with target —
  the int2 head work in `radiance_drafthead.py` ports to the shared head),
  `:516-528` (draft attention groups + `init_cudagraph_manager`), `:783-784`
  (`capture()`), `:1572-1586` (`propose()` inline on the main stream after the
  AsyncOutput is created), `:1588-1603`
  (`draft_tokens_handler.set_draft_tokens` → `take_draft_token_ids`).
- **Draft token custody on V2:** `v1/worker/gpu/spec_decode/utils.py`
  `DraftTokensHandler` — `set_draft_tokens(input_batch, draft_tokens)` stores
  req_ids + **one batch-wide width** (`draft_tokens.shape[1]`); without
  structured output it returns `[-1]` placeholder rows and the real tokens ride
  `req_states.draft_tokens` → scheduler `update_draft_token_ids` →
  `request.spec_token_ids` → next step's `scheduled_spec_decode_tokens`
  (`scheduler.py:640-656,1070-1085,2146-2200`) → V2 runner stages
  `num_draft_tokens_per_req / cu_num_logits / expanded_idx_mapping`
  (`model_runner.py:950-1000`). Correction C3: the ragged *consumers* exist
  (`update_draft_token_ids` accepts per-request lists; runner derives lengths
  from scheduled list lengths), but **no producer path carries controller-chosen
  per-request lengths** — the handler accepts no length input. The port must add
  an explicit `[B]` valid-length result (T2c); it does not "slot into existing
  fields".
- **V2 propose signature** (`autoregressive/speculator.py:129-155`): receives
  `input_batch` (with `idx_mapping/idx_mapping_np`, `seq_lens_cpu_upper_bound`,
  `num_scheduled_tokens`), `num_sampled/num_rejected` (GPU), `last_sampled`,
  `temperature/seeds` (GPU). Everything the V1 controller reads from its own
  runner (`token_ids_cpu_tensor`, `num_tokens_no_spec`, `sampling_states`) has a
  V2 counterpart: matcher context comes from `req_states.all_token_ids.gpu`
  (device mirror, no new host copy) + `num_computed_tokens`; policy inputs that
  need sampling params come from `sampler.sampling_states` (same object family
  the verify-head gate already reads on V1).
- **Sampling on V2:** `speculator.sample_draft` (speculator.py:308-333) greedy
  path is `_greedy_sample_draft` → `model.compute_logits(hidden).argmax`
  (or `model.get_top_tokens` under `use_local_argmax_reduction`, which
  `Qwen3_5MTP` provides via `LocalArgmaxMixin`). The int2 head replaces
  `_apply_head` under `compute_logits` — same one-line binding point as V1,
  then `sample_draft` needs no change.
- **What does NOT port:** `SpecDecodeBaseProposer._greedy_sample/propose`
  hooks, `_radiance_pack_cpu` via `input_batch.token_ids_cpu_tensor`,
  `_prepare_match_gpu`'s CPU-tensor mirror logic, `_postprocess_gpu`'s
  ragged-list return (V2 returns a dense `[max_reqs, K]` tensor;
  `draft_tokens[:num_reqs]`). All four get rewritten against the bullets above,
  not patched. Plus correction C4: matcher context must be rebuilt from
  **committed history + this iteration's valid sampled tokens** (V1 padded path
  drafts before bookkeeping, so the CPU mirror can be one step stale;
  `gpu_model_runner.py:4641-4658` vs `:4698-4701,4733-4737`), not from a
  presumed-current mirror — on V2 from `req_states.all_token_ids.gpu` with an
  explicit valid-history bound, never assumed interchangeable counters.

- **Sub-tracks (explicit, replacing the single "port the controller" bullet):**
  - **T2a. Boot + enumerate.** `VLLM_USE_V2_MODEL_RUNNER=1` + `method=mtp`;
    record `_validate_v2_model_runner` / attention-backend / multimodal /
    hybrid-KV verdicts. Pull first: #55390, #56709, #56734, #58368.
  - **T2b. Fixed-capacity proposal contract.** Keep `[B, Kmax]` tokens, add an
    explicit `[B]` valid-length result; separate max forwards from final draft
    length (n-gram tails make outputs longer than forward count).
  - **T2c. Synchronous length bridge.** Extend `DraftTokensHandler` + caller to
    carry lengths with request IDs; ordinary requests return placeholder rows of
    length L[i] (no full token D2H); structured-output requests return actual
    prefixes before grammar validation. Sync scheduling only — async installs
    batch-wide placeholder lengths and skips the draft update
    (`async_scheduler.py:25-50`, `engine/core.py:664-671`).
  - **T2d. Hybrid rollback.** Preserve GDN recurrent-state selection
    (`fused_sigmoid_gating.py:103-120,156-166`) + conv window
    (`causal_conv1d.py:862-888`) across the new zero/variable-length
    transitions; cover 0-draft-after-N-accepted explicitly (non-spec path loses
    the acceptance convention). Do not ship variable lengths before this proves
    across cache modes, block boundaries, preemption.
  - **T2e. Controller on the new contract.** Subclass/wrap `MTPSpeculator`;
    confidence/state writes in captured device ops, host stop decisions between
    replays, persistent device buffers (init/resets outside capture).
- **Multi-module check:** N/A for this checkpoint (1 layer). Skip unless a future
  checkpoint has depth >1.
- **Gate:** V2-mtp ms/step vs V1-mtp at matched acceptance; controller ON vs OFF
  on V2. Expectation: host chain collapses toward single-digit ms; serial
  forwards remain (physics) but each is one replay. If V2-mtp boots clean, **all
  further tracks build on V2** and V1 tracks become fallback.

### T3 — Graph the draft loop body (the correct "P1", either runner)

Static per-slot launch audit (0.27.1, batch-1 decode — what a graph removes).
One V1 loop iteration (proposer lines 682-760) issues, in order:

1. `eagle_step_update_slot_mapping_and_metadata` Triton (788-798) — positions,
   slot mapping, seq_lens bump. Always runs (`constant_draft_positions=False`).
2. `build_per_group_and_layer_attn_metadata` → per-group
   `build_for_drafting(draft_index=...)` (707-712, via 995-1008). For the GDN
   hybrid target this rebuilds full + linear metadata objects per slot in Python.
3. 2-3 small buffer copies (`input_ids`/`hidden_states`, 715-716, int cast 688).
4. MTP forward (749 → `Qwen3_5MultiTokenPredictor.forward`): embed lookup +
   2 RMSNorms + fc concat-project + **one full decoder layer** (input norm,
   GQA full attention incl. KV-cache save + paged-attention kernel(s),
   post-attn norm, SwiGLU MLP gate/up/down) + final norm. ~10-20 kernels, each
   with Python launch overhead; attention carries the only seq-len-dependent
   work (append 1 KV per slot).
5. Head: int2 coarse Triton (`nblk≈3880` programs at N=248320) + `bm.topk(32)`
   + `_rerank_exact` Triton + `scatter_` (radiance_drafthead.py:309-333).
6. `_greedy_sample` argmax path → `compute_logits(hidden).argmax` (428-438);
   radiance `_local_draft` adds `_cap_s1` (B×64 programs) + `_cap_s2_local` +
   `local.argmax(-1)` (radiance_draft.py:275-283, radiance_draft_gpu.py:293-307).
7. Controller D2H: one coalesced `packed.cpu().numpy()` (2×B fp32) + numpy
   `slot_decide` (~µs) (radiance_draft.py:389-419).
8. `set_forward_context` enter/exit per iteration (741-748) — dispatcher +
   P2P state Python per slot.

V1: capture one loop iteration (MTP layer + head + sample + metadata update)
  as a breakable/PIECEWISE graph replayed per slot at fixed B (pad to
  {1,2,4,8}); keep `_radiance_stop` between replays. Precedents:
  `BreakableCUDAGraphWrapper` (propose:528), eagle dispatcher, unpad patch.
  Hard parts, named: the MTP forward writes KV-cache slots (slot mappings must
  be graph inputs, not Python-rebuilt — feed `_slot_mapping_buffer` as a graph
  tensor and update positions/slot mapping with the existing Triton update
  kernel *inside* the graph); the controller D2H cannot live inside a graph —
  replay graph-per-slot with the host gate between replays (keeps graphs valid,
  keeps the loopbreak).
- V2: already per-step graphed — validate `decode_cudagraph_manager` actually
  replays (FULL) for the mtp shapes served here (uniform decode batches,
  attention backend FULL-compatible); fix dispatch/capture sizes if it falls
  back to eager. Then decide per C2: per-slot graphs keep immediate stopping;
  upstream-`main`-style fused multi-step capture removes host replay overhead
  but ordinarily executes full captured depth — a depth-variant bank or a
  chunked (1/2/4-step) granularity (T8) preserves stopping benefits. Do NOT
  mutate `num_speculative_steps` after capture; keep Kmax immutable, runtime cap
  and valid lengths separate.
- Removes ~7× Python + launch overhead per step **inside each captured step**;
  does not remove serial forwards or (served tree) the host replay loop.
  Effort high, needs rebuild + capture-validation runs.

### T4 — Shrink each draft forward (complements T3; kernels, not policy)

- **T4a. Finish the fused head (contract repair first, §8 C5):**
  `_draft_head_int2_top1` already omits the `Y` store — the remaining work is
  reductions, selection, rerank, intermediates, and interface correctness, NOT
  "removing the store". Required before any fusion claim: (i) keep
  `_apply_head → logits` for all stock/fallback/V2 paths and add a distinct
  proposer-facing `draft_top1_into(hidden, workspace, ids_out, conf_out)` used
  only for supported greedy drafting (TOP1-as-tuple breaks controller-off,
  local-failure fallback, probabilistic, and V2 sampling paths); (ii) fix the
  confidence contract — current TOP1 mixes an exact numerator with an uncorrected
  coarse denominator, can exceed 1, includes padding, and breaks the cumulative
  product — choose calibrated-approximate (retune thresholds) or exact-hybrid
  (replace selected coarse contributions, exclude padding); (iii) fix the
  missing-rerank fallback control flow (coarse computation gated behind
  `elif not warned` leaves `ids/conf` unbound after the first warning).
  Gate on acceptance (recall), not equivalence.
- **T4b. Metadata fast path (reuse, never skip):** positions, seq_lens, and
  physical KV slots advance **every slot** (`_update_positions_dependent_metadata`,
  771-821; rejection adjustment is separate, 670-678) — skipping is unsafe
  across block boundaries. The valid optimization reuses metadata *objects*
  while updating dynamic tensor contents + scalar bounds, or fuses the update
  into another device op.
- **T4c. Draft attention/KV:** verify the MTP draft path uses the cheapest
  compatible backend at M≤8 and that its KV-cache groups don't force extra
  metadata builds per slot; borrow `patch_gdn_metadata`-style per-step Python
  cuts for the draft attention path. (Do not assume hybrid-target ⇒ GDN draft
  metadata: proposer iterates actual `draft_attn_groups`; served MTP layers are
  `full_attention`.)
- Each item is ~10-100 us/slot **(unmeasured estimate, not a finding)**;
  together ~0.5-2 ms/step. Medium effort (Triton).

### T5 — Run fewer draft forwards (controller extension + cross-step depth)

- The loopbreak already stops at the batch max. Two staged refactors precede
  any new policy (§8 C6): **(i) pre/post-head policy split** — strong/recency
  and cap decisions that need no current-slot head move before the forward
  (`need_head` / `take_tail` / `stop`), preserving candidate-1-over-2 priority
  and prior-cum-then-multiply semantics; per-row compute avoidance needs real
  compaction, stopping rows alone doesn't remove dense forwards. **(ii) First-pass
  KV maintenance** — an all-strong batch may skip head sampling + later forwards
  but not the first pass (KV sync, proposer 607-616); n-gram-only output must not
  depend on an MTP placeholder row.
- Then add **cross-step depth adaptation**: promote the controller's stop
  distribution / acceptance EMA into next step's `num_speculative_tokens`
  (V1: proposer arg; V2: `num_speculative_steps` per batch — immutable after
  capture, so Kmax stays fixed and only the runtime cap + valid lengths vary)
  via the existing knob-file/`_batch_ceil` machinery. This is dynamic SD driven
  by acceptance, not batch size. Explicitly route probabilistic sampling (bypasses
  `_greedy_sample`, controller never runs there) and initialize per-call state
  before every branch (current skip path leaks `_radiance_stop`/gate state).
- Keep the n-gram tail fill (no extra MTP forwards — but matching, sync,
  assembly, verification, and cache maintenance are still real costs, not "free").
  Consider matcher stride only if T0 timers say so (M1 row: ~1-3 ms).
- Gate on tok/update jointly with ms/step; reject on code/json acceptance loss.

### T6 — Raise acceptance per forward (model work — the tail problem)

- Plan tail (positions 6-8: 0.22/0.18/0.15) is head quality, not plumbing.
- **T6a. Rollout-aware native MTP retrain (methodology borrowed from HASS /
  EAGLE-3 "training-time test", applied to this head).** Freeze target, init
  from the existing one-layer MTP, train short recursive unrolls on
  MTP-generated (not only teacher-forced) hidden states, match inference
  cache/context construction, distill against the *served quantized* target on
  representative code/tool/reasoning/chat mix, keep shared embed/lm_head frozen,
  recalibrate controller thresholds after. Sequential teacher-data generation +
  MTP-only training fits a one-card workflow (microbatch/offline features);
  full-teacher-resident training fit is NOT assumed. Raises tok/update at fixed
  ms/step. Separate training project; serving change is a checkpoint swap.
- **T6b. Parallel/native-single-pass head (only if T6a happens):** a PARD-style
  masked-slot adaptation of the MTP head would collapse 7 forwards to 1. Needs
  training + `parallel_drafting` plumbing. Listed for completeness; do not start
  before T1-T5 land.
- **T6c. Sampling:** `draft_sample_method` probabilistic (shared-Gumbel coupling)
  accepts with sum(min(p,q)) vs p(argmax) — measure acceptance vs full-logits
  cost; currently bypasses the int2 fast path, so it needs the sparse
  `draft_logits` integration to be viable. Experiment, default stays greedy.

### T7 — System-level (verify forward + scheduler — shared with every method)

- Verify forward is the ~22 ms floor **(roofline subtraction is a bound, not a
  measurement — draft-layer/head traffic is extra weight movement on top)**:
  decode-GEMM geometry at M=9 (SPEC=8, batch 1), attention backend at long ctx,
  `RADIANCE_VERIFY_HEAD` gating (already -8% under dflash; re-gate for mtp
  sampling params), KV-cache sizing at MAXLEN.
- Scheduler/host: `disable_padded_drafter_batch=true` stays (the +50% lever —
  but note it selects the **after-bookkeeping** draft branch,
  `gpu_model_runner.py:4698-4701,4733-4737`, so "MTP always drafts before
  bookkeeping" was unqualified), sync scheduling (async measured identical),
  chunk/prefill settings that keep decode batches uniform (graph-friendly).
- These help MTP and everything else; they do not fix the draft loop, so they
  follow T1-T5, not lead.

### T8 — New-methodology candidates (research-grade, ordered by evidence fit)

All preserve the native single-layer MTP; none switches to DFlash. Status labels:
established mechanism vs proposed integration. Metric for all: **cost per
committed token** = total decode wall / committed tokens, with drafting,
verification, commit/replay, controller, wasted work, and per-position acceptance
recorded separately.

- **T8a. Controller-aware chunked graphs (proposed integration, high fit).**
  Capture 1/2/4-step MTP chunks: `MTP chunk → device controller update →
  continue/verify`. Preserves stopping benefits while removing per-slot host
  dispatch inside chunks. Constraints: fixed-address buffers, keyed variants,
  padded/inactive rows must not touch KV/conv/recurrent state, CPU stop-flag
  reads reintroduce sync (depth-variant bank from prior-round info avoids
  within-round host decisions). Do not assume HIP == CUDA conditional-graph
  support in the installed runtime.
- **T8b. Native chain + one root alternative (recent experimental evidence,
  high fit, substantial verifier work).** GDN Tree-Scan (arXiv:2609.23900, Sep
  2026 — 64-layer Qwen3.6-27B hybrid w/ native MTP, closest architecture found):
  keep the native MTP chain, add ONE runner-up candidate at draft position 1
  from the same logits tensor (a leaf — no second recursive trajectory); on win,
  its verify row seeds continuation. Reported +27% token-weighted decode t/s
  (batch-1 GB10, 4 SWE/Codex tasks, temp 0.6) — design evidence, NOT a gfx1201
  prediction; equivalence evidence scoped to rescore-closure, not a full
  distribution proof. Implementation requires branch-local everything: ancestry
  mask PLUS parent recurrent state, branch-local conv history, path-depth
  positions, accepted-node (not just length) commit, accepted-path-only state
  publication. Start fixed-topology + exact-order reference before any compact
  factorization.
- **T8c. Compact GDN commit (recent papers, pursue when state costs justify).**
  Bole (arXiv:2608.01651) tree-WY factorization, TreeWY (arXiv:2608.20961 —
  same memory at identical acceptance, wider trees affordable but **not yet a
  throughput win**; tree path lost full graphs in that stack), ReplaySSM
  (dao-lab.ai/blog/2026/replayssm, PR #49887 — compact recent inputs/corrections;
  best at larger batches). WY/chunked transforms change rounding — needs
  long-horizon logit + state validation, not one-step tolerance.
- **T8d. Joint block verification (established algorithm, conditional).**
  ICLR 2025 block-verification changes acceptance/correction to verify sub-blocks
  jointly, distribution-preserving. Needs genuine stochastic proposals + correct
  proposal law + residual sampling; content-dependent truncation needs its own
  proof; no help for greedy serving or GDN rollback. Move earlier only if
  stochastic sampling dominates.
- **T8e. Bounded optimistic continuation (established methodology, experimental
  here).** PEARL/PipeSpec/Saguaro-style: draft ahead while verifying, reuse only
  on outcome-prefix hit, isolate rejected state. Native MTP reseeds from *target*
  hidden states — continuing from predicted MTP state is a different algorithm
  (hidden-state mismatch + bonus-token gap), and one gfx1201 shares bandwidth
  between contender streams (Saguaro uses separate hardware; SPD warns 1-GPU can
  be slower). Bounded experiment AFTER T6a, never an assumed win.
- **T8f. Retrieval-augmented budget fill (Graft-style adaptation).** When the
  controller stops an unreliable MTP continuation, spend part of the remaining
  candidate budget on matching n-gram continuations as *proposals* (not just stop
  signals). Mixed MTP/retrieval candidates need a correct multi-proposal
  acceptance rule; first version uses committed history only.

## 5. Sequencing and stop rules

1. **T0** (timers) → **M0 DONE (T1 dead: 1 MTP layer).**
2. **T2 spike** (T2a boot + T2b-e contract design). If V2 boots: V2 is the
   vehicle; T3-V2 + T4 + T5 build on it. If blocked on >~2 patches: V1 T3+T4
   fallback, revisit V2 after.
3. **Head-contract repair + T3 → T4 → T5** (§8 C5-C6 order: contracts before
   graphs), each gated end-to-end (ms/step + tok/upd + acc/draft +
   combined t/s, both thinking modes, card 1, warmup 3 + 20 passes/category min,
   full BetterBench before landing).
4. **T6a** in parallel once acceptance-by-position data exists (training lead
   time; needs no T1/T2 outcome — head is single-layer either way).
5. **T8** candidates after T2-T5 baseline exists (T8b needs a correct verifier;
   T8e only after T6a). Cost-per-committed-token metric throughout.
6. **Stop rule:** if T2/T3 combined move step <5% at matched acceptance, the
   serial-forward floor is structural for this checkpoint — further effort goes
  to T6 (head quality) or method choice, not more overlap schemes.

## 6. Open questions

- ~~Blend target depth~~ ANSWERED (disk): 1 MTP layer (T1 dead).
- V2-mtp boot blockers (attention FULL-graph compat, KV groups, mm inputs) —
  PARTIALLY ANSWERED statically: custody/length/sampling surfaces mapped
  (§T2); actual ROCm backend selection, capture success, multimodal/MXFP4
  execution, rollback across cache modes/block boundaries/preemption, and graph
  hit rates need the T2a boot.
- Per-slot split (metadata vs forward vs head vs sample) — STATIC PREDICTION in
  §T3 (metadata + launch overhead dominate; head ~0.4 ms; controller <0.1 ms),
  needs T0 timers to confirm; `_phase` wall times absorb queued GPU work, so T0
  must separate host-submit, GPU-execution, sync-wait, and end-to-end.
- Acceptance by position 1-8 per category — needs serving measurement (sizes
  T5/T6/T8b payoff).
- Fused multi-step eligibility for this checkpoint's draft attention groups —
  ANSWERED conditionally: MTP layers are `full_attention`; Triton declares
  `supports_draft_decode_metadata_update` (no-op updater) — but the served
  0.27.1 tree predates the fused path (single-step capture only,
  `speculator_ar.py:114-126`); fused needs the `main`-era tree or a backport.
- Matcher-context staleness on the padded path — ANSWERED statically (C4):
  padded drafts run pre-bookkeeping off a CPU mirror missing current sampled
  tokens; fix is committed-history + valid-sampled-tokens construction.
- TOP1 confidence semantics — ANSWERED statically (C5): current formula is
  neither standard-softmax nor calibrated; needs contract choice + fallback fix
  before thresholds mean anything.

## 8. Static corrections (supersede §§2-4 claims as noted)

Recorded 2026-09-27 from the three-way audit + primary-source verification.
Evidence: served 0.27.1 tree (`/tmp/kilo/vllm-src`), upstream `main`
(`/tmp/kilo/vllm-main` @ `231fdb8`), repo sources, arXiv:2609.23900 /
arXiv:2608.20961 abstracts + histories.

- **C1. Served V2 keeps a host loop.** `speculator_ar.py:114-126` captures ONE
  `_generate_draft` step; `_multi_step_decode` (374-424) loops in Python with
  per-step slot-map/metadata prep + replay-or-eager. §§2.3/T2 claims of "no
  per-slot host loop / entire chain eliminated" are overstated — V2 removes
  in-step launch overhead + input updates, not the replay loop.
- **C2. Upstream `main` adds fused multi-step decode.** `_configure_fused_
  multi_step_decode` (97-121, backend-gated on
  `supports_draft_decode_metadata_update`), `_generate_fused_drafts` (598-638,
  serial loop inside capture), `_fused_multi_step_decode` (551-596, one FULL
  replay). Served 0.27.1 predates it. Fused ≠ parallel: forwards stay serial;
  full-depth execution vs early-stop is the tradeoff (depth-variant bank or T8a
  chunks preserve stopping).
- **C3. Length bridge is missing, not "already plumbed".** `DraftTokensHandler`
  (verified in-image) records one batch width, takes no length input, returns
  `[-1]` rows off structured-output. Ragged *consumers* exist
  (`update_draft_token_ids`, `num_draft_tokens_per_req` derivation,
  `combine_sampled_and_draft_tokens`); the controller→scheduler producer path
  does not. T2b/c is required work, async scheduling excluded from the first
  port.
- **C4. Matcher context can be stale.** Padded V1 drafts pre-bookkeeping while
  the CPU mirror updates inside bookkeeping — history may miss current sampled
  tokens (plus async placeholder `-1`s). Rebuild context from committed history
  + valid sampled tokens with explicit bounds on both runners.
- **C5. TOP1 needs contract repair.** Tuple-instead-of-logits breaks
  controller-off/fallback/probabilistic/V2 paths; confidence mixes exact
  numerator with coarse denominator (can exceed 1, includes padding, breaks cum
  product); argmax equivalence unguaranteed (unrescored coarse can win in Y);
  missing-rerank fallback leaves `ids/conf` unbound after first warning
  (`radiance_drafthead.py:414-430`). And: `_draft_head_int2_top1` already omits
  the `Y` store — "remove the store" was never the remaining work.
- **C6. Metadata updates cannot be skipped.** Positions/seq_lens/physical slots
  advance every slot incl. no-rejection uniform batches; blind slot+1 is unsafe
  across block tables. Reuse objects + update contents, or fuse the update —
  never omit semantics. Pre/post-head policy split and first-pass KV maintenance
  (§T5) are the correct "fewer forwards" refactors, with priority/compaction
  caveats as written.
- **C7. Timings are wall, not cost.** `_phase` = `perf_counter` around launches;
  blocking `.cpu()` absorbs earlier queued GPU work; "slack" is nested-interval
  subtraction. T0 must split host-submit / GPU-execution / sync-wait / e2e, or
  every downstream priority is guesswork. Roofline subtraction likewise doesn't
  isolate "non-weight overhead" (draft traffic is extra weight movement).
- **C8. "Free" n-gram tail isn't free; outputs-can-change needs care.**
  Matching, sync, assembly, verification, cache maintenance are real. Rejection
  sampling preserves the distribution; fixed-seed byte-identity is a separate,
  stronger claim — keep acceptance-gated language exact.

## Appendix — key files (0.27.1 served tree)

- Runner selection: `vllm/config/vllm.py:69-79,578-640,654-675,2183-2281`;
  `v1/worker/gpu_worker.py:174,384-415`.
- V1 step: `v1/worker/gpu_model_runner.py:3692-3721 (_sample),3723 (_bookkeeping_sync),
  4553-4812 (sample_tokens),5010+ (propose_draft_token_ids),595-650 (drafter select)`.
- V1 loop: `v1/spec_decode/llm_base_proposer.py:428-502 (sample),502-769 (propose),
  682 (loop),771-821 (metadata),1799-1841 (padding/dispatch),411-426 (piecewise keys)`.
- V2 step: `v1/worker/gpu/model_runner.py:1143-1175 (sample),1177-1206
  (postprocess),1457-1600 (sample_tokens),1527-1535 (AsyncOutput overlap),
  1572-1585 (propose inline)`.
- V2 draft: `v1/worker/gpu/spec_decode/autoregressive/speculator.py:84-126
  (capture),128-274 (propose),374-424 (multi-step decode),746-776 (device-side
  input update)`; `mtp/speculator.py`; `multi_module_mtp/speculator.py`.
- MTP model: `model_executor/models/qwen3_5_mtp.py:80,139-177 (predictor+spec_step_idx),
  281-300 (wrapper drops it)`.
- Radiance: `radiance_draft.py:233-298 (_local_draft),350-484 (hooks),488-536
  (runner wrap)`; `radiance_draft_gpu.py` (capture/matcher kernels);
  `radiance_drafthead.py` (int2 head); `patch_mtp_loopbreak.py`, `patch_mtp_mm_mask.py`.
- V2 port surface: runner `model_runner.py:202-207,321-346,516-528,783-784,
  1572-1603`; `spec_decode/utils.py DraftTokensHandler`;
  `autoregressive/speculator.py:129-155,374-424,746-776`;
  `speculator.py:302-356 (sample_draft)`; scheduler
  `scheduler.py:640-656,2146-2200`, runner staging `model_runner.py:950-1000`.
- Upstream refs: PR #35607 (spec_step_idx — N/A single-layer, hygiene only),
  #55369 (n_predict text_config), #55390/#56709/#56734/#58368 (MTP KV/prefix
  fixes to pull), #57312 (MTP fast-start daemon), #58065 (async DFlash),
  #57396 (hetero-vocab sync removal), #34234 (eagle prefill graphs, V2-only),
  DBO design (`docs/design/dbo.md` — MoE-comm only),
  dynamic SD (`num_speculative_tokens_per_batch_size`), `vllm-project/speculators`
  MTP finetune support.

## 7. No-GPU session notes (2026-09-27, GPUs busy with quantization)

- M0 answered from disk: `mtp_num_hidden_layers=1`, `mtp.layers.0` only → T1 dead.
- No boots run. Next GPU window: (a) T0 timer boot (1 boot, no rebuild —
  `SPEC_METHOD=mtp RADIANCE_DRAFT_PHASE_TIMERS=1 RADIANCE_STEP_TRACE=20
  bench-quick.sh`), then (b) T2 spike
  (`VLLM_USE_V2_MODEL_RUNNER=1 SPEC_METHOD=mtp`, expect config/attn/mm blockers,
  enumerate don't fix). Both commands ready to run when a card frees.
