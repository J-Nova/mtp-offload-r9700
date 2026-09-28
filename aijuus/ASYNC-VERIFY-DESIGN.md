# Async verification pipeline (P0) — design investigation

Status: **not viable in the active architecture.** Recorded 2026-09-27 after reading the
installed vLLM 0.27.1 sources in `stilldeadcode/vllm-radiance:0.9.3`.

> **Correction (2026-09-27, later same day).** The first version of this document said
> "the active path is V2". That is only true for **dflash**. Runner selection is
> per-request and depends on the spec method:
>
> - **dflash + DFlash2 draft -> V2** (`vllm.py:631 _is_dflash2_draft()` forces it;
>   `_is_default_v2_model_runner_model()` also gates it). The radiance controller and
>   the int2 head's `_local_draft` are **inert** here.
> - **mtp -> V1.** `Qwen3_5ForConditionalGeneration` is **not** in
>   `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`vllm.py:69-79` lists only DeepseekV2,
>   GraniteMoe, Inkling, KimiK3, LongcatFlashNgram, Qwen2Moe), so
>   `_is_default_v2_model_runner_model()` is False and `use_v2_model_runner` returns
>   False (`vllm.py:611-612`). MTP therefore runs **V1**, where
>   `radiance_draft._install_drafter_hooks()` (patches `SpecDecodeBaseProposer`) and
>   B2's fused path (inside `radiance_draft._local_draft`) **are** live, and the plan's
>   V1 measurements are valid.
>
> So "async verify" is not viable for **dflash** (V2 path, no host loop, and the
> cross-step dependency still holds), and for **mtp** (V1) it is not viable because the
> data dependency blocks cross-step overlap; the stubs were never a working
> implementation. The withdrawal of P0/P1/PD/WP stands on the never-applied / stub
> grounds, not because V1 is unused.

## Verdict

The hypothesis behind P0 — "the MTP draft loop is host-bound by a per-slot D2H
readback, so overlap the draft loop of step N with the verify forward of step N+1" —
is partly true on the **V1** mtp path (that host loop exists and is the plan's
diagnosis) but the proposed fix is unsound: verify N+1 depends on draft N, so they
cannot overlap. On the **V2** dflash path the host loop does not exist at all:
drafting is already CUDA-graphed with device-side input updates, and the sampled-token
D2H is already on a separate stream.

No patch is written, because no correct implementation of the stated design exists.

## The active path (evidence)

Runner selection, `vllm/v1/worker/gpu_worker.py`:

- `self.use_v2_model_runner = vllm_config.use_v2_model_runner` (line 174)
- `if self.use_v2_model_runner: logger.info_once("Using V2 Model Runner")` (384-385)
  — the server log prints exactly this line, so V2 is active.

Decode step, `vllm/v1/worker/gpu/model_runner.py`, `sample_tokens()`:

- `sampler_output, num_sampled, num_rejected = self.sample(...)` (1495)
- `sample()` is GPU-only: `compute_logits` then the GPU rejection sampler, no host
  sync (1143-1175).
- `async_output = AsyncOutput(..., main_stream=self.main_stream,
  copy_stream=self.output_copy_stream, ...)` (1527-1535) — the sampled-token copy to
  host is **already** on a dedicated stream, explicitly so it overlaps the speculator
  (comment at 1527).
- `draft_tokens = self.speculator.propose(...)` runs inline on the main stream
  (1562-1585); `postprocess_sampled` is GPU-only (1177-1206).

Speculator, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`:

- `capture()` CUDA-graphs the per-step draft routine `_generate_draft` via
  `decode_cudagraph_manager` (111-126).
- `propose()` runs the draft prefill (graphed) then `_multi_step_decode`, which loops
  `for step in range(1, num_speculative_steps)` and replays the **FULL decode graph**
  each step (`run_fullgraph`, 388-424).
- Inputs for the next step are updated **on device** (`_generate_draft` ->
  `update_draft_inputs`, 456-478; Triton kernels from 481 onward). There is no
  per-step host readback.
- The only `.item()` is on `seq_lens_cpu_upper_bound` (159), a **CPU** tensor — no GPU
  sync.

`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`: no host syncs.
`vllm/v1/worker/gpu/spec_decode/speculator.py`: base classes carry no syncs.

## Why the original hypothesis is wrong

- The `_bookkeeping_sync` (28.4 ms/step) and `d2h 1.82 ms/slot` figures come from the
  **V1** files `vllm/v1/worker/gpu_model_runner.py` (`_bookkeeping_sync` at 3723,
  `sample_tokens` at 4553) and `vllm/v1/spec_decode/llm_base_proposer.py` (the V1
  proposer). Neither module executes under V2. The `P0`/`P1` patches were written
  against those V1 shapes — and even then, three of P0's four anchors match 0x in
  0.27.1.
- The proposed overlap is impossible by construction: the verify forward of step N+1
  takes the draft tokens of step N as its `input_ids`
  (`speculator.propose` -> `req_states.draft_tokens` -> `draft_tokens_handler`, then
  the scheduler schedules them). Verify N+1 cannot start before draft N finishes.
- The one overlap that is physically available — sampled-token D2H vs. the draft
  forward — already ships (`output_copy_stream`).

## Relationship to P1/WP/PD

- P1 ("CUDA graph the serial MTP loop"): **already upstream on the V2/dflash path**
  (autoregressive/speculator.py:111-126, 413-424). On the **V1/mtp** path it is *not*
  upstream, and `llm_base_proposer.py` was the right file -- but the shipped P1 patch
  applied only its `__init__` (shared-sentinel bug) and its `_propose_cudagraph` body
  was a stub that called `self.propose()`, so it did nothing. A real P1 for V1/mtp
  would need to capture the radiance controller's per-slot chain, not just the
  proposer loop.
- PD ("parallel drafting") is **already upstream** on the V2 speculator:
  `parallel_drafting` in `llm_base_proposer.py` and the dflash/dflash2 speculators draft
  all slots in one forward.
- WP has no cross-layer weight handle in the dispatch layer and gfx1201 exposes no
  usable prefetch intrinsic (the `__builtin_amdgcn_prefetch` build failed).

## If real decode-latency work is wanted

1. **Confirm which runner production is on.** If `VLLM_USE_V2_MODEL_RUNNER=0` is ever
   served, the V1 serial loop becomes live again — but the plan's V1 numbers are old
   and would need re-baselining before any optimization is justified.
2. **Measure before hypothesizing.** `RADIANCE_DRAFT_PHASE_TIMERS=1` exists for the
   V1-style path; the V2 runner exposes its own step counters. Establish where the
   decode step actually spends time under V2 before proposing a change.
3. **B2 (int2 top-1 head)** is the only already-implemented decode optimization still
   A/B-pending; it targets the draft-head op chain that the plan's PHASE row
   identified as the real per-step cost.

## Recommendation

Do not implement P0 as specified. The mechanism does not exist in the active runner,
and the cross-step dependency makes the stated overlap unsound even in principle.
