# Adaptive KV Streaming + MTP roadmap (llama.cpp V2) — transfer analysis for our vLLM MTP stack

Source: `RaymondHuang210129/llama.cpp-adaptive-kv-streaming` README + `MTP_ADAPTIVE_KV_ROADMAP.md` +
`DEVICE_MEMORY_INFRASTRUCTURE.md` (provided 2026-09-30).

## Regime (why most of it is not drop-in)
- llama.cpp/ggml C++ under CUDA; single serial Qwen3.8-27B on a **16 GiB** GPU, 262144 ctx, Q8_0 K /
  Q4_0 V, Qwen35 geometry (48 recurrent Gated-DeltaNet + 16 full-attention layers, MTP = logical layer 17).
- We are vLLM 0.29 (Python) on ROCm/RDNA4, two 31.86 GiB cards, fp8 KV + fp16/fp32 ssm, block-paged KV
  with a **block-store** offload (CPU tier + shared fs disk tier), MTP head sharing target weights.
- Their KV architecture (attention over host-resident *spans* + a shared ring) is a much larger feature
  than our block-restore offload, and the ggml arena/lease/copy contracts and CUDA PDL lessons do not port.

So: mine *mechanisms and ordering rules*, not code.

## Transferable mechanisms (ranked by value to us)

### E1. MTP KV is volatile; handle it by suffix-only invalidation, not by exclusion (Milestones 10.x, MTP-opt 1)
What they do: MTP keeps its **own logical cache identity** and authoritative host history but shares the
17-layer physical budget; its tail is mutable and only its resident prefix may be reused across rounds
(stage 10.5/10.6, MTP-opt 2). Rejected drafts **truncate only the suffix** and rebase the unchanged
resident/ring prefix without H2D; only rewritten tail rows transfer (MTP-opt 1: target suffix
truncation previously invalidated the whole resident mirror and caused context-sized re-uploads).
Why it matters to us: our vLLM boot log already shows the same hazard being *avoided* rather than solved:
> `KV offloading: EAGLE/MTP draft attention groups [8] detected. The trailing chunk of these groups will be excluded from offloading due to volatility.`

That is the same classification — MTP/draft KV is too volatile to offload — and vLLM's answer is to
exclude it. The transferable idea is their answer: keep it, but track a **content generation**, dirty
rows, and **suffix-only invalidation** so a rejected draft does not invalidate or re-store the prefix.
Concretely for us: on draft rejection, invalidate/free only the truncated KV suffix and preserve the
prefix's block hashes / dirty state, both for the GPU prefix cache and for the offload tier; and treat
the MTP/drafter KV groups as include-able once the tail-handling exists instead of permanently excluded.
Effort: medium (offload/mamba bookkeeping), high payoff for decode+multi-turn, low risk to prefill.

### E2. Three-tier publication: reserved / device-ready / host-ready / committed (Stage 5.6)
What they do: one K/V publication ticket per append separates **device readiness** (attention may
consume) from **host readiness** (durable mirror), and a **committed frontier** that only advances over
a contiguous, generation-matched, host-ready range. Reserve-before-submit; never publish K without V;
never publish bytes past the committed frontier; retire resources only after the last referencing
completion. A backend without events uses the same state machine synchronously.
Why it matters to us: this is exactly the shape of our measured prefill problem — the offload store is
async on its own stream but its completion is **awaited at request finish (~18% of a cold 18k
prefill)**. Their rule "device attention proceeds on device-ready; host durability advances later" is
the design that removes our finish-time stall: let the append's KV durability lag the response, and
advance the offload frontier only when both K and V D2H complete, in token order.
Concrete for us: audit vLLM `OffloadingConnector`/tiering `wait()` sites and split store completion from
the response/attention path; add a host-committed frontier if the tier currently advances per-block.
Effort: medium-high (touches the offload adapter) but directly targets the prefill 18%.

### E3. Host-spilled recurrent rollback via a bounded GPU stage (Milestone 8)
What they do: current recurrent state stays on GPU; candidate checkpoints are published through a
**2-slot GPU stage** into **pinned host snapshots**; on rejection the chosen host snapshot is restored
and the KV suffix truncated (depth-3 ≈ 448.9 MiB pinned + 18.70 MiB GPU stage for their 48-layer
geometry). This replaces multiple full device snapshot planes.
Why it matters to us: we **disabled** lazy GDN snapshots because they corrupt multi-turn chat
(`RADIANCE_GDN_LAZY=0`). This is the exact mechanism shape for a correct+cheap GDN rollback: bounded
host snapshots keyed by acceptance prefix, restored only for the accepted rollback point.
Concrete for us: research a staged-snapshot GDN rollback in the spec path (the 48 recurrent blocks map
1:1 to their 48). Decode/correctness item, not prefill, but the highest-value borrow overall.
Effort: high; payoff: unblocks `GDN_LAZY` and reduces device snapshot memory.

### E4. Phase arena: prefill workspace vs decode KV under one parent (Milestone 6, esp. 6.3c)
What they do: one device parent allocation is lent to prefill (large graph/gather workspace, small KV
pool) and decode (small workspace, larger KV), with leases + drain + graph invalidation at the handoff.
Stage 6.3c measured **+4.98% decode** purely by converting idle prefill workspace into KV, unchanged
parent.
Why it matters to us: our MTP has a **draft-prefill pass** (speculator `prefill_cudagraph_manager` /
`_prefill`) whose workspace coexists at peak with target prefill, and the deployment has both prefill
workspace and the KV pool sized statically. The transferable idea is to size each phase's workspace
from measured phase maxima and reclaim the prefill-only workspace for KV during decode — but note we
*excluded* the phase-sharing A/B by decision, so this is a documentation-level follow-up, not scheduled.
Effort: high; vLLM already separates activation and KV, so the gain is smaller than their 4.98%.

### E5. Cross-token prefetch + sparse deadline feedback (Stages 6.6, 6.7a) — decode
They carry a layout-stable next-token prefetch window across the decode boundary (+2.26% decode) and
back off deadline instrumentation geometrically after clean runs, resetting on a miss (+0.63%, -0.258
ms/token). Both are decode-only; relevant only if we later stream KV for long context, and their own
gains are single-digit. Low priority for our prefill goal.

### E6. PDL early-completion lesson (Stage 5.5c) — not applicable
They found a resumed kernel inheriting `cudaTriggerProgrammaticLaunchCompletion()` let a consumed event
fire before the final ring read, and disabled early PDL for resumable kernels. CUDA-PDL-specific; ROCm
has no equivalent, so record the *principle* (a "consumed"/completion fence must not be recorded before
the last real read) but there is no code to port.

## What to expect (their own calibration)
- Their strict-quality prefill (full-layer gather) costs **-4.16% to -22.92%** prefill vs their
  partitioned reference at 64K–160K, and decode stays within 0.65–15.97% behind long-context. So even a
  mature adaptive-KV system pays a real prefill price for stock arithmetic — consistent with our own
  finding that MTP prefill is not free.
- Their wins are decode/long-context: 6.3c +4.98%, 6.6 +2.26%, 6.7a +0.63%. They do not report a large
  prefill win anywhere; do not expect this repo to solve our prefill gap.

## Not applicable
- ggml view/lease/arena contracts, `ggml-kv-stream` span kernels, CUDA streaming attention/copy kernels.
- Their regime numbers (single serial request, 16 GiB, IQ4/UD quants, Q8_0/Q4_0 KV) are not comparable
  to our aggregate serving numbers or our fp8 KV.
- Speculative-prefill / sparse prefill is **not** in this repo (that is vLLM #39060); this repo only
  handles MTP *decode* speculation + KV streaming.

## Concrete follow-ups to fold into our roadmap
1. **E1 (highest, transferable now):** implement suffix-only invalidation for draft rejection in the
   offload/prefix path, then reconsider including MTP/draft KV groups (currently excluded as volatile).
2. **E2:** split offload store completion from the attention/response path using a
   reserved/device-ready/host-ready/committed frontier — directly attacks the measured 18% prefill store.
3. **E3:** design a bounded host-snapshot GDN rollback (2-slot GPU stage) to re-enable lazy snapshots.
4. **E4/E5:** park; excluded by decision / decode-only with single-digit gains.
5. **E6:** keep the fence-ordering principle; no code.
