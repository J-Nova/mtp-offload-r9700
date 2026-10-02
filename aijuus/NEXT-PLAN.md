# NEXT PLAN — consolidated work index

Owner: project. Created 2026-10-02 (cont.92). This is the single prioritized index for the current
work batch; `OPEN-TASKS-INDEX.md` remains the long-horizon catalog. Status: **READY** (actionable) ·
**USER** (needs your restart/recreate) · **MEASURE-FIRST** (gated on a measurement) · **DESIGN**
(design+validation) · **PARKED**.

Restarts/recreates are user-owned. Overlay patches are marker-idempotent, so a `docker start` does
NOT re-apply changed patches — durable changes need a container **recreate**.

## Progress (cont.92)

| ID | Status | Result |
|---|---|---|
| N1 | **DONE** | `patch_gdn_lazy.py` sentinel comment now contains the sentinel → `apply()` NOOPs on re-run (unit-tested: 2 applies → 1 block). Recreate resets live count to 1. |
| N3 | **DONE** | registry `RADIANCE_MXFP4_DECODE_MAX_M` 192→128. |
| N4 | **DONE** | committed `8ecdabe`. |
| N8 | **DONE — thermal confirmed** | sustained decode on vllm-1 + 2 Hz sampling: **GPU1** (load) junction max 71 °C, sclk ≤3053 MHz, mclk 1258, fan 12 %. **GPU0** (vllm-0) junction 98–**103 °C**, sclk pulled 2997→2711 MHz, fan only **34 % (2322 rpm)**, power avg 209 W (cap 215 applied). The `>3.3 GHz` knee is avoided, but **GPU0 saturates thermally because LACT `fan_control_enabled: false` leaves the PMFW acoustic curve (target 85 °C, acoustic_limit 2600) too lazy** → junction throttle. That is the vllm-0 throughput constraint. |
| N7 | **DONE (bs=1 + bs=8)** | Kernel attribution captured (traces at `/tmp/kilo/prof/`, analyzer `aijuus/profile_analyze.py`). Target GEMM ~58–60 % of GPU-busy; **bf16 unquantized weights ~21 % (bs=8) / ~28 % (bs=1)** (MTP block `wvSplitK` + `lm_head` `Cijk`); GDN ~3 %→**9 %** (bs1→bs8); attention ~0.5–1 % (short ctx); draft head int2 ~2 %. Kernel-busy ≈ **80 % of the generation wall** at both bs1 and bs8 → **step is kernel-bound, not host/launch-bound**. Profiler caveat: torch profiler × rocprofiler-sdk crashes on the **second** stop in one process (`External init callback must run in same thread as registerClient`); single start/stop per boot is safe. |
| N2 | **USER** | recreate both containers (Coolify redeploy of project `w7p20ww076tcrg9zhq7zhmne`, or `docker compose ... up -d --force-recreate`) to apply N1/N3 + acc-gate-off + reset invalidate. |

## Decision log (this session — closed/clarified)

| # | Decision | Evidence |
|---|---|---|
| D1 | **MTP acceptance gate is disabled** (`RADIANCE_ACC_GATE=0`). As configured it was net **−16%** on sampled high-entropy (the live default), +6% greedy, neutral predictable. | acc_gate_check on vllm-1, cont.92 |
| D2 | **Draft depth is a sub-10% lever**, not a "4× loss". The draft loop is ~8% of the step; the target weight stream dominates. The vllm-0 10 tok/s was thermal + context. | step 42.7→39.3 ms vs tokens/step 2.90→2.23, cont.92 |
| D3 | **`mxfp4a8`/`_r4d` is NOT broken.** `RADIANCE_MXFP4_R4D_DECODE_MAX_M=0` disables that alternate path by design; the active `radiance_mxfp4_fp8.hip` W4A8 decode kernel is ON (304/304, 95.5% BW). `r4d.select(mxfp4a8)=None` because our libr4d extras never built it. A redeploy changes nothing. | boot log + `r4d.select` in both containers |
| D4 | **48×/N× `invalidate` is a patch-hygiene bug, not correctness.** It zeroes the lazy stash header N times (idempotent). Cost only when lazy ON: N launches + N H2D/step. Accumulates per restart. | live counts vllm-0=48, vllm-1=6; sentinel absent from inserted text |
| D5 | **Acceptance collapse = high-entropy output**, not thermal/draft/lazy. Same drafter: p0 1.00 predictable, 0.03 random. | acc_gate_check, cont.91/92 |
| D6 | **Thermal/clock was a real vllm-0 constraint** (97 °C junction, sclk 3364 MHz); LACT changed 08:00 to 215 W / −65 mV / sclk −450. Under-load validation pending. | rocm-smi + /etc/lact/config.yaml |
| D7 | Upstream vLLM's dynamic-spec-depth is **broken on MRv2 too** (#51510); fixes unmerged. Per-request MTP depth is not a viable track. Adaptive Verification is DSpark-only (MTP has no confidence head here). | upstream issues/PRs, agent report |

## Phase 0 — Durability & hygiene (READY, no perf risk)

| ID | Task | Why | Effort | Risk |
|---|---|---|---|---|
| N1 | Fix `patch_gdn_lazy.py` **sentinel** (use a string present in the inserted text) | stops N× invalidate growth per restart | trivial | none |
| N2 | **Recreate** both containers (fresh FS) after N1/N3/N4 | apply all `/patches` changes durably; reset invalidate count to 1; `docker start` cannot | low (USER) | none |
| N3 | Set `RADIANCE_MXFP4_DECODE_MAX_M` **192 → 128** | kernel serves ≤128 (`DEC_MAX_TM·16`); 129–192 silently takes the prefill tile | trivial | none |
| N4 | **Commit** the overlay set (rx12 patch, `serve-mxfp4.sh`, `coolify-compose-2gpu.yml`, `entrypoint.sh`, `model-registry.json`, `radiance_gdn_lazy.py`, `acc_gate_check.py`, `apply_acc_gate_live.py`, WORKLOG/INDEX/NEXT-PLAN) | H1 continuity | low | none |
| N5 | Verify libr4d **v0.5.0/rx12** live after recreate; confirm `r4d mxfp4a8` intentionally OFF | D3 | trivial | none |
| N6 | Confirm `RADIANCE_ACC_GATE=0` inert after recreate | D1 | trivial | none |

## Phase 1 — Measurement (MEASURE-FIRST; decides Phase 2/3)

| ID | Task | Command sketch | Why |
|---|---|---|---|
| N7 | **GPU kernel-time trace** bs1/bs8, aggregate by op + inter-kernel gaps | restart with `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/prof}'`; `POST /start_profile` → 20 steps → `POST /stop_profile`; alt `rocprofv3 --kernel-trace --stats` | localizes the 17–31 ms gap: kernel/clock vs host/launch bubbles. **The key experiment.** |
| N8 | **Thermal/clock validation under load** | `rocm-smi --showclocks --showtemp --showpower -l 0.1` during a decode run; record sclk>3300 fraction, mclk residency, junction, power vs 215 W | confirms D6 and whether LACT fixed it |
| N9 | Attention isolated timing at live N/ctx (N∈{1,2,8}, ctx∈{17k,34k}) | kernel bench or trace grouping R4D decode+combine | verifies the 7% claim vs ~30% at N=8 |
| N10 | Re-run `bench-eval.py` + `acc_gate_check.py` on both models | `./aijuus/acc_gate_check.py --base http://172.18.0.20:8000 --model mtp-27B-MXFP4-Thinkingcap` etc. | baseline + acceptance by position |

## Phase 2 — Cheap wins (low risk; mostly post-recreate)

| ID | Task | Est. | Effort | Risk | Gate |
|---|---|---|---|---|---|
| N11 | GDN: gate `core_attn_out.zero_()` (all-R4D) | +0.6% exact | low | low | — |
| N12 | GDN: unblock `in_proj` merge under PRESHUFFLE | −2.9% c1 / −6.5% stacked | med | med | GSM8K paired |
| N13 | GDN: fuse `conv_update`+`lazy_update` (lazy-only) | 0.1–0.2 ms | med (kernel) | med | fp32 bit-exact |
| N14 | Attention **output-quant epilogue fusion** (upstream-inspired) | small; −1 launch × 17 | med | low-med | A/B + bit-compare |
| N15 | **int2 verify head** `RADIANCE_VERIFY_HEAD=1` | +7% (GSM8K) … +36% (long greedy); +17.5% bs8 | low | med: needs `top_k ≤ RERANK/4` → RERANK≥80 for sampled | **DONE/PASS (cont.92).** Bug fixed (one-row liveness check; hook logs the first exception). Head arms, 194/200 on int2. **GSM8K 200q exactness A/B: both arms 93.5% (187/200) with the IDENTICAL wrong-question set**; arm A deterministic (0/200 rerun diff), arm B perturbs 8/200 answer texts but all keep the same final number (7) or are both truncations (1) → no correctness change. Kernel: lm_head `Cijk` 2.285→0.573 ms/call; sampler top-k collapses (80 finite entries vs 248320). Throughput A/B clean+idle: bs1 greedy 107.6→146.6, bs8 342.9→403.1, GSM8K 107.6→115.6 tok/s. **Sampled margin gate (temp 0.7, top_k=20=RERANK/4, seed 1234): accuracy 92.0% both, IDENTICAL wrong sets; 18/200 answer-text diffs (not bit-exact) but only 2 final-number changes, both already-wrong/truncated → correctness preserved.** **To enable in prod: `VERIFY_HEAD=1` + `RADIANCE_DRAFT_RERANK=80`** (prod is 32); net-vs-32 baseline not yet measured. If bit-reproducibility of sampled output is required, raise RERANK (128) or lower selector top_k. |
| N16 | Attention split/TILE retune at live N/ctx | low-med | low | low | **gated on N9** |

## Phase 3 — Structural (DESIGN / research)

| ID | Task | Est. | Effort | Risk |
|---|---|---|---|---|
| N17 | **Quantize MTP block bf16 → mxfp4** (largest un-streamed bf16, ~0.85 GB/slot) | up to 5–7 ms | high | acceptance (design+validation; MTP Q8_0/Q4_0 restrictions) |
| N18 | **Batch the 48 GDN layers** into fewer launches | up to ~2.2 ms | high | high (state/weight slicing) |
| N19 | **M10 SSM dtype A/B** fp16/bf16/fp32 (rx12 enables bf16+lazy) | accuracy/throughput | med | med |
| N20 | Adaptive Verification for MTP (wire a confidence head) | unknown | high | research |
| N21 | Prefill/TTFT track (P2/P3/P10) — only if prefill prioritized | — | — | — |

## Parked / do not repeat

- Per-request MTP dynamic depth (upstream broken #51510; for fork, K is batch-level by design).
- `RADIANCE_ACC_GATE` (disabled; net-negative sampled).
- libr4d `r4d mxfp4a8` decode path (off by design; no evidence it beats 95.5%-BW kernel).
- Marlin/CUTLASS persistent GEMMs (CUDA-only).
- See `OPEN-TASKS-INDEX.md` §"Do NOT repeat".

## Suggested order
**N1 → N3 → N4 → N2 (recreate) → N7 ∥ N8 → N9/N10 → N11–N16 → N17+**
