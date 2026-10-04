# Plan: File-level KV pre-caching (EPIC / CacheBlend) for the int5 paro stack

Status: **PLAN ONLY — no implementation yet.** Deep research done 2026-10-03.

## Goal
Pre-compute KV for individual source-code files, store them on the existing disk KV tier, and
reuse them when those files appear in later prompts — WITHOUT recomputing the full file KV each
time. Files are "spans" that can appear at different positions / in different multi-file contexts.

Target stack: `paro5-27B-int5` (Qwen3.8-27B int5, RoPE, GQA, ~64 layers, ~8 KV heads, head_dim
128, bf16 KV), 2× R9700 (ROCm, **RDNA4 / gfx1201**), TP=2, **vLLM 0.29 V2 model runner**,
radiance overlay (boot-time Python source patches), existing disk KV tier (`/kvcache/blocks`,
60 GiB, content-deterministic hashes, `PYTHONHASHSEED=0`).

---

## 1. Why naive per-file pre-caching does NOT work
Transformer KV is sequential: the KV at token N depends on all tokens before it. A file prefilled
in isolation only attended to itself; its layer-2+ KV never saw the system prompt or other files.
vLLM's prefix cache is a content-addressed **prefix tree** (a block's hash chains through all
preceding tokens), so it only matches an *identical* prefix, not an arbitrary file embedded in a
varying multi-file context. Both EPIC and CacheBlend are "Position-Independent Caching (PIC)"
schemes that solve this by (a) making KV position-independent and (b) recomputing a small subset
to recover the lost cross-attention.

## 2. How each works (mechanism + benefits + our-stack caveats)

### 2.1 EPIC (Hu et al., ICML 2025, arXiv 2410.15332) — "LegoLink"
- **Two steps.** *Compile*: submit each immutable span in isolation, run a standard prefill with
  `max_tokens=0`, store its KV under a cache ID (position IDs start at 0). *Link*: retrieve cached
  KVs by ID, concatenate in any order, append the query tokens, **recompute a small subset**, decode.
- **KVSplit** (optional, standalone): chunk into semantic boundaries or fixed-size (512/1024 tok).
  **Code-file boundary detection is NOT documented; code was explicitly EXCLUDED from their eval.**
- **LegoLink (a.k.a. AttnLink in v1):** the "attention-sink" insight — a chunk's first tokens
  disproportionately absorb attention and trap the query. Fix: **recompute the first `k` tokens of
  each chunk (except the first) at their new absolute positions.** `k` is a fixed hyperparameter
  (tested 2/16/20/32). Recompute = `min(k, L)` per chunk, independent of L for L ≥ k.
- **Complexity O(k·N) ≈ O(N)** (k scales with #chunks, not N). No new CUDA kernels — the
  "masked attention across scattered tokens" is done in Python by re-driving existing attention
  kernels with custom block tables/masks/positions.
- **Benefits:** 8× TTFT / 7× throughput vs CacheBlend — but the 7× is a memory-capacity artifact of
  an all-cache-hit recompute-only workload; the defensible single-request number is **~3× TTFT**.
  Quality cost **0–7%** (LegoLink-2…32), measured on ≤9B document-QA, NVIDIA only.
- **Precompute** = one full prefill per file (one-time, amortized). **Storage** = 256 KiB/token
  (bf16) → 60 GiB ≈ 240K tokens ≈ ~480 files @500 tok / ~240 @1K / ~48 @5K.
- **Our-stack caveats:** it is a **vLLM fork** (not a library), ~2K lines Python, based on a
  **~1.0-era V1 runner** (paper says 0.4.1, README 0.7.0, tree has `vllm/v1/` — none match our
  0.29 V2). **NVIDIA-only** (H800/A100, CUDA 12.2, FlashAttention). All 5 touch-points (scheduler,
  KV manager, model runner, attention backend, entrypoints) must be re-derived onto 0.29 V2.

### 2.2 CacheBlend (Yao et al., EuroSys '25, arXiv 2405.16444) — LMCache
- **Selective recompute.** Reuse precomputed KV from a different context, then recompute the
  **High-KV-Deviation (HKVD) tokens** — the ~15% with the highest K-deviation — because those are
  the ones that attend across the chunk boundary. Selected by a **gradual per-layer filter**
  (layer 1 picks r1% slightly high, next layer re-picks from those, narrowing toward r=15%).
  `r=15%` is a **fixed target fraction** (not a threshold early-stop; that's a TODO in LMCache).
- **Position:** handles repositioning via **delta RoPE re-rotation** (rotate K by the position
  shift `m`, once, negligible cost). Distinct from the "store unrotated K, rotate in-kernel"
  approach — both work; pick one and stay consistent.
- **Pipelining:** overlaps recompute of layer L with load of layer L+1; hides recompute entirely
  while `recompute ≤ per-layer load time`.
- **Complexity O(0.15·N²)** — **quadratic in file size**. This is the weakness EPIC attacks.
- **Benefits:** 2.2–3.3× TTFT vs prefix caching (flatters the method); **4.1–6.6× vs full recompute**
  at 5–18% ratio. Quality **≤0.02 F1/Rouge-L** at 5–18%. Measured on NVIDIA A40, 4K RAG contexts,
  7B–70B Llama/Yi/Mistral.
- **Precompute** = one full prefill per file. **Storage** = 256 KiB/token → 10k file ≈ 2.6 GB,
  50k file ≈ 12.8 GB; 60 GiB ≈ 23× 10k files / 4–5× 50k files.
- **Our-stack caveats (CRITICAL):** it is a **Python library + vLLM KV-connector plugin**
  (`pip install lmcache`), not a fork. But:
  - **R9700 is gfx1201 (RDNA4). LMCache ships NO ROCm wheel for it** (only Instinct gfx942/gfx950).
    Must build LMCache's native (HIP) extension from source for gfx1201 + verify its Triton
    block-sparse CacheBlend kernels compile/run on RDNA4 (the R9700 is "second-class" in ROCm).
  - **MP-mode CacheBlend is CUDA-only** (GPU store/retrieve needs CUDA IPC events). On ROCm the
    only viable path is the **legacy in-process mode** (`LMCACHE_ENABLE_BLENDING`), which needs
    `enable_sparse=True` (Triton) + the gfx1201 build. A maintenance liability.
  - LMCache files **Qwen3.8 under "Hybrid Attention"** — must verify all ~64 layers are standard
    full-attention GQA with `.self_attn.rotary_emb` (the blender calls it per layer).
  - LMCache's disk store format ≠ our tier (blake3, chunk 256) — either run a second store or write
    a custom plugin.

### 2.3 Interaction: competing, NOT complementary
Both solve the *same* sub-problem (recover cross-attention for non-prefix reuse) via *competing*
selection strategies (static first-k vs dynamic deviation-top-15%). **Stacking both is redundant** —
you'd recompute a superset with no quality gain and added cost (EPIC's first-k ⊂ CacheBlend's 15%).
Position-independence (RoPE re-rotation) is shared by both and orthogonal — implement once.
**For large immutable files, EPIC/LegoLink (O(kN)) is the better fit; CacheBlend (O(0.15N²)) is the
better reference** for the deviation-selection + pipelining + storage-controller ideas.
Our prior research already established **position-independence is solved** (store V as-is; store K
unrotated as `W_K·X`, rotate in-kernel at ~5.7% overhead — MiniPIC, `github.com/IBM/vllm`). EPIC's
distinct contribution on top is the **attention-sink fix (recompute first k)**.

---

## 3. Performance comparison matrix (APPROXIMATES)

**Assumptions (flagged, to be measured):** prefill ~3000 tok/s (R9700, TP=2, conservative);
disk read ~3 GB/s (NVMe); file KV on **disk** (the realistic pre-cached case, not GPU-hot);
bf16 KV (256 KiB/token); EPIC `k=32` (LegoLink-32); CacheBlend `r=15%`. TTFT = time-to-first-token
for a request whose file-span is a cache hit. "Both" is shown but is **redundant** (see §2.3).

Per-token: compute ≈ 0.333 ms; disk load ≈ 0.087 ms → **max speedup ≈ compute/load ≈ 3.8×**
(this is the ceiling on this hardware with bf16 KV; both approaches are bounded by it).

| File size | Baseline (no cache, full recompute) | EPIC / LegoLink-32 | CacheBlend-15% | Both (redundant) |
|---|---|---|---|---|
| **1K tok** | 333 ms | ~98 ms (load 87 + recompute ~11) → **3.4×** | ~137 ms (load 87 + recompute ~50) → **2.4×** (≈3.8× pipelined) | ≈ CacheBlend, no gain |
| **5K tok** | 1667 ms | ~448 ms (load 437 + recompute ~11) → **3.7×** | ~687 ms (load 437 + recompute ~250) → **2.4×** (≈3.8× pipelined) | ≈ CacheBlend, no gain |
| **20K tok** | ~6667 ms (real higher, O(N²)) | ~1759 ms (load 1748 + recompute ~11) → **3.8×** | ~2748 ms (load 1748 + recompute ~1000) → **2.4×** (≈3.8× pipelined) | ≈ CacheBlend, no gain |

Quality / storage (all approaches reuse the same cached KV):

| | Quality vs full recompute | Storage per file (bf16) | Recompute cost model |
|---|---|---|---|
| EPIC / LegoLink-32 | **0–7%** (unvalidated on code) | 256 KiB/tok | O(k·N), k≤32, negligible here |
| CacheBlend-15% | **≤2%** (unvalidated on code) | 256 KiB/tok | O(0.15·N²), quadratic |
| Both | ≈ CacheBlend (no gain) | 256 KiB/tok | O(0.15·N²) (superset, wasted) |

**Reading the matrix:**
1. **Both are bounded by the disk-load ceiling (~3.8× here).** EPIC's recompute is negligible
   (always ~10 ms), so it always sits at the ceiling. CacheBlend's 15% recompute is hidden by
   pipelining only while `recompute ≤ load` (true for small/medium files); for large files where
   `T_full` grows super-linearly, CacheBlend's recompute exceeds load and it falls below the ceiling.
2. **To exceed ~3.8× you need faster disk or quantized KV** (4-bit KV → ~4× less load → ~4× higher
   ceiling). This is the main lever if 3.8× is not enough.
3. **Quality tradeoff:** CacheBlend is higher quality (≤2%) but more expensive; EPIC is cheaper
   (0–7%). For code, **both are unvalidated** — must be measured on our model/files.
4. **GPU-hot case (file KV already in GPU, no disk load):** EPIC → ~100×+ (recompute only, k/N of
   full); CacheBlend → ~6.7× (recompute only, 0.15 of full). The disk-load floor disappears.
5. All published numbers are **NVIDIA, ≤9B (EPIC) / 4K RAG (CacheBlend), code excluded** — our
   27B / R9700 / code numbers are first-order approximates, not measurements.

---

## 4. Key constraints & the single biggest blocker

- **S1 — ROCm attention backend (THE blocker).** The algorithmic heart of both is "a small set of
  tokens attends to all N tokens" (scattered/non-contiguous causal mask over borrowed blocks). This
  is **untested on the ROCm backend (Triton / `rocm_flash_attn`) on the V2 runner**. Everything else
  (KV store, scheduler, API, disk tier) is re-derivable or already exists; this one piece gates it all.
- **S2 — vLLM version gap.** Both targets are older/different vLLM (EPIC ~1.0 V1; LMCache examples
  target an older `gpu_worker.py` layout). All touch-points must be re-derived onto 0.29 V2
  (`vllm/v1/worker/gpu/model_runner.py`).
- **S3 — Storage is tight.** 60 GiB ≈ 240K tokens (bf16). A handful of large files or a few hundred
  small ones. Need an eviction/TTL policy (our content hashes map cleanly onto a cache-ID index).
- **S4 — CPU staging tier too small for large files.** The 2 GiB staging tier holds only ~8K tokens;
  a 20K-token file (5.12 GiB) does NOT fit → must raise the staging tier for large files or stream
  the file KV in chunks. (Connects to the earlier offload-sizing discussion.)
- **S5 — Code is unvalidated.** Neither paper evaluated code; both quality figures are document-QA.
- **S6 — int5 KV-group structure.** paro uses `RADIANCE_KV_GROUP_OPT=1` + a different KV-group
  layout; the stored-KV format and the in-kernel rotation must match it.

**Do NOT adopt either as-is.** EPIC = stale V1 fork, NVIDIA-only. LMCache/CacheBlend = no gfx1201
wheel, CUDA-only MP mode, O(0.15N²). **Steal the design, not the code** — build a minimal radiance
overlay on our existing tier.

---

## 5. Recommended approach (steal the design)

Build a minimal, env-gated radiance overlay that reuses our existing disk tier:
1. **Position-independence** — store **unrotated K (= `W_K·X`) + V** per file-span on the disk tier;
   re-rotate K in-kernel to the target position at link time (MiniPIC approach, ~5.7% overhead).
   (Or CacheBlend's delta-rotation — pick one, stay consistent.)
2. **Sink-fix recompute (EPIC/LegoLink)** — recompute the **first `k` tokens** of each reused
   file-span at its new position. O(k·N), k≤32, no runtime probe. This is the primary recompute.
3. **Deviation selection (CacheBlend) — only if needed** — if pure first-k quality is insufficient
   on code, add the HKVD top-15% selection as a fallback. Measure first; do not add speculatively.
4. **Pipelined load/recompute** (CacheBlend §5) to hide disk load behind recompute.
5. **Compile job** — a "precompute file KV" tool that runs each file through a `max_tokens=0`
   prefill and stores unrotated K + V on the disk tier, keyed by content hash.
6. **Link path** — on a request referencing cached file-spans: load spans from disk → re-rotate K to
   target positions → recompute first-k → decode.

---

## 6. Implementation plan (phased) — NO CODE YET

### Phase 0 — Validation gates (measure BEFORE building; de-risks S1/S5)
- **G1 (make-or-break, S1):** prove the ROCm attention backend can express "first-k tokens attend to
  all N over borrowed, non-contiguous blocks." A standalone Triton/`rocm_flash_attn` micro-benchmark
  (serving stopped) comparing scattered-mask attention vs a dense reference, for correctness + speed.
  If the backend cannot do it efficiently, stop and reconsider (this gates everything).
- **G2 (quality, S5):** on Qwen3.8-27B int5, measure quality of (a) full recompute [ceiling],
  (b) unrotated-K reuse + first-k recompute, (c) + CacheBlend deviation selection — on a few real
  code files at varied positions. Use turnbench/equivbench-style token+logprob comparison.
- **G3 (latency):** measure recompute + disk-load TTFT for 1K/5K/20K files on the R9700 (validates
  the matrix).
- **G4 (storage, S3/S4):** how many hot files fit in 60 GiB (bf16 vs 4-bit); confirm the staging-tier
  size needed for the largest target file.
- **Deliverable:** a go/no-go with measured (not approximated) numbers. **Gate the whole project on G1.**

### Phase 1 — Position-independence foundation
- Store unrotated K + V per span on the disk tier (extend the existing block format / namespace roll
  so new packed pages coexist with existing files without invalidating them).
- In-kernel RoPE re-rotation to target position at read time. Verify bit-exact vs a rotated reference.

### Phase 2 — Compile (precompute)
- `precompute-file-kv.py` tool: tokenize file → `max_tokens=0` prefill → extract unrotated K + V →
  write to disk tier keyed by content hash. Batch mode for "precompute the whole codebase."
- Record per-file token count + bytes for the storage ledger.

### Phase 3 — Link (reuse)
- Detect cached file-spans in an incoming prompt (by content hash).
- Load spans (disk→staging→GPU), re-rotate K to target positions, assemble the request block table
  over borrowed blocks, run the first-k recompute (Phase-0 G1 kernel), then decode.

### Phase 4 — Quality tuning
- Sweep `k` (2/16/32) on the G2 code workloads; pick the smallest k within the quality budget.
- Add CacheBlend deviation selection ONLY if first-k misses the quality bar (measure the delta).

### Phase 5 — Throughput + storage management
- Pipelined load/recompute (hide disk load behind recompute).
- Eviction/TTL policy on the disk tier (reuse the reaper; content-hash cache-ID index).
- Raise the CPU staging tier for large files, or chunk the promotion (S4).

### Phase 6 — Production hardening
- Env-gate the whole feature (default off); wire into the entrypoint like the other radiance patches.
- Full validation: turnbench/equivbench bit-identical gate, Grafana offload metrics, reaper health.
- Document in WORKLOG.

**Rollback:** env-gate off (single knob). Precomputed disk blocks remain (harmless, reaper-managed).

---

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ROCm backend can't do scattered-mask attention (S1) | Medium | Blocks project | Phase 0 G1 gate; fallback = smaller k / chunked spans |
| Code quality worse than 0–7% (S5) | Medium | Wrong answers | Phase 0 G2; CacheBlend deviation fallback; per-file TTL |
| vLLM 0.29 V2 re-derivation is large (S2) | High | Effort | Steal-design minimal overlay; isolate to attention + link path |
| 60 GiB too small (S3) | High | Few files cached | 4-bit KV (TurboQuant on-GPU), eviction/TTL, bigger tier |
| Staging tier < large file (S4) | Medium | Large files fail | Raise staging tier or chunk promotion |
| int5 KV-group mismatch (S6) | Low-Med | Format bugs | Match stored-KV format to `RADIANCE_KV_GROUP_OPT=1` |

## 8. Effort estimate
Multi-week to multi-month, dominated by (a) the Phase-0 G1 ROCm attention proof, (b) re-deriving the
link path onto 0.29 V2, and (c) code-specific quality validation. High integration risk, moderate
algorithmic risk (the algorithm is small), **high validation risk** (no published code/ROCm/27B data).

## 9. References
- EPIC: arXiv 2410.15332 (v1 AttnLink / v3 LegoLink), PMLR v267, `github.com/DerekHJH/epic` (fork, demo).
- CacheBlend: arXiv 2405.16444, `github.com/LMCache/LMCache` (`lmcache/v1/compute/blend/blender.py`,
  `.../attention/{utils,triton_sparse}.py`, PR #3092 ROCm Triton sparse backend).
- MiniPIC (position-independence, in-kernel RoPE): arXiv 2606.13126, `github.com/IBM/vllm`.
- KVShareArena (benchmark, "mutually blind" cross-source limit): arXiv 2609.10266.
- Prior session research: WORKLOG cont.34 (position-independent KV + cross-attention findings).
