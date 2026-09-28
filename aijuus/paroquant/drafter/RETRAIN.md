# DFlash2 drafter: build, retrain, serve

Runbook for producing a DFlash2 speculative drafter matched to a specific served target, and for
improving it over successive rounds. Written from the `Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ`
session on the 2x R9700 host, but the shape applies to any target.

## 1. Why this exists

A DFlash2 drafter's `fc` reads the **target's** hidden states. The stock
`tcclaviger/Qwen3.8-27B-DFlash2-FP8` was trained against stock Qwen3.8-27B, so serving it against a
requantized/retrained body (a blend, an OCP-GPTQ conversion) collapses acceptance. Measured here:
stock drafter on the blend target gave accept length ~1.28 (9.6% draft acceptance); after one
fine-tune round on target captures it reached ~2.1 expected accepted/block offline.

The fix is self-distillation: run the target, record its own hidden states + sampled tokens, and
fine-tune the drafter to predict the target's next tokens from those states. The prompt text is
irrelevant — only the spread of hidden-state trajectories matters.

## 2. Files and roles

| Path | Role |
|---|---|
| `serve-mxfp4.sh` | Launcher. `CAPTURE_DIR=<dir>` turns on drafter-data capture (forces prefix caching off) |
| `radiance_dflash_capture.py` | Capture hook; wraps `DFlashSpeculator.propose`, writes one `.pt` per request |
| `paroquant/drafter/build_drafter_for_target.sh` | Round-1 pipeline: prompt pool → capture → train → export, one target |
| `paroquant/drafter/retrain_r2.sh` | Later-round pipeline: richer pools, warm start, union captures, export |
| `paroquant/drafter/build_prompts.py` | Round-1 pool (1200 ultrachat / 700 CodeAlpaca / 500 GSM8K) |
| `paroquant/drafter/build_prompts2.py` | Round-2 categories: summarization, file edit, reasoning, JSON, prose (~3.2k) |
| `paroquant/drafter/build_prompts3.py` | Polyglot code, multi-turn chat, tool-calling (sets `messages`) |
| `paroquant/drafter/build_pool.py` | Large diverse pool, ~25k prompts across ~35 sources |
| `paroquant/drafter/generate.py` | Drives the served target over a pool; appends responses, resumes by id |
| `paroquant/drafter/train_drafter.py` | Self-distillation fine-tune (plain-torch DFlash2, fp32 master weights on CPU) |
| `paroquant/drafter/export_fp8.py` | Quantizes the bf16 fine-tune back to the FP8 serving layout |
| `bench-async.py` | Serves + benchmarks a target; reports `spec_acceptance` |

## 3. Host prerequisites and gotchas

- **Runtime**: `podman` (preferred) or `docker`. This host has **docker only**. Two consequences:
  - `--group-add keep-groups` is podman-only; for docker use numeric render/video GIDs.
  - Docker runs containers as root, so outputs under `~/models` and `~/drafter_ft` land root-owned.
- **GPUs**: HIP index 0 is the PCIe **x1** card (`05:00.0`, upstream bridge 1 lane @ 8 GT/s,
  ~0.8 GB/s host copy); HIP index 1 is the **x16** card (`09:00.0`, ~28 GB/s). Always capture/train
  on **GPU 1** (`GPUS=1 TRAIN_GPU=1`). Confirm with the launcher's own bandwidth report, or
  `rocm-smi` and `/sys/bus/pci/devices/0000:0{5,9}:00.0/current_link_width`.
- **Ports**: `serve-mxfp4.sh` defaults to **8080**, which `coolify-proxy` holds on this host. Always
  pass a free `PORT=` (3454 is used in the scripts; 8081 is free too).
- **GPUs must be idle**: stop any vLLM/bench first. `rocm-smi --showpids` should show no KFD PIDs.
- **HF cache**: `~/.cache/huggingface` may be absent (deleted for space). Capture/train don't need
  it (the target carries its own tokenizer; training reads local dirs). The **pool builders do** —
  they stream from the Hub and are best-effort (a missing source just shrinks the mix).
- **Disk**: captures cost ~**27 KB per captured token**. A 6.6k-prompt round at up to 1536 tokens
  is ~50-100 GB. Check `df -h /home/juup` first.
- **Run it in a file, detached**: never paste a long script into an interactive shell (paste
  corruption silently truncates it). Use `tmux new -d -s retrain 'bash <abs path to script> …'`.
- **AITER rebuild**: `module_aiter_core` is compiled into the container's site-packages, so every
  fresh serve container pays ~**421 s** to rebuild it. Expect ~10 min from launch to `/health`.
  (The Triton/inductor cache in `~/.radiance-cache-*` is separate and persists.)

## 4. What capture records

With `CAPTURE_DIR` set, prefix caching is forced off (a cache-hit prefix produces no hidden states).
For every request, per token position, the hook writes:

- the target's **aux hidden states** (Eagle3 layers `[5,19,33,47,61]`, concatenated, stored e4m3
  with one scale per token) — what the drafter's `fc` reads at serve time;
- the **token** and **position**.

One `.pt` per request, flushed when the request leaves the batch; rank 0 only. Which drafter
generates does **not** affect the data: speculative decoding is lossless, so the recorded tokens and
hidden states are the target's regardless. A matched drafter only makes generation faster.

## 5. Round 1 — build a drafter for a new target

`build_drafter_for_target.sh <TARGET_DIR_NAME> [OUT_DRAFTER_DIR_NAME]` runs the whole pipeline
(pool → capture → train → export) for one target.

```bash
cd /home/juup/radiance-vllm-mxfp4
# stop production first; it needs the GPU
PORT=3454 GPUS=1 TRAIN_GPU=1 \
  paroquant/drafter/build_drafter_for_target.sh \
  Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ
```

Key env (defaults in brackets): `MODELS [~/models]`, `DRAFTER_WORK [~/drafter_ft]`,
`DRAFTER_BASE [$MODELS/Qwen3.8-27B-DFlash2-FP8]`, `IMG [stilldeadcode/vllm-radiance:0.9.3]`,
`PORT [8000]`, `TP [1] GPUS [1] TRAIN_GPU [1]`, `MAXLEN [32768] MAXSEQS [8]`,
`CONC [8] GEN_TOKENS [768] SPEC [3]`, `EPOCHS [2] LR [5e-5] SEQS [6] ANCHORS [64]`,
and resume flags `SKIP_PROMPTS SKIP_CAPTURE SKIP_GENERATE SKIP_TRAIN SKIP_EXPORT`,
`REUSE_CAPTURE`.

At capture time it uses the **stock** drafter at `SPEC=3`: capture only needs `propose()` to run,
and a mismatched drafter accepts `<1 tok/draft`, so deeper proposals would be wasted work. Output:
`$MODELS/<TARGET>-DFlash2-FP8`.

## 6. Round N — improve with richer data (warm start + union)

Use `paroquant/drafter/retrain_r2.sh` (edit the knobs at the top; no interactive paste):

```bash
tmux new -d -s retrain \
  'bash /home/juup/radiance-vllm-mxfp4/paroquant/drafter/retrain_r2.sh 2>&1 | tee -a ~/drafter_ft/r2/run.log'
tmux attach -t retrain        # detach with Ctrl-b then d (NOT Ctrl-C)
```

It does, in order:

1. builds `build_prompts2.py`, `build_prompts3.py`, `build_pool.py` into `~/drafter_ft/r2/`;
2. samples `SAMPLE` (default 2500) from each and merges to `r2/prompts.jsonl` (keeps `messages`,
   so multi-turn/tool prompts stay structured);
3. captures with the **round-1 drafter** at `SPEC=7` (matches its `block_size=8`),
   `GEN_TOKENS=1536`, `RADIANCE_DFLASH_CAPTURE_MAX_TOKENS=8192`;
4. trains **warm-started** from the round-1 export, on the **union** of round-1 + round-2 captures;
5. exports to `$MODELS/<TARGET>-DFlash2-FP8-r2`.

Why warm start + union: round 1 saturated on its own data (eval CE 1.72→1.66 and
expected-accepted 2.03→2.115 from step 300 to 768), so epochs on the same prompts are wasted, but
initializing from those weights saves re-learning and the union prevents forgetting. Warm start is
supported because `train_drafter.py` dequantizes the exported fp8 via `weight_scale_inv`; **the
`--drafter` dir must contain `config.json`**, so point at the exported drafter, not the bf16 `ft-`
intermediate.

### Training knobs that matter

| Knob | Default | Effect |
|---|---|---|
| `--seqs` | 6 (script) / 4 (trainer) | sequences per step; raise to 8-12 for more signal per 12 s step |
| `--anchors` | 64 (script) / 64 | anchors per sequence per step; 96-128 with more data |
| `--gamma` | 4.0 | position weight `exp(-(k-1)/gamma)`; lower to 2-3 to stop discounting deep block positions |
| `--lr` | 5e-5 (script) | peak LR; cosine, warmup 4% |
| `--epochs` | 2 | only helps if there is new data |
| `--max-len` | 4096 | must be raised to match longer captures (use `CAP_MAX`) |
| `--val-dir` | — | hold out a capture dir from a *different* pool for a cleaner proxy |

## 7. Resume semantics

Resumable, by design:

- `generate.py` appends each completed response to `resp-*.jsonl` and, on restart, skips ids already
  present. `.pt` captures persist.
- `retrain_r2.sh` **skips pool build/merge when `prompts.jsonl` exists**, so ids stay identical.
  Never set `FORCE_POOLS=1` on a resume — rebuilding reshuffles ids and the done-set skips/duplicates.
- It trims a truncated final line from `resp-*.jsonl` before parsing (a kill mid-write would
  otherwise abort the resume).
- `trap … INT TERM` stops the capture container on interrupt; the next run's preflight also removes
  a lingering `radiance-drafter-cap`.

Not resumable: **training**. No optimizer/scheduler state is saved, so stopping during step 4 means
re-running it (generation itself resumes instantly). Stop only during generation.

Stop / resume:

```bash
tmux kill-session -t retrain 2>/dev/null
docker stop radiance-drafter-cap
# later
tmux new -d -s retrain 'bash /home/juup/radiance-vllm-mxfp4/paroquant/drafter/retrain_r2.sh 2>&1 | tee -a ~/drafter_ft/r2/run.log'
```

In-flight requests' buffer is lost on kill (not corrupted) — a handful of files.

## 8. Doing stages manually

For full control, run the stages directly.

Capture serve (round-1 behaviour: stock drafter, `SPEC=3`):

```bash
CAPTURE_DIR=$HOME/drafter_ft/r2/cap-Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ \
SERVED_NAMES=Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ DETACH=1 NAME=radiance-drafter-cap \
PORT=3454 GPUS=1 MAXLEN=32768 MAXSEQS=8 SPEC=3 SPEC_METHOD=dflash \
RADIANCE_DFLASH_CAPTURE_MAX_TOKENS=8192 \
SNAP=$HOME/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ \
DRAFTER=$HOME/models/Qwen3.8-27B-DFlash2-FP8 \
./serve-mxfp4.sh
```

Drive it (wait for `curl -sf localhost:3454/health` first):

```bash
BENCH_URL=http://localhost:3454/v1/chat/completions \
BENCH_MODEL=Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ \
python3 paroquant/drafter/generate.py <pool.jsonl> <resp.jsonl> 8 1536
sleep 40; docker stop radiance-drafter-cap
```

Train (union + warm start; docker group flags shown for docker):

```bash
docker run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=1 \
  -v "$HOME/drafter_ft":/data:z -v "$HOME/models":/models:z -v "$PWD/paroquant/drafter":/scripts:z \
  --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc \
  "cd /data && python3 /scripts/train_drafter.py \
     --capture /data/cap-<TARGET>,/data/r2/cap-<TARGET> \
     --drafter /models/<TARGET>-DFlash2-FP8 \
     --target  /models/<TARGET> --out /data/ft2 \
     --epochs 2 --lr 5e-5 --seqs 8 --anchors 96 --gamma 3 --max-len 8192 \
     --eval-every 150 --save-every 300"
```

Export (`ref` must be the FP8 drafter whose `quantization_config` supplies the layout):

```bash
docker run --rm -v "$HOME/drafter_ft":/data:z -v "$HOME/models":/models:z \
  -v "$PWD/paroquant/drafter":/scripts:z --entrypoint python3 stilldeadcode/vllm-radiance:0.9.3 \
  /scripts/export_fp8.py /data/ft2 /models/Qwen3.8-27B-DFlash2-FP8 /models/<OUT>
```

## 9. Serve and verify

```bash
cd /home/juup/radiance-vllm-mxfp4
PORT=8081 SNAP=~/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ \
  DRAFTER=~/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-DFlash2-FP8-r2 ./serve-mxfp4.sh
```

Serve with `SPEC=7` (default for dflash; matches `block_size=8`). The cheapest acceptance lever is
inference-time candidate width — the selector is a trained rank-256 scorer and `top_k` only
truncates its output:

```bash
RADIANCE_DFLASH_SELECTOR_TOPK=32 RADIANCE_DRAFT_RERANK=128 ... ./serve-mxfp4.sh
```

Keep `RADIANCE_DRAFT_RERANK >= 4 x` the new k (and >= 4 x the sampler's top_k=20).

Verify with the benchmark, not the offline proxy:

```bash
./bench-async.py --launch --phases all --label dflash-ft \
  --save-dir "$PWD/bench" --models /home/juup/models \
  --target '{"name":"<T>-ft","model":"<T>","snap":"/models/<T>","drafter":"/models/<OUT>","spec_method":"dflash","maxlen":"32768"}'
```

Valid phases are `decode`, `prefill`, `concurrency` (`pressure` is an alias; `cold`/`hit` were
removed). Read `spec_acceptance` in `bench/overview.json` or the `SpecDecoding metrics` log lines.

## 10. Reading training eval output

```
[eval step N] weighted CE <c> | top1 by position a b c d e f g | prefix-expected accepted/block <e>
```

- `weighted CE`: loss at the 7 mask positions.
- `top1 by position`: per-position draft accuracy; position 1 dominates acceptance.
- `prefix-expected accepted/block`: expected accepted tokens per verify step (the acceptance
  proxy). Stock-on-blend ≈ 0.18 pre-training, ≈ 2.1 post round 1; in-serve accept length was ~1.28
  with the stock drafter. Treat the offline number as optimistic.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `port 8080 is already in use` | `serve-mxfp4.sh` default is 8080, held by `coolify-proxy`. Pass `PORT=8081` |
| `unable to find group keep-groups` | `--group-add keep-groups` is podman-only. Use numeric `render`/`video` GIDs (patched in both scripts) |
| Capture container vanishes on startup | `bench-async.py`'s `cleanup_stale` (`bench-async.py:1014`) `docker rm -f`s every `bench-*` container. The capture container is named `radiance-drafter-cap` to avoid this |
| RAM climbs ~1 GiB/30 s, GPU VRAM stays 0%, `ps` shows a chain of `_build_optional_torch_c_dlpack.py` | tvm_ffi JIT-build recursion, triggered by the capture sitecustomize hook importing vllm in the build subprocess. `serve-mxfp4.sh` now sets `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` whenever `CAPTURE_DIR` is set |
| `/health` silent for many minutes | Normal: weight load + `torch.compile` + (first run) AITER build. Watch container CPU% and GPU VRAM; the script prints a 30 s heartbeat |
| Both a bench and a capture stall | Don't run two vLLM startups at once; they thrash the shared compile cache. Stop production first |
| `phase 'cold' was removed` | bench-async phases are `decode,prefill,concurrency` |
| Resume skips/duplicates prompts | `FORCE_POOLS=1` was set on a resume; ids must stay stable |
| Resume crashes parsing `resp-*.jsonl` | Truncated last line; the script trims it, or delete the partial line manually |
| Warm start loads 0 tensors / KeyError | `--drafter` must be a dir with `config.json` (the export), not the bf16 `ft-` dir |
| Capture unnecessarily slow | Using the stock drafter at `SPEC=3`; use the matched drafter at `SPEC=7` (this speeds generation; data is unchanged) |

## 12. Disk and cleanup

- Keep `~/drafter_ft/cap-<TARGET>` while it is part of a union; it is large (~43 GB for 1.6M tokens).
- Safe to delete: `~/drafter_ft/ft-*` (bf16 intermediate), empty capture dirs, `docker system prune -f`
  (stopped containers + dangling images; avoid `docker image prune -a` so the radiance image stays).
- After the new export is verified, the previous round's drafter and captures are removable.
- `~/.cache/huggingface` is only needed by the pool builders (they stream, so it does not refill to
  its old size). `~/.radiance-cache-*` is the compile cache — keep the active variant.

## 13. Repository changes made in this session

- `serve-mxfp4.sh`: when `CAPTURE_DIR` is set, default `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` and pass it
  into the container (`=0` otherwise). Breaks the tvm_ffi capture fork-bomb.
- `paroquant/drafter/build_drafter_for_target.sh`: podman/docker group-flag split; capture container
  renamed `bench-*` → `radiance-drafter-cap`; full container logs streamed to the terminal; 30 s
  progress heartbeat; dumps the container log tail if it dies during startup.
- `paroquant/drafter/retrain_r2.sh`: the round-N pipeline (pools, capture with the matched drafter,
  warm-start union training, export) with the resume-safety guards described in section 7.
