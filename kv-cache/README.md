# Opt-in KV-cache offload (off by default)

Adds a second-tier prefix cache behind the GPU prefix cache for **Qwen3.8-27B**
(Gated-DeltaNet hybrid, MXFP4 weights, FP8 KV) on one **AMD Radeon AI PRO R9700**
(gfx1201 / RDNA4, 32 GB, TP=1). Served by vLLM 0.27.1 / radiance 0.9.3.

This is **opt-in and off by default.** With `KVCACHE` unset (or `off`), `serve-mxfp4.sh`
generates a serve command that is byte-identical to the unmodified script — nothing is
mounted, patched, or passed to the engine.

## The knob

`serve-mxfp4.sh` reads one variable (and two sizing options):

```bash
KVCACHE=off    # default; also the state when KVCACHE is unset. Stock behaviour.
KVCACHE=ram    # GPU -> RAM tier in /dev/shm + the 6 "always" behavioural patches.
KVCACHE=disk   # GPU -> RAM -> disk + the full 15-patch instrumented set.

KVCACHE_RAM_GIB=16   # size of the RAM tier, GiB (KVCACHE=ram or disk)
KVCACHE_DISK=/kvcache   # host path of a dedicated filesystem (KVCACHE=disk only)
```

## The two options

### GPU -> RAM (the RAM option)

A RAM tier in `/dev/shm` behind the GPU prefix cache, plus the 6 "always" behavioural
patches (mixed-hit, eagle-groups, mamba-stride, reconcile-reask, swa-align/touch,
align-last-block). Needs `/dev/shm` big enough for `KVCACHE_RAM_GIB`. Runs with the
stock LRU policy and prompt-only offload (vLLM defaults).

### GPU -> RAM -> disk

The same RAM tier with a filesystem secondary tier behind it, plus the full 15-patch
instrumented set. Needs `KVCACHE_DISK` (a dedicated filesystem, mounted read-write into
the container) **and `kvcache-reap.sh`** — the fs tier writes and never deletes on its own.
Give the tier its own filesystem: the reaper keeps that filesystem at or below 70% by deleting
the oldest blocks, so anything else stored on it is paid for with cache.

```bash
sudo cp kv-cache/kvcache-reap.sh /usr/local/bin/
sudo cp kv-cache/kvcache-reap.{service,timer} /etc/systemd/system/
sudoedit /etc/systemd/system/kvcache-reap.service     # set KVCACHE_ROOT=<KVCACHE_DISK>/blocks
sudo systemctl daemon-reload && sudo systemctl enable --now kvcache-reap.timer
```

## What it does on this build

Measured with `./serve-tp1.sh` defaults (fp16 ssm cache, MAXSEQS 3, 220k context, 248,235-token
GPU pool), changing only `KVCACHE` (`KVCACHE_RAM_GIB=16`). The workload is `turnbench.py`: three
agent-style sessions growing to ~110k tokens each, served round-robin, so together they outgrow
the GPU pool from turn 4 on. Times are whole requests (prefill plus 320 generated tokens); "no
cache" is the same request replayed with a fresh cache salt. Brackets: tokens served from the tier.

| turn | prompt | no cache | `KVCACHE=ram` | `KVCACHE=disk` |
|---|--:|--:|--:|--:|
| B4 | 65,981 | 30.7 s | 12.8 s (49k) | 12.8 s (49k) |
| A5 | 80,895 | 37.8 s | 15.1 s (63k) | 15.2 s (63k) |
| C5 | 81,756 | 38.8 s | 13.2 s (67k) | 13.3 s (67k) |
| A6 | 95,502 | 46.7 s | 15.5 s (77k) | 15.5 s (77k) |
| B6 | 95,278 | 46.9 s | 21.6 s (67k) | 16.6 s (77k) |
| C6 | 96,116 | 47.2 s | 27.9 s (53k) | 18.0 s (77k) |
| A7 | 110,063 | 51.1 s | 44.2 s (21k) | 14.9 s (92k) |
| B7 | 108,878 | 51.8 s | 42.4 s (28k) | 16.5 s (92k) |
| C7 | 110,553 | 52.9 s | 49.6 s (11k) | 15.5 s (95k) |

A 16 GiB RAM tier carries the sessions until they outgrow it (turn 6 here); the disk tier keeps
every later turn near 15 s. All 21 turns served cleanly in both modes.

## Correctness gate

`turnbench.py` runs 3 sessions x 7 turns and compares every cached turn
token-for-token and logprob-for-logprob against a cold twin. It is the gate for
bit-identical KV-cache serving (the focus of this patch set; peak speed was not chased).
It compares bit-for-bit, so run it with an fp32 ssm cache.

## Watching it work

`watch -n 5 python3 kv-cache/kvwatch.py` shows GPU and offload-tier hit rates (lifetime and
since the last refresh), bytes the tier loaded and stored, and deferred lookups. It reads only
metrics stock vLLM exports, from `http://127.0.0.1:$PORT/metrics` (`KVWATCH_METRICS` overrides).
Read the "this refresh" column; lifetime rates on a long-lived server barely move.

Example, one R9700 after about five days of agent coding traffic (GPU -> RAM -> disk):

```
KV CACHE  qwen3.8-27b-vllm     17:49:04   delta: first sample
  running 0   waiting 0    pool   0.0%  preemptions 248

PREFIX CACHE            lifetime                   this refresh
  GPU             84.9%  (105.3 M / 124.0 M)      --    idle
  offload tier    47.5%  (8.9 M / 18.7 M)      --    idle

OFFLOAD TIER
  xfer buffers   0.00% busy  read 0.00%  write 0.00%   (in-flight, NOT tier fill)
  load     331.8 GB in   28.09s   ~11.8 GB/s     idle
  store    400.2 GB in   36.78s   ~10.9 GB/s     idle
  deferred lookups  n=467    mean wait 0.123s
```

The offload row counts only what the GPU cache missed: of the 18.7 M prompt tokens that fell
through to the tier, 8.9 M came back from RAM or disk instead of being prefilled again.

## Sizing

With a 16-bit ssm cache the attention block is 880 tokens rather than 1,648, so each offloaded
token carries twice as many Mamba states. Size `KVCACHE_RAM_GIB` from the token count you want
to hold.

## Files

- `patch_*.py` — the 15 consolidated patches (applied at boot from this directory, which
  rides the repo's existing `/patches` mount).
- `turnbench.py` (+ its deps `tierbench.py`, `equivbench.py`) — the correctness gate.
- `kvwatch.py` — live hit rates and tier traffic.
- `kvcache-reap.sh` + `.service`/`.timer` — eviction for the disk tier.

Evidence for these claims comes from radiance 0.9.3 / vLLM 0.27.1 on one R9700.
