# Fastboot: cutting a vLLM/ROCm boot from ~20 min to ~4 min

If your model takes 14–22 minutes to come up *every* time, nearly all of that is
work being redone from scratch on each boot. There are two independent costs,
and they need two different fixes. Doing only one of them barely helps.

| | cost per boot | fix |
|---|---|---|
| 1. aiter JIT-builds `module_aiter_core.so` | ~7 min | **bake it into the image** (§1) |
| 2. vLLM traces + compiles the model graphs | the rest | **persist a cache volume** (§2) |

Measured on a 2×R9700 (gfx1201) TP=2 serve, each fix against its own control:

- **§1 saves ~440 s** — 650 s → 211 s, same warm cache either side.
- **§2 saves ~130 s** — 381 s on a cold cache directory → 251 s warm, same image.

A warm boot of the current prod image is **3 min 53 s** wall clock.

---

## §1 — Bake the aiter `.so` into the image

### Why caching it does not work

With `VLLM_ROCM_USE_AITER=1`, aiter builds `module_aiter_core.so` lazily on
first use. The obvious move is to point `AITER_ROOT_DIR` at a persistent volume
and let it cache. **That does not work, and it is not obvious why.**

`AITER_ROOT_DIR` does not govern the JIT output path. Proof from the live
container, which has `AITER_ROOT_DIR=/cache/aiter` set:

```console
$ docker exec <container> printenv AITER_ROOT_DIR
/cache/aiter

$ docker exec <container> find /cache/aiter
/cache/aiter
/cache/aiter/build                 # empty skeleton, nothing else, ever

$ docker exec <container> /opt/vllm/bin/python3 -c "
import sys, importlib.util
J='/opt/vllm/lib/python3.12/site-packages/aiter/jit'
sys.path.insert(0, J+'/utils')
s=importlib.util.spec_from_file_location('c', J+'/core.py'); c=importlib.util.module_from_spec(s)
sys.modules['c']=c; s.loader.exec_module(c)
print(c.get_user_jit_dir())"
/opt/vllm/lib/python3.12/site-packages/aiter/jit
```

The mount is never the target. `aiter/jit/core.py` builds its probe path as
`os.path.join(get_user_jit_dir(), f"{md_name}.so")` and rebuilds when that file
is absent — and as shown above, `get_user_jit_dir()` resolves **inside
site-packages**, i.e. inside the container's writable layer. That layer is
destroyed on every `docker compose down` / recreate (a plain `stop`/`start`
keeps it, which is why the problem can look intermittent). So the mounted cache
collects an empty `build/` directory and the ~7 minutes of hipcc is paid again
on the next recreate, forever.

The fix is therefore not to cache it. It is to put the `.so` somewhere that
*does* survive: a **read-only image layer**.

### Method A — build it at image-build time (preferred)

This repo ships `bake_aiter_core.py`, which drives aiter's own
`core.build_module("module_aiter_core")` during `docker build` and drops the
artifact where `get_module()` will find it:

```dockerfile
COPY bake_aiter_core.py /opt/bake_aiter_core.py
RUN set -e; /opt/vllm/bin/python3 /opt/bake_aiter_core.py
```

`Dockerfile.ggz14` already calls it this way. Note it is a **plain `COPY` + plain
`RUN` of a repository file, not a heredoc**: the classic builder
(`DOCKER_BUILDKIT=0`) discards a no-backslash in-file heredoc and runs `python -`
clean on EOF, which fails silently — every test passes and no `.so` exists
anywhere. The file form is safe under both classic and buildkit.

Why this is the better method: **hipcc needs no GPU**, so it runs on any builder,
and you never pay a slow first boot at all. The script asserts its way through
(layout check, build-arg identity check, size check, and a final GPU-free
`import module_aiter_core`), so drift is a loud build failure rather than a
silently skipped bake.

**The one thing you must change for your hardware.** The script sets the build
env explicitly:

```python
os.environ["GPU_ARCHS"] = "gfx1201"   # your GPU's arch
os.environ["CU_NUM"] = "64"           # your GPU's CU count
```

These are the values the runtime would otherwise resolve from a live device
query, and they are hard-coded because the builder has no GPU. Set them to your
card's values or you are baking for the wrong target. Two mitigations make this
safe in practice: the hipcc `--offload-arch` output carries no `amdhsa--gfx`
marker, so aiter 0.1.17's `_needs_arch_rebuild` treats it as *adopted, never
rebuilt*; and a genuinely chip-marked `.so` built for the wrong arch would be
force-rebuilt in place by aiter itself. You get a slow boot, not a wedge.

### Method B — copy it out of a running container (overlay)

Use this when you cannot rebuild the image — a published tag, someone else's
image, a base you don't have sources for.

**Step 1.** Boot the container once, normally, and wait for healthy. That boot
is slow; it is the last slow one.

**Step 2.** Copy the built JIT tree out (~3 MB):

```bash
mkdir -p aiter-jit-prebuilt
docker cp <container>:/opt/vllm/lib/python3.12/site-packages/aiter/jit/. aiter-jit-prebuilt/
ls -l aiter-jit-prebuilt/module_aiter_core.so     # ~0.67 MB for a single-TU, CK-off build
```

Check the python version in that path matches your image (`python3.12` above).

**Step 3.** Make a one-layer child image. That is the entire Dockerfile:

```dockerfile
ARG BASE_IMAGE=<your-image>
FROM ${BASE_IMAGE}
COPY aiter-jit-prebuilt/ /opt/vllm/lib/python3.12/site-packages/aiter/jit/
```

```bash
docker build -f Dockerfile.fastboot \
  --build-arg BASE_IMAGE=<your-image> \
  -t <your-image>-fastboot .
```

Run `<your-image>-fastboot` from then on. These are the exact bytes the runtime
was already importing, built on the actual target device, so it is not a
numerics change — verified perf-null on full benchmark lanes here.

**Rebuild `aiter-jit-prebuilt/` whenever the base image changes.** A stale `.so`
carried onto a new base is a crash, not a slow boot.

### Which method

| | Method A (build-time bake) | Method B (copy-out overlay) |
|---|---|---|
| needs image sources | yes | no |
| needs a GPU | no | yes (for the one slow boot) |
| arch correctness | you set `GPU_ARCHS` / `CU_NUM` by hand | implicit — built on the real device |
| slow first boot | never | once |

Both put a usable `module_aiter_core.so` at the path aiter probes, so they are
interchangeable — pick by which constraint you have. They are not byte-for-byte
equivalent operations, though: A drops the single `.so`, while B overlays the
**whole** `jit/` tree, `utils/` and all. That is why the stale-base warning
above bites harder on B — an overlay from an older base can quietly downgrade
aiter's own python alongside the `.so`.

---

## §2 — Persist the compile cache

Point vLLM's caches at a mounted volume:

```yaml
environment:
  VLLM_CACHE_ROOT: /cache/vllm            # traced graphs   (~400 MB here)
  TORCHINDUCTOR_CACHE_DIR: /cache/inductor   # stays empty on this stack; harmless
  TRITON_CACHE_DIR: /cache/triton         # triton kernels  (~90 MB here)
  HF_HUB_OFFLINE: "1"                     # no live Hub calls during load
volumes:
  - ./cache-<name>:/cache:rw
```

First boot on a new directory is still cold. Every boot after is warm.

### The one rule that keeps this from breaking

**One cache directory per (image + model + config). New image, new directory.**

Keep the old directories on disk — a few hundred MB each, and they are your
rollback. Name them for what they hold, e.g. `cache-<model>-<image-tag>`.

Reusing a directory across a change is where "stale cache" start failures come
from. It is also quietly dangerous: engine-specific env knobs are generally
**not** part of vLLM's compile-cache key, so a reused directory does not error —
it replays the graph traced under the *old* settings. Anything you measure after
that is measuring the old config while the logs claim the new one. We have
produced false A/B results exactly this way.

Start a fresh directory whenever any of these move:

- the image, or any compiled `.so` inside it
- the vLLM version (a bump re-keys every traced graph)
- the model checkpoint or its quantization
- KV-cache layout or dtype
- tensor-parallel size, max-model-len, max-num-seqs
- speculative-decoding config

---

## §3 — Don't set the health timeout too low

A cold boot is legitimately several minutes. If the orchestrator's health
timeout is shorter than the cold boot, it **destroys the container mid-load** —
and you get no crash, no traceback, and no logs, which is about the most
confusing failure in this whole stack. Give it generous headroom (a 1800 s start
period here) and tighten it later if you want.

---

## §4 — Check where your time actually goes

Before assuming it is cache, read the boot log:

```bash
docker logs <container> 2>&1 | grep -E \
  "Loading weights took|Model loading took|init engine|Graph capturing|Application startup complete"
```

A healthy **warm** boot here:

```
Loading weights took 33.70 seconds
Model loading took 12.85 GiB memory and 48.80 seconds
Graph capturing finished in 6 secs, took 2.34 GiB
init engine (profile, create kv cache, warmup model) took 37.85 s (compilation: 9.40 s)
```

Read it like this:

| symptom | cause | fix |
|---|---|---|
| `compilation:` is 60–90 s+ | cold graph cache | §2 |
| minutes unaccounted for *before* the worker lines appear | aiter JIT | §1 |
| `Loading weights took` is minutes | slow disk, or a live HF Hub call | `HF_HUB_OFFLINE=1`, weights on local NVMe |

A cold-cache boot with the aiter bake already in place reads ~381 s here, against
251 s warm, so ~6 min on a genuinely new cache directory is expected and not a
fault.
