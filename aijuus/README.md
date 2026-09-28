# aijuus overlay

Everything personal lives here. Upstream files are never edited, so
`git fetch && git rebase origin/main` never conflicts.

## Two different things called "patches"

1. **Overlay patches** — `aijuus/patches/*.patch`: diffs against upstream-owned
   files (Dockerfile, serve-mxfp4.sh, radiance_*.py, docs, …). Applied on demand
   by `aijuus/apply.sh`, never committed. These are **build-time** (baked into the
   image), except two files the entrypoint also re-copies at runtime.
2. **Runtime patch scripts** — `aijuus/patch_*.py` (plus upstream `patch_*.py`):
   executed by the vLLM entrypoint from the `/patches` mount on every container
   start. Idempotent; editing one needs only a container restart, no rebuild.

## Layout

- `patches/` — overlay diffs (see above).
- personal additions — model router/controller/registry, compose, ops/, bench/,
  plans, and the runtime `patch_*.py`.

## Deploy flow

`aijuus/deploy.sh` is the single entry point (it can't build a stale overlay):

```sh
aijuus/deploy.sh check                 # read-only: overlay / branch / IMAGE / runtime state
aijuus/deploy.sh apply                 # apply the overlay to the working tree
aijuus/deploy.sh build [build.sh args] # apply overlay, then ./build.sh …
aijuus/deploy.sh full  [build.sh args] # apply overlay, then ./build.sh --push …
```

End to end:

```sh
aijuus/deploy.sh apply                              # so the image bakes your changes
aijuus/deploy.sh full --registry=<host/path>        # build + push
# then in Coolify: set IMAGE=<registry>/ggz14/vllm-radiance-mxfp4:<VERSION>-<SHA>, redeploy
```

The vLLM services in `coolify-compose-2gpu.yml` pin the published base
(`image: stilldeadcode/vllm-radiance:0.9.3`). If you do build, point Coolify at the
built tag — set the vLLM services' `image:` (or a Coolify `IMAGE` env that the
compose templates) to the tag `build.sh` prints
(`ggz14/vllm-radiance-mxfp4:<VERSION>-<SHA>`, or the pushed `<registry>/…` form).

## Coolify never builds — don't rebuild unless you must

Coolify only pulls the image and runs the compose; the vLLM entrypoint then
applies `aijuus/patch_*.py`, copies the runtime files, and compiles the kernel at
every boot. So:

- **No rebuild** (redeploy/restart is enough): `aijuus/patch_*.py`, `patch_*.py`,
  compose/env, `model-registry.json`, `model-router.py`, `model-controller.py`,
  and overlay `062`/`063` (apply the overlay + restart).
- **Rebuild only for** build-time files: Dockerfile / base image (overlay `010`/
  `011`, or `build.sh --base-only`/`--full`), and baked-only files `060`/`061`
  (`radiance_draft.py`, `radiance_draft_gpu.py`) and `070`/`071` (paroquant). Even
  then, baking those is a boot-speed/robustness choice, not a correctness one.
- **No deploy impact at all**: docs/ops overlay patches `020`, `030`-`032`,
  `040`-`042`, `050`, `051`, `080`.

`aijuus/deploy.sh check` prints this rule; `build`/`full` warn before rebuilding.

## When to apply / rebuild / restart

| Change | Action |
|---|---|
| `aijuus/patch_*.py`, `patch_*.py`, compose/env, model registry/router/controller | restart/redeploy containers — no apply, no rebuild |
| overlay `062-radiance-drafthead` / `063-radiance-preamble` | `apply.sh` + restart vLLM — no rebuild |
| overlay on baked/ops files (`060`/`061` radiance_draft*.py, `050`/`051` fp8/quantize, `080` serve-mxfp4.sh, `010`/`011` Dockerfile, `030`-`032` ops, docs, `070`/`071` paroquant) | `apply.sh` + rebuild + push + point Coolify `IMAGE` at the new tag |

## Upstream sync

```sh
aijuus/revert.sh                       # before touching upstream history
git fetch && git rebase origin/main    # pull upstream freely
aijuus/apply.sh                        # re-apply the overlay for build/run
```

## Gotchas

- Coolify never rebuilds: a plain deploy only pulls the image and runs the
  runtime scripts. Build-time overlay changes reach production only through
  `apply.sh` → `build.sh` → push → Coolify `IMAGE`.
- Keep the overlay applied while deployed. The entrypoint re-copies
  `radiance_preamble.py` and `radiance_drafthead.py` from the live tree at every
  boot, so reverting silently undoes `062`/`063` in the running container on the
  next restart (`aijuus/deploy.sh check` warns about this).
- `build.sh` bakes the working tree as-is: it warns on a dirty tree but applies
  nothing. Use `aijuus/deploy.sh build|full`, which runs `apply.sh` first.

The deployment bind-mounts the repo root at `/patches` and reads personal files
via the `aijuus/` prefix (see `aijuus/coolify-compose-2gpu.yml`); the registry is
at `/patches/aijuus/model-registry.json` inside the container.
