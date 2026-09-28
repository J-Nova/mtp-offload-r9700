# aijuus overlay

Everything personal lives here. Upstream files are never edited, so
`git fetch && git rebase origin/main` never conflicts.

## Layout

- `patches/` — diffs against upstream-owned files (Dockerfile, serve-mxfp4.sh,
  radiance_*.py, docs, …). Applied on demand, never committed.
- personal additions — model router/controller/registry, compose, ops/, tools/,
  bench/, plans. These are new paths and cannot conflict with upstream.

## Flow

```sh
aijuus/revert.sh                       # before touching upstream history
git fetch && git rebase origin/main    # pull upstream freely
aijuus/apply.sh                        # re-apply the overlay for build/run
```

The deployment bind-mounts the repo root at `/patches` and reads personal files
via the `aijuus/` prefix (see `aijuus/coolify-compose-2gpu.yml`); the registry is
at `/patches/aijuus/model-registry.json` inside the container.
