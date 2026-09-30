#!/usr/bin/env python3
"""Standalone repro for the n-gram matcher HSA fault at bs>=2 (WORKLOG cont.28 residual 1).

`RADIANCE_DRAFT_NGRAM=1` faults at the bs=1->bs=2 transition with
HSA_STATUS_ERROR_EXCEPTION merely by running `match_gpu` during a bs>=2 draft step,
independent of draft-row width and matcher launch shape. This drives `match_gpu` off-server
so the faulty kernel can be isolated without wedging a serving instance.

Run with the target GPU free (serving stopped) inside the image, e.g.:

  docker run --rm --privileged --ipc=host --network=host --device /dev/kfd --device /dev/dri \
    --group-add 993 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES=0 -v "$PWD":/patches:z \
    --entrypoint python3 stilldeadcode/vllm-radiance:0.9.3 /patches/aijuus/match_gpu_repro.py --iters 200

Exit 0 = no fault; nonzero/branch identifies the failing batch size. Compare B=1 (proven safe)
against B=2, then the kernel-level variants (cross=0/1, horizon, window) to localise it.
"""
import argparse
import sys

sys.path.insert(0, "/patches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--nspec", type=int, default=8)
    ap.add_argument("--cross", type=int, default=0)
    ap.add_argument("--maxl", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2, help="batch size to stress (1 is the known-safe path)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", type=int, default=4096, help="token id range for the synthetic context")
    ap.add_argument("--invalid", type=int, default=0,
                    help="inject invalid ids into the context: 0 none, 1 negatives (-1,-2), "
                         "2 out-of-vocab (>= vocab*4), 3 both")
    ap.add_argument("--no-sync", type=int, default=0,
                    help="1 = do not torch.cuda.synchronize() each iter (mimic queued graph replay)")
    a = ap.parse_args()

    import numpy as np
    import torch

    import radiance_draft_gpu as gpu

    torch.manual_seed(a.seed)
    dev = torch.device("cuda:0")
    B = a.bs

    # A repeat-heavy context so the matcher actually finds suffixes (else it early-exits and
    # never touches the gather path that is suspected).
    ctx = torch.randint(0, a.vocab, (B, a.ctx_len), dtype=torch.int64, device=dev)
    ctx[:, a.ctx_len // 2:] = ctx[:, : a.ctx_len // 2]  # second half repeats the first
    # patch_mamba_repro-style invalid-id injection: the engine's token history can carry -1
    # padding / out-of-range ids, which would make the matcher's gather read out of bounds.
    if a.invalid in (1, 3):
        ctx[0, ::97] = -1
        if B > 1:
            ctx[1, ::89] = -2
    if a.invalid in (2, 3):
        ctx[0, 5::101] = a.vocab * 4 + 7
        if B > 1:
            ctx[1, 7::103] = a.vocab * 8 + 11
    n = torch.full((B,), a.ctx_len, dtype=torch.int32, device=dev)
    base = torch.zeros(B, dtype=torch.int32, device=dev)

    nblk = gpu._nblk(a.ctx_len, 0)
    bufs = gpu.make_match_buffers(B, a.nspec, max(1, 2 * nblk), dev)

    print(f"[repro] B={B} nspec={a.nspec} cross={a.cross} maxl={a.maxl} nc={2 * nblk} iters={a.iters}",
          flush=True)
    out = None
    for it in range(a.iters):
        pk = gpu.match_gpu(ctx, n, a.nspec, a.ctx_len, base, 2 * nblk, bool(a.cross), bufs, a.maxl)
        if a.no_sync:
            if it == a.iters - 1:
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
        if it == 0:
            out = pk.detach().cpu().numpy()
        if (it + 1) % 50 == 0:
            print(f"[repro] iter {it + 1}/{a.iters} ok", flush=True)
    print("[repro] PASS: no fault; pack shape", out.shape, "meta tail", out[0, 2 * a.nspec:], flush=True)
    print("[repro] compare B=1 vs B=2 pack rows for a kernel-level difference", flush=True)


if __name__ == "__main__":
    main()
