#!/usr/bin/env python3
"""Engine-exact repro for the in-engine n-gram HSA fault (M1).

The old repro tested the *matcher* with a synthetic context; the engine additionally runs a torch
gather (`index_select`) over the UVA host-mapped `all_token_ids` tensor, and passes the window BASE
(not size) to `_nblk`. This reproduces the engine sequence off-server.

Modes:
  sel          : build a UVA host-mapped [reqs, ML] int32 tensor and loop `uva.index_select(0, idx)`
                 (the engine-only op), interleaved with a global-pool graph replay.
  sel_nograph  : same index_select loop, no graph replay.
  host         : the proposed fix -- gather on the host (pinned source of truth) + H2D.
  matcher      : matcher only on an engine-sized device context (no index_select) -- control.

Run inside the image with the GPU free, e.g.:
  docker run --rm --privileged --ipc=host --network=host --device /dev/kfd --device /dev/dri \
    --group-add 993 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES=0 -v "$PWD":/patches:z --entrypoint bash <image> \
    -c 'cd /patches && python3 aijuus/eng_ngram_repro.py --mode sel --iters 200'
"""
import argparse
import sys

sys.path.insert(0, "/patches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sel", "sel_nograph", "host", "matcher"], default="sel")
    ap.add_argument("--ml", type=int, default=160000)
    ap.add_argument("--reqs", type=int, default=8)
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--nspec", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--graph", type=int, default=1, help="1 = interleave a global-pool graph replay")
    a = ap.parse_args()

    import numpy as np
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dev = torch.device("cuda:0")
    torch.manual_seed(1)
    B = 2
    ML = a.ml

    # UVA source exactly like RequestState.all_token_ids: pinned host int32, wide rows.
    host = torch.randint(0, 4096, (a.reqs, ML), dtype=torch.int32, pin_memory=True)
    host[:, ML // 2:] = host[:, : ML // 2]
    uva = get_accelerator_view_from_cpu_tensor(host)
    idx = torch.tensor([0, 1], dtype=torch.int64, device=dev)
    n = np.array([160000, 120000], dtype=np.int32)[:B]
    base = np.maximum(0, n - a.window).astype(np.int32)

    g = None
    if a.graph and a.mode != "sel_nograph":
        # Capture into the SAME global graph pool the engine uses, on the current stream.
        try:
            from vllm.platforms import current_platform
            pool = current_platform.get_global_graph_pool()

            def dummy():
                x = torch.zeros(4096, device=dev)
                return (x + 1).sum()

            dummy()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            cap_stream = torch.cuda.Stream()
            with torch.cuda.graph(g, pool, stream=cap_stream):
                dummy()
            print("[repro] captured dummy graph in the global pool", flush=True)
        except Exception as e:
            print(f"[repro] graph capture unavailable: {e!r}; running without replay", flush=True)
            g = None

    print(f"[repro] mode={a.mode} ml={ML} iters={a.iters} graph={g is not None}", flush=True)
    for it in range(a.iters):
        if g is not None:
            g.replay()
        if a.mode in ("sel", "sel_nograph"):
            ctx = uva.index_select(0, idx)          # <-- engine-only op: torch gather over UVA
        elif a.mode == "host":
            ctx = host[idx.cpu().numpy()].to(dev)   # proposed fix: host gather + H2D
        else:
            ctx = torch.randint(0, 4096, (B, ML), dtype=torch.int32, device=dev)
        if a.mode == "matcher":
            import radiance_draft_gpu as gpu
            nblk = gpu._nblk(int(n[0]), a.window)
            buf = gpu.make_match_buffers(1, a.nspec, 2 * nblk, dev)
            buf["base"].copy_(torch.from_numpy(np.asarray([base[0]], dtype=np.int32)).to(dev))
            pk = gpu.match_gpu(ctx[0:1], torch.from_numpy(n[0:1]).to(dev), a.nspec, int(n[0]),
                               buf["base"], 2 * nblk, False, buf, gpu._MAXL)
            _ = pk.cpu().numpy()
        else:
            _ = ctx.shape
        torch.cuda.synchronize()
        if (it + 1) % 50 == 0:
            print(f"[repro] iter {it + 1}/{a.iters} ok", flush=True)
    print("[repro] PASS: no fault", flush=True)


if __name__ == "__main__":
    main()
