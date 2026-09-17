"""GPU unit test for radiance's paroquant_mxfp6: load -> prep -> linear vs an fp32 reference.

The MXFP6 twin of test_mxfp4_loader.py, and the same contract: a real module's on-disk buffers
through the served path, checked against a straightforward fp32 dequant of Quark's own fp6 bytes
(nibble plane | 2-bit plane -> E2M3 magnitude x 2^(E-127)). Tolerance is the e4m3 activation
quantization, not the weight grid. Covers every M band (decode, folded, A-tiled), the merged
single launch against the per-partition loop, and the fused per-token producers (rms+rot,
silu-mul, attn-gate, gdn-norm) against the plain path. PQM_TP=2 emulates one rank's shard.

    podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
      --group-add keep-groups -e HIP_VISIBLE_DEVICES=0 -e RADIANCE_PAROQUANT=1 \
      -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_MXFP4_DECODE_MAX_M=64 \
      -v ~/deadcode-vllm:/patches -v ~/models:/models --entrypoint bash \
      stilldeadcode/vllm-radiance:0.9.3 -lc 'cd /patches && <build both .so into site-packages as
      run_paroquant.sh does> && PQM_CKPT=/models/Qwen3.8-27B-PARO-MXFP6 \
      python3 paroquant/test_mxfp6_loader.py'
"""
import json
import os
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, "/patches/paroquant")
import radiance_paroquant_mxfp4 as M   # noqa: E402

CKPT = os.environ.get("PQM_CKPT") or "/models/Qwen3.8-27B-PARO-MXFP6"
TP = int(os.environ.get("PQM_TP", "1") or 1)
E4M3_MAX = 448.0


def _weight_map():
    import glob
    idx = f"{CKPT}/model.safetensors.index.json"
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for fn in glob.glob(f"{CKPT}/*.safetensors"):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                wm[k] = os.path.basename(fn)
    return wm


def load_module(name):
    idx = _weight_map()
    out = {}
    for leaf in ("weight", "weight_scale", "theta", "pairs", "channel_scales"):
        with safe_open(f"{CKPT}/{idx[f'{name}.{leaf}']}", framework="pt") as f:
            out[leaf] = f.get_tensor(f"{name}.{leaf}")
    return out


def dequant_w(packed, e8m0):
    b = packed.reshape(packed.shape[0], -1, 3).to(torch.int16)
    v0 = b[..., 0] & 0x3F
    v1 = ((b[..., 1] & 0x0F) << 2) | (b[..., 0] >> 6)
    v2 = ((b[..., 2] & 0x03) << 4) | (b[..., 1] >> 4)
    v3 = b[..., 2] >> 2
    codes = torch.stack([v0, v1, v2, v3], -1).reshape(packed.shape[0], -1)
    E, Mn = ((codes >> 3) & 3).float(), (codes & 7).float()
    mag = torch.where(E == 0, Mn / 8, torch.exp2(E - 1) * (1 + Mn / 8))
    val = torch.where((codes & 0x20) != 0, -mag, mag)
    sc = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return val * sc


def rotate_ref(x, pairs, theta, cs_stored):
    x = (x.float() * cs_stored.float().view(1, -1)).clone()
    K = x.shape[1]
    for r in range(pairs.shape[0]):
        for g in range(K // 128):
            b = g * 128
            idx = pairs[r, b:b + 128].long()
            th = theta[r, b // 2:(b + 128) // 2].float()
            i, j = idx[0::2] + b, idx[1::2] + b
            c, s = torch.cos(th), torch.sin(th)
            xi, xj = x[:, i].clone(), x[:, j].clone()
            x[:, i] = xi * c + xj * s
            x[:, j] = -xi * s + xj * c
    return x


def main():
    torch.manual_seed(0)
    dev = "cuda"
    mods_env = os.environ.get("PQM_MODULES") or "linear_attn.in_proj_qkv,linear_attn.in_proj_z"
    layer_idx = int(os.environ.get("PQM_LAYER", "0") or 0)
    rank = int(os.environ.get("PQM_RANK", "0") or 0)
    assert 0 <= rank < TP, f"PQM_RANK={rank} outside TP={TP}"
    names = [f"model.language_model.layers.{layer_idx}.{m}" for m in mods_env.split(",")]
    mods = [load_module(n) for n in names]
    row_parallel = all(m.rsplit(".", 1)[-1] in ("o_proj", "down_proj", "out_proj") for m in mods_env.split(","))
    if TP > 1 and not row_parallel:
        for m in mods:
            rows = m["weight"].shape[0] // TP
            m["weight"] = m["weight"][rank * rows:(rank + 1) * rows].contiguous()
            m["weight_scale"] = m["weight_scale"][rank * rows:(rank + 1) * rows].contiguous()
    elif TP > 1:
        for m in mods:
            for leaf in ("weight", "weight_scale", "theta", "pairs", "channel_scales"):
                t = m[leaf]
                w = t.shape[-1] // TP
                assert w * TP == t.shape[-1], f"{leaf} {tuple(t.shape)} not divisible by TP={TP}"
                m[leaf] = t.narrow(-1, rank * w, w).contiguous()
    print(f"checkpoint {CKPT} | layer {layer_idx} | modules {mods_env} | TP-emulation {TP} rank {rank} "
          f"({'row' if row_parallel else 'column'}-parallel)")
    K = mods[0]["weight_scale"].shape[1] * 32
    sizes = [m["weight"].shape[0] for m in mods]
    N = sum(sizes)
    print(f"merged linear: K={K} N={N} partitions={sizes}")

    layer = nn.Module()
    layer.weight = nn.Parameter(torch.cat([m["weight"] for m in mods]), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.cat([m["weight_scale"] for m in mods]), requires_grad=False)
    layer.theta = nn.Parameter(torch.stack([m["theta"] for m in mods]), requires_grad=False)
    layer.pairs = nn.Parameter(torch.stack([m["pairs"] for m in mods]), requires_grad=False)
    layer.channel_scales = nn.Parameter(torch.stack([m["channel_scales"].reshape(-1) for m in mods]),
                                        requires_grad=False)
    layer.pq_output_partition_sizes = list(sizes)
    cfg = M.ParoQuantMXFP6Config.from_config({"quant_method": "paroquant_mxfp6", "format": "mxfp6_e2m3",
                                              "bits": 6, "group_size": 128, "mx_block": 32, "krot": 8,
                                              "max_exp_spread": 6})
    method = M.ParoQuantMXFP6LinearMethod(cfg)
    layer.to(dev)
    method.process_weights_after_loading(layer)
    print(f"prepared: P={layer.rec.shape[0]} pb1={layer.pq_pb1} pb2={layer.pq_pb2} "
          f"weight={tuple(layer.weight.shape)} wh={tuple(layer.wh.shape)} ws_t={tuple(layer.ws_t.shape)} "
          f"wref={tuple(layer.wref.shape)}")
    assert layer.rec.shape[0] == len(mods), "distinct rotations must NOT dedup"
    assert tuple(layer.weight.shape) == (N, K // 2) and tuple(layer.wh.shape) == (N, K // 4)

    bounds = [0]
    for s_ in sizes:
        bounds.append(bounds[-1] + s_)
    for p, m in enumerate(mods):
        n0, n1 = bounds[p], bounds[p + 1]
        xq = (torch.randn(7, K, device=dev) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
        s = torch.rand(7, device=dev) + 0.5
        ref6 = M._mx._exact_ref6(xq, s, layer.weight[n0:n1], layer.wh[n0:n1],
                                 layer.ws_t[:, n0:n1].contiguous(), n1 - n0, K)
        w = dequant_w(m["weight"].to(dev), m["weight_scale"].to(dev))
        ind = (xq.float() * s.view(-1, 1)) @ w.T
        rel = ((ref6 - ind).norm() / ind.norm()).item()
        print(f"  _exact_ref6 vs independent dequant, part {p}: rel={rel:.2e}")
        assert rel < 1e-6, f"part {p}: _exact_ref6 disagrees with the Quark-bytes dequant (rel {rel})"

    for Mrows in [int(x) for x in (os.environ.get("PQM_MS") or "1,5,40,64,200,600,2048").split(",")]:
        x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        y = method.apply(layer, x).float()
        ref = torch.empty(Mrows, N, device=dev)
        for p, m in enumerate(mods):
            xr = rotate_ref(x, m["pairs"].to(dev), m["theta"].to(dev), m["channel_scales"].to(dev))
            s = xr.abs().amax(1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
            xq = (xr / s).to(torch.float8_e4m3fn).float() * s
            w = dequant_w(m["weight"].to(dev), m["weight_scale"].to(dev))
            ref[:, bounds[p]:bounds[p + 1]] = xq @ w.T
        rel = ((y - ref).norm() / ref.norm()).item()
        band = ("decode" if Mrows <= M._mx.DECODE_MAX_M else
                "A-tiled" if M._mx.A_TILED_MIN_M and Mrows >= M._mx.A_TILED_MIN_M else "prefill")
        parts = " ".join(f"p{p}={((y[:, a:b] - ref[:, a:b]).norm() / ref[:, a:b].norm()).item():.4f}"
                         for p, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])))
        print(f"  M={Mrows:4d} {band:7s} rel={rel:.4f}  [{parts}]  finite={bool(torch.isfinite(y).all())}")
        assert rel < 4e-2, f"M={Mrows}: rel {rel} too large -- layout or scale mismatch"
    print("PASS: paroquant_mxfp6 linear matches fp32 reference at every band")
    print(f"single launch vs per-partition loop (P={len(mods)}):")
    for Mrows in [1, 5, 8, 40, 64, 200, 600, 2048]:
        x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        M.SINGLE_LAUNCH = True;  y1 = method.apply(layer, x)
        M.SINGLE_LAUNCH = False; y0 = method.apply(layer, x)
        M.SINGLE_LAUNCH = True
        rel = ((y1.float() - y0.float()).norm() / y0.float().norm()).item()
        nd = int((y0 != y1).sum())
        print(f"  M={Mrows:4d}: single vs loop rel={rel:.2e}  differing elems {nd}/{y0.numel()}")
        assert rel < 1e-3, f"M={Mrows}: single-launch output differs from the per-partition loop (rel {rel})"
    print("PASS: single-launch merged GEMM matches the per-partition loop to bf16 rounding")

    print("rotation stream (per-token producers) vs plain path on the producer's own hs:")
    wn = (torch.randn(K, device=dev) * 0.1).to(torch.bfloat16)
    for Mrows in [1, 5, 8, 40, 64, 200, 600]:
        y = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        res = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        hs, ro, a, as_tok = torch.ops.radiance.pqm_add_rms_rot(y, res, wn, 1e-6, layer.rec, layer.cs)
        ro_ref = (y.float() + res.float()).to(torch.bfloat16)
        assert torch.equal(ro, ro_ref), f"M={Mrows}: residual out differs"
        v = (y.float() + res.float())
        hs_ref = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + wn.float())).to(torch.bfloat16)
        hs_rel = ((hs.float() - hs_ref.float()).norm() / hs_ref.float().norm()).item()
        assert hs_rel < 2e-3, f"M={Mrows}: hs rel {hs_rel} vs torch rmsnorm"
        y_pre = method.apply(layer, (hs, a, as_tok))
        y_plain = method.apply(layer, hs)
        same = torch.equal(y_pre, y_plain)
        print(f"  norm site M={Mrows:4d} {'tiled' if M._tiled(Mrows) else 'row-major'}: pre == plain {same}  hs rel {hs_rel:.2e}")
        assert same, f"M={Mrows}: stream tuple output differs from the plain path"
    l1 = nn.Module()
    l1.weight = nn.Parameter(mods[0]["weight"].clone(), requires_grad=False)
    l1.weight_scale = nn.Parameter(mods[0]["weight_scale"].clone(), requires_grad=False)
    l1.theta = nn.Parameter(mods[0]["theta"].unsqueeze(0).clone(), requires_grad=False)
    l1.pairs = nn.Parameter(mods[0]["pairs"].unsqueeze(0).clone(), requires_grad=False)
    l1.channel_scales = nn.Parameter(mods[0]["channel_scales"].reshape(1, -1).clone(), requires_grad=False)
    l1.pq_output_partition_sizes = [sizes[0]]
    l1.to(dev)
    method.process_weights_after_loading(l1)
    w128 = (torch.randn(128, device=dev) * 0.5).to(torch.bfloat16)
    for mode, label in ((0, "silu-mul"), (1, "attn-gate"), (2, "gdn-norm")):
        for Mrows in [1, 8, 64, 600]:
            if mode == 0:
                x = (torch.randn(Mrows, 2 * K, device=dev) * 0.8).to(torch.bfloat16)
                yy = x
                g, u = x[:, :K].float(), x[:, K:].float()
                hs_ref = ((g / (1 + torch.exp(-g))).to(torch.bfloat16).float() * u).to(torch.bfloat16)
            else:
                x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
                yy = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
                if mode == 1:
                    hs_ref = (x.float() * (1 / (1 + torch.exp(-yy.float()))).to(torch.bfloat16).float()).to(torch.bfloat16)
                else:
                    xv = x.float().view(Mrows, -1, 128)
                    n = xv * torch.rsqrt(xv.pow(2).mean(-1, keepdim=True) + 1e-6) * w128.float()
                    z = yy.float().view(Mrows, -1, 128)
                    hs_ref = (n * (z / (1 + torch.exp(-z)))).view(Mrows, K).to(torch.bfloat16)
            hs, a, as_tok = torch.ops.radiance.pqm_ew_rot(mode, x, yy, w128 if mode == 2 else l1.cs, 1e-6, l1.rec, l1.cs)
            hs_rel = ((hs.float() - hs_ref.float()).norm() / hs_ref.float().norm()).item()
            y_pre = method.apply(l1, (hs, a, as_tok))
            y_plain = method.apply(l1, hs)
            same = torch.equal(y_pre, y_plain)
            print(f"  {label:9s} M={Mrows:4d}: pre == plain {same}  hs rel {hs_rel:.2e}")
            assert same, f"{label} M={Mrows}: stream tuple output differs from the plain path"
            assert hs_rel < 5e-3, f"{label} M={Mrows}: hs rel {hs_rel} vs torch reference"
    print("PASS: per-token rotation-stream producers are output-identical to the plain path")


if __name__ == "__main__":
    main()
