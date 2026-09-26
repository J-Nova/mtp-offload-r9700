"""MXFP6-E2M3 RTN builder: bf16 base weights + z-lab's TRAINED rotations -> MXFP6, one shot.

The MXFP6 twin of build_hybrid.py, and it reuses that script's finding: rotations that tame
per-group outliers for int4 do the same for E2M3, so only the weight grid changes. Quantization is
paroquant.optim.mxfp6 (added to the paroquant source by paroquant_radiance.patch), whose codes
are raw-identical to amd-quark's mxfp6 when MAX_EXP_SPREAD is off. The clamp is the one addition:
it raises every block exponent in a row to within N binades of the row's largest, because the
radiance kernel folds that exponent into the e4m3 weight byte and the fold is exact only 6
binades down -- MAX_EXP_SPREAD=off writes a checkpoint the serving loader then refuses.

  SCALE_RULE=even|ocp|mse2  the shared-exponent rule handed to quantize_to_codes (default even)
  MAX_EXP_SPREAD=6|off      the per-row clamp above
  WRITE_PSEUDO=1            also write the fp16 rotate -> quantize -> inverse-rotate checkpoint,
                            which loads on the stock path -- the accuracy gate

    BASE=/models/Qwen3.8-27B-bf16 PARO=/models/Qwen3.8-27B-PARO \
    OUT_REAL=/models/Qwen3.8-27B-PARO-MXFP6-rtn python3 paroquant/build_mxfp6.py

Runs inside the paroquant source container (needs paroquant.optim.mxfp6 and the rotation kernel);
requant.sh FORMAT=mxfp6 STAGE=finetune is the fine-tuned path to the same format.
"""
import json, os, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, "/src")
from paroquant.kernels.cuda import scaled_pairwise_rotation
from paroquant.optim.mxfp6 import quantize_to_codes, dequantize_from_codes

GS, MX_BLOCK = 128, 32
CH = 1024
dev = "cuda"
BASE = Path(os.environ.get("BASE", "/models/Qwen3.8-27B-bf16"))
PARO = Path(os.environ.get("PARO", "/models/Qwen3.8-27B-PARO"))
OUT_REAL = Path(os.environ.get("OUT_REAL", "/models/Qwen3.8-27B-PARO-MXFP6-rtn"))
WRITE_PSEUDO = os.environ.get("WRITE_PSEUDO", "0") == "1"
OUT_PSEUDO = Path(os.environ.get("OUT_PSEUDO", "/models/Qwen3.8-27B-PARO-MXFP6-rtn-pseudo"))
SCALE_RULE = (os.environ.get("SCALE_RULE") or "even").strip().lower()
assert SCALE_RULE in ("even", "ocp", "mse2"), f"SCALE_RULE must be even, ocp or mse2, got {SCALE_RULE!r}"
_spread = (os.environ.get("MAX_EXP_SPREAD") or "6").strip().lower()
MAX_EXP_SPREAD = None if _spread in ("off", "none") else int(_spread)
assert MAX_EXP_SPREAD is None or MAX_EXP_SPREAD >= 0, f"MAX_EXP_SPREAD must be >= 0 or off, got {_spread!r}"


def rot(w, pairs, theta):
    return scaled_pairwise_rotation(w, pairs, theta, None, GS)


def unrot(w, pairs, theta):
    return scaled_pairwise_rotation(w, torch.flip(pairs, [0]), -torch.flip(theta, [0]), None, GS)


def main():
    if MAX_EXP_SPREAD is None or MAX_EXP_SPREAD > 6:
        print(f"WARNING: MAX_EXP_SPREAD={MAX_EXP_SPREAD}: the radiance E2M3 fold is exact only to 6 and its "
              "loader rejects wider rows -- this checkpoint is for format studies, not serving", flush=True)
    cfg = json.load(open(PARO / "config.json"))
    OUT_REAL.mkdir(parents=True, exist_ok=True)
    if WRITE_PSEUDO: OUT_PSEUDO.mkdir(parents=True, exist_ok=True)

    paro = safe_open(PARO / "model.safetensors", framework="pt")
    quantized = {k[:-len(".theta")] for k in paro.keys() if k.endswith(".theta")}
    krot = {paro.get_slice(f"{m}.pairs").get_shape()[0] for m in quantized}
    assert len(krot) == 1, krot
    krot = krot.pop()

    index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    expect = {m for m in quantized if not m.startswith("mtp.")}
    missing = sorted(m for m in expect if f"{m}.weight" not in index)
    assert not missing, f"{len(missing)} rotation modules have no base weight, e.g. {missing[:2]}"
    shards = sorted({s for n, s in index.items() if not n.startswith("mtp.")})
    print(f"z-lab quantized modules: {len(quantized)} (building {len(expect)}) | mxfp6 e2m3 rule {SCALE_RULE} "
          f"spread<={MAX_EXP_SPREAD} krot {krot} | {dev}", flush=True)
    pseudo_map, real_map = {}, {}
    n_q = 0; worst_rt = 0.0; worst_spread = 0; qerr2 = qnorm2 = 0.0
    t_rot = t_q = 0.0; t0 = time.time()

    for si, shard in enumerate(shards):
        pseudo_t, real_t = {}, {}
        with safe_open(BASE / shard, framework="pt") as f:
            for name in f.keys():
                if name.startswith("mtp."): continue
                mod = name[:-len(".weight")] if name.endswith(".weight") else None
                if mod in quantized:
                    w = f.get_tensor(name).to(dev, torch.float32)
                    N, K = w.shape
                    pairs = paro.get_tensor(f"{mod}.pairs").to(dev)
                    theta = paro.get_tensor(f"{mod}.theta").to(dev, torch.float32)
                    cs_opt = (1.0 / paro.get_tensor(f"{mod}.channel_scales").to(dev, torch.float32)).view(1, -1)
                    ta = time.time()
                    w_rot = rot(w * cs_opt, pairs, theta); del w
                    tb = time.time(); t_rot += tb - ta
                    packed, e8m0 = [], []
                    for lo in range(0, N, CH):
                        p, e = quantize_to_codes(w_rot[lo:lo + CH], rule=SCALE_RULE, max_spread=MAX_EXP_SPREAD)
                        packed.append(p); e8m0.append(e)
                    packed = torch.cat(packed); e8m0 = torch.cat(e8m0)
                    ws = e8m0.to(torch.int16)
                    worst_spread = max(worst_spread, int((ws.amax(dim=1, keepdim=True) - ws).max()))
                    w_rot_q = dequantize_from_codes(packed, e8m0)
                    qerr2 += float((w_rot_q - w_rot).double().pow(2).sum()); qnorm2 += float(w_rot.double().pow(2).sum())
                    t_q += time.time() - tb
                    real_t[f"{mod}.weight"] = packed.cpu()
                    real_t[f"{mod}.weight_scale"] = e8m0.cpu()
                    for leaf in ("theta", "pairs", "channel_scales"):
                        real_t[f"{mod}.{leaf}"] = paro.get_tensor(f"{mod}.{leaf}")
                    if WRITE_PSEUDO:
                        w_pseudo = unrot(w_rot_q, pairs, theta) / cs_opt
                        rt = ((rot(w_pseudo * cs_opt, pairs, theta) - w_rot_q).norm() / w_rot_q.norm()).item()
                        worst_rt = max(worst_rt, rt)
                        pseudo_t[name] = w_pseudo.to(torch.float16).cpu()
                    del w_rot, w_rot_q
                    n_q += 1
                else:
                    t = f.get_tensor(name); t = t.to(torch.float16) if t.is_floating_point() else t
                    real_t[name] = t
                    if WRITE_PSEUDO: pseudo_t[name] = t
        save_file(real_t, OUT_REAL / shard, metadata={"format": "pt"})
        for k in real_t: real_map[k] = shard
        if WRITE_PSEUDO:
            save_file(pseudo_t, OUT_PSEUDO / shard, metadata={"format": "pt"})
            for k in pseudo_t: pseudo_map[k] = shard
        del pseudo_t, real_t
        torch.cuda.empty_cache()
        print(f"  shard {si+1}/{len(shards)} {shard}  quantized so far {n_q}  rel err {(qerr2 / max(qnorm2, 1e-30)) ** 0.5:.5f}"
              f"  worst spread {worst_spread}  worst round-trip {worst_rt:.2e}"
              f"  {time.time()-t0:.0f}s (rotate {t_rot:.0f}s, quantize {t_q:.0f}s)", flush=True)

    assert n_q == len(expect), (n_q, len(expect))
    cfg.pop("torch_dtype", None)
    outs = [(OUT_REAL, real_map)] + ([(OUT_PSEUDO, pseudo_map)] if WRITE_PSEUDO else [])
    for out, wmap in outs:
        json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
        for fn in PARO.iterdir():
            if fn.suffix in (".json", ".jinja", ".txt") and fn.name != "model.safetensors.index.json":
                shutil.copy(fn, out / fn.name)
    qc = dict(cfg.get("quantization_config", {}))
    qc.update({"quant_method": "paroquant_mxfp6", "format": "mxfp6_e2m3",
               "bits": 6, "group_size": GS, "krot": int(krot)})
    cfg_r = dict(cfg); cfg_r["quantization_config"] = qc
    json.dump(cfg_r, open(OUT_REAL / "config.json", "w"), indent=2)
    if WRITE_PSEUDO:
        cfg_p = dict(cfg); cfg_p.pop("quantization_config", None)
        json.dump(cfg_p, open(OUT_PSEUDO / "config.json", "w"), indent=2)
    print(f"DONE: {n_q} modules -> {OUT_REAL}, rel err {(qerr2 / max(qnorm2, 1e-30)) ** 0.5:.5f}, worst spread "
          f"{worst_spread}, worst pseudo round-trip {worst_rt:.2e}, "
          f"{time.time()-t0:.0f}s (rotate {t_rot:.0f}s, quantize {t_q:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
