#!/usr/bin/env python3
"""Local patch (ours, 2026-09-28): make vLLM's AOT compile-cache key see the RADIANCE_* environment.

vLLM 0.29 keys torch_aot_compile/{hash} on the vLLM config + the top-level forward only; on load it re-checks the
SOURCE of every traced file, but never the environment. Our plugins decide the traced graph from env at install time
(rotation streams, the fp8-stream contract, GDN no-copy, ...), so flipping one of those with unchanged sources LOADS
THE OLD GRAPH silently -- found 2026-09-28: stream 1 looked inert because radiance_arnq's zero-epilogue install
overwrote the layer forwards, and any A/B of such a switch on a warm cache measured the stale artifact.

Patch: the hash dir becomes {hash}-{env12}, env12 = sha256 of the sorted RADIANCE_* variables (first 12 hex). A new
dir is SEEDED with a copy of {hash}/inductor_cache (the base dir only, never a sibling experiment's) before first use, so pieces
whose FX graph did not change hit the inductor/autotune cache and compile to the same kernels as before (a fresh
inductor cache re-autotunes and can change reduction orders, i.e. numerics).

Gated: RADIANCE_LOCAL_AOT_ENVKEY=1 applies it; unset leaves the file untouched."""
import os
import sys
import sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_LOCAL_AOT_ENVKEY", "0") != "1":
    sys.exit(0)

def patch_decorators():
  F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "compilation" / "decorators.py"
  src = F.read_text()
  if "patch_aot_envkey" in src:
    print("[aot-envkey] decorators.py already applied")
    return

  OLD = (
      "            cache_dir = os.path.join(\n"
      "                envs.VLLM_CACHE_ROOT,\n"
      "                \"torch_compile_cache\",\n"
      "                \"torch_aot_compile\",\n"
      "                hash_key,\n"
      "            )\n"
  )
  NEW = (
      "            cache_dir = os.path.join(\n"
      "                envs.VLLM_CACHE_ROOT,\n"
      "                \"torch_compile_cache\",\n"
      "                \"torch_aot_compile\",\n"
      "                hash_key,\n"
      "            )\n"
      "            # patch_aot_envkey.py: key the artifact on the RADIANCE_* env too; seed a new dir's inductor cache\n"
      "            _rad_env = sorted((k, v) for k, v in os.environ.items() if k.startswith(\"RADIANCE_\"))\n"
      "            _rad_ek = hashlib.sha256(str(_rad_env).encode()).hexdigest()[:12]\n"
      "            _rad_base = cache_dir\n"
      "            cache_dir = f\"{_rad_base}-{_rad_ek}\"\n"
      "            if not os.path.isdir(os.path.join(cache_dir, \"inductor_cache\")):\n"
      "                import shutil as _rad_shutil\n"
      "                _rad_seed = os.path.join(_rad_base, \"inductor_cache\")\n"
      "                if not os.path.isdir(_rad_seed):\n"
      "                    _rad_seed = None\n"
      "                if _rad_seed is not None:\n"
      "                    _rad_shutil.copytree(_rad_seed, os.path.join(cache_dir, \"inductor_cache\"), dirs_exist_ok=True)\n"
      "                logger.warning(\"patch_aot_envkey: new AOT dir %s (env key %s), inductor cache seeded from %s\",\n"
      "                               cache_dir, _rad_ek, _rad_seed)\n"
      "            else:\n"
      "                logger.info(\"patch_aot_envkey: AOT dir %s (env key %s)\", cache_dir, _rad_ek)\n"
  )
  n = src.count(OLD)
  if n != 1:
    print(f"[aot-envkey] anchor found {n} times, NOT applied", file=sys.stderr)
    return
  if "import hashlib" not in src:
    print("[aot-envkey] decorators.py has no hashlib import, NOT applied", file=sys.stderr)
    return
  F.write_text(src.replace(OLD, NEW))
  print("[aot-envkey] applied: torch_aot_compile/{hash}-{RADIANCE_* env key}, inductor cache seeded from the base dir")


# ---- 2026-09-28 (day): the SECOND cache. vLLM's piecewise backend keeps compiled graph pieces in
# torch_compile_cache/<10-char hash>/rank_R_D/{backbone,eagle_head}, keyed on [vLLM env, config, traced code, compiler] and
# LOADED BY PIECE INDEX. An env-driven graph variant (MXFP4 vs fp8 drafter linears, GDN no-copy, streams ...) with unchanged
# sources got another variant's compiled pieces: "copy_misaligned_inputs: Expected tensors only, but got int" (the drafter,
# 2026-09-28) and, most likely, the GPU memory faults of the night before (kernels run with another graph's arguments).
def patch_backends():
  B = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "compilation" / "backends.py"
  bsrc = B.read_text()
  if "patch_aot_envkey" in bsrc:
    print("[aot-envkey] backends.py already applied")
    return
  BOLD = "            factors = [env_hash, config_hash, code_hash, compiler_hash]\n"
  BNEW = ("            factors = [env_hash, config_hash, code_hash, compiler_hash]\n"
          "            # patch_aot_envkey.py: env-driven graph variants must not share compiled pieces\n"
          "            factors.append(str(sorted((k, v) for k, v in os.environ.items() if k.startswith(\"RADIANCE_\"))))\n")
  if bsrc.count(BOLD) != 1:
    print(f"[aot-envkey] backends.py anchor found {bsrc.count(BOLD)} times, NOT applied", file=sys.stderr)
    return
  B.write_text(bsrc.replace(BOLD, BNEW))
  print("[aot-envkey] applied: vLLM piecewise compile cache dir keyed on the RADIANCE_* env too")


patch_decorators()
patch_backends()
