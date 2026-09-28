#!/usr/bin/env python3
"""Self-test for the fused int2 draft-head top-1 kernel (B2).

Verifies that the fused path (_draft_head_int2_top1) produces the same top-1 id and
confidence as the standard path (_draft_head_int2 + capture + argmax). The exact winner
is always a block maximum, so argmax(Y) == idx[argmax(ex)] whenever the winner is among
the rescored candidates.

Run on the R9700 box:
    docker run --rm --privileged --ipc=host --network=host --device /dev/kfd --device /dev/dri \
    --group-add 993 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 -v <repo>:/patches:z --entrypoint bash \
    stilldeadcode/vllm-radiance:0.9.3 -lc 'cd /patches && python3 drafthead_top1_selftest.py'
"""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radiance_drafthead as dh

# Test parameters
N = 248320          # vocab size (Qwen3.8)
K = 5120            # hidden size
GROUP = dh.GROUP    # 128
KCAND = dh.KCAND    # 8
RERANK = dh.RERANK  # 32
BLOCK_N = dh.BLOCK_N  # 64

def make_quantized_head(n, k, device, seed=42):
    """Create a quantized int2 head with random weights for testing."""
    torch.manual_seed(seed)
    # Random bf16 weights
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.01
    
    # Quantize to int2 with quarter-split packing
    BITS = 2
    per_byte = 8 // BITS
    lv = (1 << BITS) - 1
    packed = torch.empty(n, k // per_byte, dtype=torch.uint8, device=device)
    scale = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=device)
    zs = torch.empty(n, k // GROUP, dtype=torch.bfloat16, device=device)
    CH = 8192
    for i in range(0, n, CH):
        j = min(i + CH, n)
        wg = w[i:j].to(torch.float32).reshape(j - i, k // GROUP, GROUP)
        lo, hi = wg.amin(dim=2), wg.amax(dim=2)
        sc = ((hi - lo) / lv).clamp(min=1e-8)
        zp = torch.round(-lo / sc).clamp(0, lv)
        q = torch.round(wg / sc[:, :, None] + zp[:, :, None]).clamp(0, lv).to(torch.uint8)
        q = q.reshape(j - i, k)
        Q = k // 4
        packed[i:j] = (q[:, :Q] | (q[:, Q:2 * Q] << 2)
                       | (q[:, 2 * Q:3 * Q] << 4) | (q[:, 3 * Q:] << 6))
        bias = 4.0
        scale[i:j] = (bias * sc).to(torch.bfloat16)
        zs[i:j] = (bias * sc + zp * sc).to(torch.bfloat16)
    
    return packed, scale, zs, w


def test_fused_top1(M, n, k, packed, scale, zs, w, device):
    """Test that the fused path produces the same top-1 id and confidence as the standard path."""
    torch.manual_seed(123)
    x = torch.randn(M, k, dtype=torch.bfloat16, device=device) * 0.01
    
    # Pad to power of 2 >= 16
    Mp = dh._pow2_at_least(M)
    if Mp != M:
        x = torch.cat([x, x.new_zeros(Mp - M, k)])
    x = x.contiguous()
    
    ng = k // GROUP
    xs = x.reshape(Mp, ng, GROUP).float().sum(-1).contiguous()
    nblk = (n + BLOCK_N - 1) // BLOCK_N
    
    # Standard path: _draft_head_int2
    y = torch.empty(Mp, n, dtype=torch.bfloat16, device=device)
    bm_std = torch.empty(Mp, nblk * KCAND, dtype=torch.float32, device=device)
    bi_std = torch.empty(Mp, nblk * KCAND, dtype=torch.int32, device=device)
    cfg = dh._cfg_for(Mp)
    dh._draft_head_int2[(nblk,)](
        x, xs, packed, scale, zs, y, bm_std, bi_std,
        k, n, packed.stride(0), scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=Mp, BLOCK_N=BLOCK_N, **cfg)
    
    # Rerank standard path
    idx_std = bi_std.gather(1, bm_std.topk(RERANK, dim=1).indices).contiguous()
    ex_std = torch.empty(Mp, RERANK, dtype=torch.float32, device=device)
    dh._rerank_exact[(Mp, RERANK)](x, w, x, idx_std, ex_std, k, w.stride(0),
                                    R=RERANK, BLOCK_K=512, FP8=False, num_warps=4)
    y_std = y.clone()
    y_std.scatter_(1, idx_std.long(), ex_std.to(torch.bfloat16))
    
    # Get standard top-1 id and confidence
    y_std_fp32 = y_std.float()
    argmax_std = y_std_fp32.argmax(dim=1)
    max_std = y_std_fp32.max(dim=1).values
    sum_exp_std = torch.exp(y_std_fp32 - max_std.unsqueeze(1)).sum(dim=1)
    conf_std = torch.exp(max_std - max_std) / sum_exp_std  # = 1 / sum_exp
    
    # Fused path: _draft_head_int2_top1
    bm_fused = torch.empty(Mp, nblk * KCAND, dtype=torch.float32, device=device)
    bi_fused = torch.empty(Mp, nblk * KCAND, dtype=torch.int32, device=device)
    sm_fused = torch.empty(Mp, nblk, dtype=torch.float32, device=device)
    dh._draft_head_int2_top1[(nblk,)](
        x, xs, packed, scale, zs, bm_fused, bi_fused, sm_fused,
        k, n, packed.stride(0), scale.stride(0), xs.stride(0),
        nblk, KCAND, G=GROUP, BLOCK_M=Mp, BLOCK_N=BLOCK_N, **cfg)
    
    # Rerank fused path
    idx_fused = bi_fused.gather(1, bm_fused.topk(RERANK, dim=1).indices).contiguous()
    ex_fused = torch.empty(Mp, RERANK, dtype=torch.float32, device=device)
    dh._rerank_exact[(Mp, RERANK)](x, w, x, idx_fused, ex_fused, k, w.stride(0),
                                    R=RERANK, BLOCK_K=512, FP8=False, num_warps=4)
    
    # Get fused top-1 id and confidence
    ex_fused_masked = torch.where(idx_fused < n, ex_fused, float("-inf"))
    argmax_ex = ex_fused_masked.argmax(dim=1)
    ids_fused = idx_fused.gather(1, argmax_ex.unsqueeze(1)).squeeze(1)
    
    blockmax = bm_fused.view(Mp, nblk, KCAND)[:, :, 0]
    Mrow = blockmax.max(dim=1).values
    S = (sm_fused * torch.exp(blockmax - Mrow.unsqueeze(1))).sum(dim=1)
    max_ex = ex_fused_masked.max(dim=1).values
    conf_fused = torch.exp(max_ex - Mrow) / S
    
    # Compare
    ids_match = (ids_fused == argmax_std).all().item()
    conf_diff = (conf_fused - conf_std).abs().max().item()
    
    return ids_match, conf_diff, ids_fused[:5].tolist(), argmax_std[:5].tolist()


def main():
    device = torch.device("cuda")
    print(f"Testing fused int2 draft-head top-1 (B2) on {device}")
    print(f"N={N}, K={K}, GROUP={GROUP}, KCAND={KCAND}, RERANK={RERANK}, BLOCK_N={BLOCK_N}")
    
    # Create quantized head
    print("Creating quantized head...")
    packed, scale, zs, w = make_quantized_head(N, K, device)
    
    # Test different M values
    all_pass = True
    for M in [1, 5, 9, 16, 32, 64, 72]:
        print(f"\nM={M}:")
        try:
            ids_match, conf_diff, ids_f, ids_s = test_fused_top1(M, N, K, packed, scale, zs, w, device)
            print(f"  IDs match: {ids_match}")
            print(f"  Conf diff: {conf_diff:.6f}")
            print(f"  Fused IDs: {ids_f}")
            print(f"  Std IDs:   {ids_s}")
            if not ids_match:
                all_pass = False
                print(f"  FAILED: IDs do not match!")
            elif conf_diff > 0.01:
                all_pass = False
                print(f"  FAILED: Confidence difference too large!")
            else:
                print(f"  PASSED")
        except Exception as e:
            all_pass = False
            print(f"  FAILED: {e!r}")
    
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
