"""End-to-end demonstration of every property the assignment asks for.

Run with `python demo.py`. Output is deterministic (fixed seed) and every number
is recomputed live, not hard-coded.
"""

import torch

from attention import CausalSelfAttention, SelfAttention, scaled_dot_product_attention


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def part1():
    banner("PART 1  Single-head scaled dot-product attention")
    torch.manual_seed(0)
    attn = SelfAttention(embed_dim=16)
    x = torch.randn(1, 4, 16)
    out, weights = attn(x, return_weights=True)
    print(f"input  {tuple(x.shape)}  ->  output {tuple(out.shape)}")
    print(f"attention weight rows (each sums to 1.0):")
    for i, row in enumerate(weights[0]):
        vals = "  ".join(f"{w:.3f}" for w in row)
        print(f"  token {i}: [{vals}]   sum={row.sum():.6f}")
    print(f"max |rowsum - 1| = {(weights.sum(-1) - 1).abs().max():.2e}")


def part2():
    banner("PART 2  Causal masking")
    torch.manual_seed(0)
    attn = SelfAttention(embed_dim=16, causal=True)
    x = torch.randn(1, 5, 16)
    _, weights = attn(x, return_weights=True)
    print("attention matrix (row = query token, col = key token):")
    for i, row in enumerate(weights[0]):
        vals = "  ".join(f"{w:.2f}" for w in row)
        print(f"  q{i}: [{vals}]")
    print("upper triangle is exactly zero -> no token sees its future")

    extra = torch.randn(1, 3, 16)
    out_short = attn(x)
    out_long = attn(torch.cat([x, extra], dim=1))
    drift = (out_short - out_long[:, :5]).abs().max()
    print(f"appending 3 future tokens changes first 5 outputs by {drift:.2e}")


def part3():
    banner("PART 3  Parallel multi-head causal attention")
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=64, num_heads=8, fused_qkv=False)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    print(f"embed_dim=64, num_heads=8, head_dim={attn.head_dim}")
    print(f"input {tuple(x.shape)}  ->  output {tuple(out.shape)}  (shape preserved)")

    try:
        CausalSelfAttention(embed_dim=30, num_heads=4)
    except AssertionError as e:
        print(f"indivisible split rejected: {e}")

    _, heads_before = attn(x, return_head_outputs=True)
    with torch.no_grad():
        attn.q_proj.weight[0 : attn.head_dim] += 5.0
    _, heads_after = attn(x, return_head_outputs=True)
    moved = (heads_before[:, 0] - heads_after[:, 0]).abs().max()
    leaked = max(
        (heads_before[:, h] - heads_after[:, h]).abs().max().item()
        for h in range(1, attn.num_heads)
    )
    print(f"perturbing head 0's weights: head 0 moved by {moved:.2e}, "
          f"other heads moved by {leaked:.2e}")


def bonus():
    banner("BONUS  KV-cache and weight tying")
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=48, num_heads=6)
    x = torch.randn(1, 8, 48)
    full = attn(x)
    cache, steps = None, []
    for t in range(x.size(1)):
        y_t, cache = attn.step(x[:, t : t + 1], cache)
        steps.append(y_t)
    incremental = torch.cat(steps, dim=1)
    print(f"KV-cache vs full recompute: max diff {(full - incremental).abs().max():.2e}")

    unfused = CausalSelfAttention(embed_dim=32, num_heads=4, fused_qkv=False)
    fused = CausalSelfAttention(embed_dim=32, num_heads=4, fused_qkv=True)
    with torch.no_grad():
        fused.qkv_proj.weight.copy_(torch.cat(
            [unfused.q_proj.weight, unfused.k_proj.weight, unfused.v_proj.weight], dim=0))
        fused.qkv_proj.bias.copy_(torch.cat(
            [unfused.q_proj.bias, unfused.k_proj.bias, unfused.v_proj.bias], dim=0))
        fused.out_proj.weight.copy_(unfused.out_proj.weight)
        fused.out_proj.bias.copy_(unfused.out_proj.bias)
    xi = torch.randn(2, 7, 32)
    print(f"3 linears vs 1 fused linear: max diff {(unfused(xi) - fused(xi)).abs().max():.2e}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    bonus()
    print()
