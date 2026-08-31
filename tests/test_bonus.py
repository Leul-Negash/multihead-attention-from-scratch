"""Bonus criteria: KV-caching and weight tying (fused QKV)."""

import torch

from attention import CausalSelfAttention


def test_kv_cache_matches_full_recompute():
    # Feeding tokens one at a time through the cache must reproduce the output
    # of a single full-sequence forward pass.
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=48, num_heads=6)
    x = torch.randn(2, 8, 48)

    full = attn(x)

    cache = None
    steps = []
    for t in range(x.size(1)):
        y_t, cache = attn.step(x[:, t : t + 1], cache)
        steps.append(y_t)
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)


def test_fused_qkv_is_numerically_identical():
    # One (C -> 3C) projection must give the same result as three (C -> C)
    # projections holding the same weights.
    torch.manual_seed(0)
    unfused = CausalSelfAttention(embed_dim=32, num_heads=4, fused_qkv=False)
    fused = CausalSelfAttention(embed_dim=32, num_heads=4, fused_qkv=True)

    with torch.no_grad():
        fused.qkv_proj.weight.copy_(
            torch.cat([unfused.q_proj.weight, unfused.k_proj.weight, unfused.v_proj.weight], dim=0)
        )
        fused.qkv_proj.bias.copy_(
            torch.cat([unfused.q_proj.bias, unfused.k_proj.bias, unfused.v_proj.bias], dim=0)
        )
        fused.out_proj.weight.copy_(unfused.out_proj.weight)
        fused.out_proj.bias.copy_(unfused.out_proj.bias)

    x = torch.randn(2, 7, 32)
    assert torch.allclose(unfused(x), fused(x), atol=1e-6)
