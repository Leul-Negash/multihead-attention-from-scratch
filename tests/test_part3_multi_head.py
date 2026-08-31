"""Part 3 correctness criteria: parallel multi-head causal attention."""

import pytest
import torch

from attention import CausalSelfAttention


def test_indivisible_embed_dim_fails_loudly():
    with pytest.raises(AssertionError):
        CausalSelfAttention(embed_dim=30, num_heads=4)


def test_output_shape_matches_input():
    attn = CausalSelfAttention(embed_dim=64, num_heads=8)
    x = torch.randn(3, 9, 64)
    assert attn(x).shape == x.shape


def test_heads_are_independent():
    # Perturbing the weights that feed one head must not change any other
    # head's attention output (measured before the mixing output projection).
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=32, num_heads=4, fused_qkv=False)
    x = torch.randn(2, 6, 32)

    _, heads_before = attn(x, return_head_outputs=True)

    head_dim = attn.head_dim
    with torch.no_grad():
        # Rows [0:head_dim] of q_proj produce head 0's query features only.
        attn.q_proj.weight[0:head_dim] += 5.0

    _, heads_after = attn(x, return_head_outputs=True)

    assert not torch.allclose(heads_before[:, 0], heads_after[:, 0])  # head 0 moved
    for h in range(1, attn.num_heads):
        assert torch.allclose(heads_before[:, h], heads_after[:, h], atol=1e-6)


def test_causality_holds_for_the_multi_head_module():
    torch.manual_seed(0)
    attn = CausalSelfAttention(embed_dim=32, num_heads=4)
    x = torch.randn(1, 5, 32)
    extra = torch.randn(1, 3, 32)
    out_short = attn(x)
    out_long = attn(torch.cat([x, extra], dim=1))
    assert torch.allclose(out_short, out_long[:, :5], atol=1e-6)


def test_generalises_across_head_configurations():
    for embed_dim, num_heads in ((32, 1), (64, 8), (128, 16)):
        attn = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(2, 10, embed_dim)
        assert attn(x).shape == x.shape
