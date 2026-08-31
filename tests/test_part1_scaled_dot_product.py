"""Part 1 correctness criteria: scaled dot-product attention, single head."""

import math

import torch

from attention import SelfAttention, scaled_dot_product_attention


def test_weights_form_a_probability_distribution():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    _, weights = scaled_dot_product_attention(q, k, v)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert (weights >= 0).all()


def test_core_broadcasts_over_leading_dims():
    # The same function must serve single-head (B, T, d) and multi-head
    # (B, n_head, T, d) shapes unchanged.
    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 8)
    out, weights = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 4, 6, 8)
    assert weights.shape == (2, 4, 6, 6)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-6)


def test_uses_three_separate_projections():
    attn = SelfAttention(embed_dim=16)
    # Q/K/V come from distinct weight matrices, not one shared view.
    assert attn.q_proj.weight.data_ptr() != attn.k_proj.weight.data_ptr()
    assert attn.k_proj.weight.data_ptr() != attn.v_proj.weight.data_ptr()


def test_generalises_across_embedding_dims():
    for dim in (32, 64, 128):
        attn = SelfAttention(embed_dim=dim)
        x = torch.randn(3, 7, dim)
        out, weights = attn(x, return_weights=True)
        assert out.shape == (3, 7, dim)
        assert torch.allclose(weights.sum(-1), torch.ones(3, 7), atol=1e-6)


def test_scaling_matches_the_sqrt_dk_definition():
    # Independent recomputation of the scaled scores confirms the 1/sqrt(d_k)
    # factor is applied exactly, not approximated or skipped.
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 3, 8), torch.randn(1, 3, 8), torch.randn(1, 3, 8)
    _, weights = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(8), dim=-1)
    assert torch.allclose(weights, expected, atol=1e-6)
