"""Part 2 correctness criteria: causal masking."""

import torch

from attention import SelfAttention


def test_future_tokens_do_not_change_past_outputs():
    # A token at position i must produce the same output whether or not tokens
    # are appended after it.
    torch.manual_seed(0)
    attn = SelfAttention(embed_dim=16, causal=True)
    x = torch.randn(1, 6, 16)
    extra = torch.randn(1, 4, 16)

    out_short = attn(x)
    out_long = attn(torch.cat([x, extra], dim=1))

    assert torch.allclose(out_short, out_long[:, :6], atol=1e-6)


def test_weights_sum_to_one_including_first_token():
    torch.manual_seed(0)
    attn = SelfAttention(embed_dim=16, causal=True)
    _, weights = attn(torch.randn(2, 7, 16), return_weights=True)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    # The first token attends only to itself: weight 1.0 on position 0.
    assert torch.allclose(weights[:, 0, 0], torch.ones(2), atol=1e-6)


def test_upper_triangle_receives_exactly_zero_weight():
    torch.manual_seed(0)
    attn = SelfAttention(embed_dim=16, causal=True)
    _, weights = attn(torch.randn(1, 5, 16), return_weights=True)
    future = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    # Not merely small: exactly zero, because e^(-inf) == 0.
    assert (weights[0][future] == 0).all()
