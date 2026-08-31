"""Single-head self-attention (assignment Parts 1 and 2).

Part 1 is the non-causal case; Part 2 adds the causal mask through the same
module by flipping one flag. The tensor plumbing here is deliberately kept flat
so the four steps of attention stay visible before heads are introduced.
"""

import torch.nn as nn

from .functional import scaled_dot_product_attention


class SelfAttention(nn.Module):
    """One head of self-attention over a batch of token embeddings.

    Args:
        embed_dim: size of each token embedding (C).
        causal: if True the head is masked so a token cannot see its future.
        bias: whether the Q/K/V projections carry a bias term.
    """

    def __init__(self, embed_dim, causal=False, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        # Three independent learned views of the same input embedding.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, return_weights=False):
        """Map (B, T, C) token embeddings to (B, T, C) context-aware vectors."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out, weights = scaled_dot_product_attention(q, k, v, causal=self.causal)

        if return_weights:
            return out, weights
        return out
