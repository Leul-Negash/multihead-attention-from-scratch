"""Causal multi-head self-attention (assignment Part 3 and both bonuses).

`CausalSelfAttention` runs several masked attention heads in parallel over
independent slices of the embedding, then recombines them with a single output
projection. The two bonus features are folded in as options rather than forks of
the code:

* `fused_qkv=True` collapses the three Q/K/V projections into one `nn.Linear`
  (weight tying); it is numerically identical to the three-layer form.
* `step()` runs the module incrementally with a K/V cache, one token at a time,
  and reproduces the full-sequence output exactly.
"""

import torch
import torch.nn as nn

from .functional import scaled_dot_product_attention


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Args:
        embed_dim: total embedding size C, split evenly across heads.
        num_heads: number of parallel heads; must divide embed_dim.
        fused_qkv: if True use one (C -> 3C) projection instead of three.
        bias: whether the projections carry bias terms.
    """

    def __init__(self, embed_dim, num_heads, fused_qkv=True, bias=True):
        super().__init__()
        # Fail loudly: an indivisible split would silently drop or overlap
        # dimensions and corrupt every head.
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fused_qkv = fused_qkv

        if fused_qkv:
            # One projection producing [Q | K | V] stacked along the last axis.
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Blends information across heads after they are concatenated.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _project_qkv(self, x):
        """Return Q, K, V, each shaped (B, T, C), from input (B, T, C)."""
        if self.fused_qkv:
            qkv = self.qkv_proj(x)
            c = self.embed_dim
            # Pure slicing along the feature axis; no data is moved.
            return qkv[..., :c], qkv[..., c:2 * c], qkv[..., 2 * c:]
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _split_heads(self, t):
        """(B, T, C) -> (B, n_head, T, head_dim).

        The embedding is split with C as the *last* contiguous axis, so a token's
        C features are cut into `num_heads` neighbouring blocks. The transpose
        then moves the head axis in front of the token axis so each head is an
        independent (T, head_dim) matrix for the batched matmul.
        """
        b, seq, _ = t.shape
        t = t.view(b, seq, self.num_heads, self.head_dim)
        return t.transpose(1, 2)

    def _merge_heads(self, t):
        """(B, n_head, T, head_dim) -> (B, T, C).

        `contiguous()` is required: the transpose leaves the tensor with a
        head-major memory layout, and `view` refuses to reinterpret that without
        a real copy back to token-major order.
        """
        b, _, seq, _ = t.shape
        t = t.transpose(1, 2).contiguous()
        return t.view(b, seq, self.embed_dim)

    def forward(self, x, return_head_outputs=False):
        """Map (B, T, C) to (B, T, C) with masked multi-head attention."""
        q, k, v = self._project_qkv(x)

        q = self._split_heads(q)  # (B, n_head, T, head_dim)
        k = self._split_heads(k)
        v = self._split_heads(v)

        head_out, _ = scaled_dot_product_attention(q, k, v, causal=True)

        merged = self._merge_heads(head_out)
        out = self.out_proj(merged)

        if return_head_outputs:
            # Per-head attention outputs, before mixing, for the independence test.
            return out, head_out
        return out

    @torch.no_grad()
    def step(self, x_t, cache=None):
        """Process one new token using cached keys/values.

        Args:
            x_t: the new token embedding, shape (B, 1, C).
            cache: dict with prior 'k' and 'v' of shape (B, n_head, T_prev,
                head_dim), or None at the start of a sequence.

        Returns:
            y_t: output for the new token, shape (B, 1, C).
            cache: updated cache including this token.
        """
        q, k, v = self._project_qkv(x_t)
        q = self._split_heads(q)  # (B, n_head, 1, head_dim)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)  # grow along the time axis
            v = torch.cat([cache["v"], v], dim=2)

        # No mask needed: every cached key is at a position <= the new token, so
        # causality already holds for a single query attending over the cache.
        head_out, _ = scaled_dot_product_attention(q, k, v, causal=False)

        y_t = self.out_proj(self._merge_heads(head_out))
        return y_t, {"k": k, "v": v}
