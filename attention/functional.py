"""Scaled dot-product attention, written from raw tensor operations.

This single function is the computational core shared by every module in the
package. It operates on the last two dimensions of its inputs and broadcasts
over any number of leading dimensions, so the identical code serves the
single-head case (leading dims `(B,)`) and the multi-head case (leading dims
`(B, n_head)`) without change.
"""

import math

import torch


def scaled_dot_product_attention(q, k, v, causal=False):
    """Compute attention outputs and weights for one set of queries/keys/values.

    Args:
        q: queries, shape (..., T_q, d_k).
        k: keys, shape (..., T_k, d_k).
        v: values, shape (..., T_k, d_v).
        causal: if True, a query at position i may not attend to any key at a
            position j > i. Assumes self-attention, i.e. T_q == T_k.

    Returns:
        out: shape (..., T_q, d_v), the attention output.
        weights: shape (..., T_q, T_k), the attention probability distribution.
    """
    d_k = q.size(-1)

    # Step 1 (Score): align every query with every key. The transpose swaps the
    # last two axes only, so the batch/head axes stay put.
    scores = q @ k.transpose(-2, -1)  # (..., T_q, T_k)

    # Step 2 (Scale): divide by sqrt(d_k). Dot products grow with dimension;
    # left unscaled they push softmax into a near-flat-gradient regime.
    scores = scores / math.sqrt(d_k)

    # Step 3 (Mask, optional): overwrite future positions with -inf so that,
    # after exponentiation, they contribute exactly zero weight.
    if causal:
        t_q, t_k = scores.size(-2), scores.size(-1)
        allowed = torch.tril(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~allowed, float("-inf"))

    # Step 3 (Softmax): turn each row of scores into a probability distribution
    # over the keys. Every row sums to 1.
    weights = torch.softmax(scores, dim=-1)

    # Step 4 (Weighted sum): blend the values by those probabilities.
    out = weights @ v  # (..., T_q, d_v)

    return out, weights
