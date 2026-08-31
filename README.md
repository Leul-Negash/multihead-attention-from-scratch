# Multi-Head Attention from Scratch

Causal multi-head self-attention built only from `nn.Linear` and raw tensor
operations (`matmul`, `softmax`, `tril`, `view`/`transpose`/`reshape`). No
`nn.MultiheadAttention`, no `F.scaled_dot_product_attention`, no `einops`.

## Layout

| File | Role |
| --- | --- |
| `attention/functional.py` | `scaled_dot_product_attention` — the four-step core (score, scale, mask, softmax, weighted sum), shared by every module. |
| `attention/single_head.py` | `SelfAttention` — one head, with an optional causal flag (Parts 1 and 2). |
| `attention/multi_head.py` | `CausalSelfAttention` — parallel heads, KV-cache, and fused/tied QKV (Part 3 and both bonuses). |
| `demo.py` | Prints every invariant with live numbers. |
| `tests/` | One test file per assignment part, asserting each stated correctness criterion. |

## How the assignment maps to the code

- **Part 1 — scaled dot-product attention:** `scaled_dot_product_attention`
  with `causal=False`, wrapped by `SelfAttention`.
- **Part 2 — causal masking:** the same core with `causal=True`; future
  positions are set to `-inf` before softmax.
- **Part 3 — parallel multi-head:** `CausalSelfAttention` splits the embedding
  across heads, runs the core batched over the head axis, and recombines with
  one output projection.
- **Bonus — KV-cache:** `CausalSelfAttention.step` processes one token at a time
  from cached keys/values.
- **Bonus — weight tying:** `fused_qkv=True` uses a single `C -> 3C` projection.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py        # end-to-end demonstration
python -m pytest -q   # 15 tests, one group per part
```
