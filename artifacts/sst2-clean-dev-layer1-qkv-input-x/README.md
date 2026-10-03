# SST-2 clean inference Q/K/V input vectors

These activations were captured from Layer 1 of the clean
`models/wise-tern-584` checkpoint while running the 872 SST-2 dev
examples. Q, K, and V receive the same input tensor, so only one X matrix is
needed for all three projections.

The model uses the average-hidden-states attention architecture:

```text
hidden [batch, sequence, 312]
  -> reshape [batch, sequence, 12, 26]
  -> mean over 12 heads
X [batch, sequence, 26]
```

With the previously exported matrices:

```text
Q = X @ WQ
K = X @ WK
V = X @ WV
```

Bias values are omitted from these equations.

## Files

- `X_all_positions`: shape **55808x26**. This includes all
  64 positions per example, including padding, matching the
  model's dense Linear calls. Row `i*64+j` is example `i`,
  position `j`.
- `X_valid_tokens`: shape **21943x26**. This removes padding and
  keeps `[CLS]`, `[SEP]`, and text tokens.
- `valid_token_index`: maps every row of `X_valid_tokens` to its zero-based
  example index, token position, and tokenizer input ID.

Both X matrices are available as `.npy` and `.csv`. The captured Q/K/V input
maximum absolute difference was **0.000e+00**. Clean dev
accuracy during export was **0.917431192661**.
