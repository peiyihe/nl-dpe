# Clean Head-12 differential conductance mapping

These files map the SST-2 clean checkpoint's Layer 1, Head 12 WQ/WK/WV
matrices to ideal differential-pair conductances. No device noise and no
noise-aware weights are included.

## Matrix orientation

The source matrices have shape 26x26 and use row-vector multiplication:

```text
Q = x @ WQ
K = x @ WK
V = x @ WV
```

## Mapping

Each matrix is scaled independently over the full 26x26 array:

```text
scale = (150 uS - 0 uS) / max(abs(W))
Gpositive = max(W, 0) * scale
Gnegative = max(-W, 0) * scale
W = (Gpositive - Gnegative) / scale
```

The conductance range is **0 to 150 uS**. For every logical output column
`j`, physical column `2*j` is the **left Gpositive** column and physical
column `2*j+1` is the **right Gnegative** column. The interleaved physical
array therefore has shape 26x52.

| Matrix | max(abs(W)) | scale (uS/weight) | inverse scale (weight/uS) | max reconstruction error |
|---|---:|---:|---:|---:|
| WQ | 0.502138614655 | 298.722296239 | 0.00334759076436 | 5.551e-17 |
| WK | 0.496498942375 | 302.11544718 | 0.00330999294917 | 2.776e-17 |
| WV | 0.606589138508 | 247.284348627 | 0.00404392759005 | 5.551e-17 |

## Files

- `W*_Gpositive.*`: separate 26x26 positive conductance array.
- `W*_Gnegative.*`: separate 26x26 negative conductance array.
- `W*_differential_interleaved.*`: 26x52 physical array with adjacent
  `[Gpositive(left), Gnegative(right)]` column pairs.
- Every array is provided as both NumPy `.npy` and comma-separated `.csv`.

For a row-vector input `x`, reconstruct the logical weight multiplication as:

```text
Ipositive = x @ Gpositive
Inegative = x @ Gnegative
y = (Ipositive - Inegative) / scale
```

Bias values are not included.
