"""Map signed x@W matrices to ideal differential-pair conductances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MATRIX_NAMES = ("WQ", "WK", "WV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g-min", type=float, default=0.0)
    parser.add_argument("--g-max", type=float, default=150.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.g_min != 0.0:
        raise ValueError("This one-sided differential mapping currently requires g_min=0.")
    if args.g_max <= args.g_min:
        raise ValueError("g_max must be greater than g_min.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    metadata: dict[str, object] = {
        "source_weights": str(args.weights_dir),
        "source_checkpoint_kind": "clean",
        "source_matrix_equation": "y = x @ W",
        "conductance_unit": "uS",
        "conductance_range_uS": [args.g_min, args.g_max],
        "mapping": "Gpositive=max(W,0)*scale; Gnegative=max(-W,0)*scale",
        "reconstruction": "W=(Gpositive-Gnegative)/scale",
        "differential_layout": (
            "26x52 interleaved output columns: physical column 2*j is left Gpositive; "
            "physical column 2*j+1 is right Gnegative"
        ),
        "scaling_scope": "one scalar per 26x26 matrix",
        "matrices": {},
    }
    readme_rows: list[str] = []

    for name in MATRIX_NAMES:
        weight = np.load(args.weights_dir / f"{name}.npy")
        if weight.shape != (26, 26):
            raise ValueError(f"Expected {name} to have shape (26, 26), got {weight.shape}")
        weight64 = weight.astype(np.float64)
        max_abs = float(np.abs(weight64).max())
        if max_abs == 0.0:
            raise ValueError(f"Cannot scale all-zero matrix {name}")
        scale = (args.g_max - args.g_min) / max_abs

        gpositive = np.maximum(weight64, 0.0) * scale + args.g_min
        gnegative = np.maximum(-weight64, 0.0) * scale + args.g_min
        paired = np.empty((26, 52), dtype=np.float64)
        paired[:, 0::2] = gpositive
        paired[:, 1::2] = gnegative

        reconstructed = (gpositive - gnegative) / scale
        max_error = float(np.abs(reconstructed - weight64).max())
        if gpositive.min() < args.g_min or gnegative.min() < args.g_min:
            raise RuntimeError(f"{name} conductance fell below g_min")
        if gpositive.max() > args.g_max + 1e-10 or gnegative.max() > args.g_max + 1e-10:
            raise RuntimeError(f"{name} conductance exceeded g_max")

        for suffix, array in (
            ("Gpositive", gpositive),
            ("Gnegative", gnegative),
            ("differential_interleaved", paired),
        ):
            np.save(args.output_dir / f"{name}_{suffix}.npy", array)
            np.savetxt(
                args.output_dir / f"{name}_{suffix}.csv",
                array,
                delimiter=",",
                fmt="%.12g",
            )

        matrix_metadata = {
            "weight_max_abs": max_abs,
            "scale_uS_per_weight": scale,
            "inverse_scale_weight_per_uS": 1.0 / scale,
            "gpositive_max_uS": float(gpositive.max()),
            "gnegative_max_uS": float(gnegative.max()),
            "reconstruction_max_abs_error": max_error,
            "logical_shape": [26, 26],
            "physical_interleaved_shape": [26, 52],
        }
        metadata["matrices"][name] = matrix_metadata  # type: ignore[index]
        readme_rows.append(
            f"| {name} | {max_abs:.12g} | {scale:.12g} | "
            f"{1.0 / scale:.12g} | {max_error:.3e} |"
        )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# Clean Head-12 differential conductance mapping

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
{chr(10).join(readme_rows)}

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
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
