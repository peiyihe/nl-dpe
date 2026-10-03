"""Export one TinyBERT attention head as three standalone matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformer.modeling import TinyBertForSequenceClassification


MATRICES = (("WQ", "query"), ("WK", "key"), ("WV", "value"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1, help="1-based layer index")
    parser.add_argument("--head", type=int, default=12, help="1-based head index")
    parser.add_argument(
        "--orientation",
        choices=("x-matmul", "pytorch"),
        default="x-matmul",
        help="x-matmul exports W for y=x@W; pytorch exports nn.Linear [out,in] weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    model = TinyBertForSequenceClassification.from_pretrained(
        str(args.model_path), num_labels=2
    )
    config = model.config
    if not 1 <= args.layer <= config.num_hidden_layers:
        raise ValueError(f"Layer must be in [1, {config.num_hidden_layers}]")
    if not 1 <= args.head <= config.num_attention_heads:
        raise ValueError(f"Head must be in [1, {config.num_attention_heads}]")

    head_dim = config.hidden_size // config.num_attention_heads
    start = (args.head - 1) * head_dim
    end = start + head_dim
    attention = model.bert.encoder.layer[args.layer - 1].attention.self

    args.output_dir.mkdir(parents=True)
    verification: dict[str, float] = {}
    generator = torch.Generator().manual_seed(20261003)
    x = torch.randn(8, head_dim, generator=generator)
    for short_name, module_name in MATRICES:
        linear = getattr(attention, module_name)
        native = linear.weight[start:end, :].detach().cpu()
        if tuple(native.shape) != (26, 26):
            raise ValueError(
                f"Expected {short_name} head block to be 26x26, got {tuple(native.shape)}"
            )
        exported = native.T.contiguous() if args.orientation == "x-matmul" else native
        array = exported.numpy()
        np.save(args.output_dir / f"{short_name}.npy", array)
        np.savetxt(args.output_dir / f"{short_name}.csv", array, delimiter=",")

        expected = F.linear(x, native, bias=None)
        actual = x @ exported if args.orientation == "x-matmul" else F.linear(x, exported)
        verification[short_name] = float((expected - actual).abs().max())

    equation = "y = x @ W" if args.orientation == "x-matmul" else "y = x @ W.T"
    metadata = {
        "source_checkpoint": str(args.model_path),
        "checkpoint_kind": "clean",
        "layer_1_based": args.layer,
        "head_1_based": args.head,
        "pytorch_output_row_slice_0_based": [start, end],
        "matrix_shape": [26, 26],
        "orientation": args.orientation,
        "equation_without_bias": equation,
        "bias_included": False,
        "files": [f"{name}.npy" for name, _ in MATRICES],
        "max_abs_equivalence_error": verification,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
