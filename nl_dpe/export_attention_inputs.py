"""Export the input vectors shared by Q/K/V during TinyBERT inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from nl_dpe.evaluate import convert_examples_to_features, get_task, get_tensor_data
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="sst-2")
    parser.add_argument("--layer", type=int, default=1, help="1-based layer index")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    task = get_task(args.task_name, str(args.data_dir))
    tokenizer = BertTokenizer.from_pretrained(str(args.model_path), do_lower_case=True)
    examples = task.get_dev_examples()
    features = convert_examples_to_features(examples, tokenizer, task)
    data, labels = get_tensor_data(task.output_mode, features)
    loader = DataLoader(
        data,
        sampler=SequentialSampler(data),
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = TinyBertForSequenceClassification.from_pretrained(
        str(args.model_path), num_labels=task.num_labels()
    )
    model.to(device)
    model.eval()
    if not 1 <= args.layer <= model.config.num_hidden_layers:
        raise ValueError(f"Layer must be in [1, {model.config.num_hidden_layers}]")
    attention = model.bert.encoder.layer[args.layer - 1].attention.self

    current_inputs: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, inputs):
            current_inputs[name] = inputs[0].detach()

        return hook

    handles = [
        getattr(attention, name).register_forward_pre_hook(make_hook(name))
        for name in ("query", "key", "value")
    ]

    all_x: list[torch.Tensor] = []
    valid_x: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    qkv_max_difference = 0.0
    try:
        with torch.no_grad():
            for batch in loader:
                input_ids, input_mask, segment_ids, _label_ids, _seq_lengths = (
                    tensor.to(device) for tensor in batch
                )
                current_inputs.clear()
                logits, _, _ = model(input_ids, segment_ids, input_mask)
                if set(current_inputs) != {"query", "key", "value"}:
                    raise RuntimeError(f"Failed to capture all Q/K/V inputs: {current_inputs.keys()}")

                query_x = current_inputs["query"]
                for name in ("key", "value"):
                    difference = float((query_x - current_inputs[name]).abs().max())
                    qkv_max_difference = max(qkv_max_difference, difference)
                if query_x.ndim != 3 or query_x.shape[-1] != 26:
                    raise ValueError(f"Expected [batch, seq, 26] QKV input, got {tuple(query_x.shape)}")

                mask = input_mask.bool()
                all_x.append(query_x.cpu().reshape(-1, 26))
                valid_x.append(query_x[mask].cpu())
                all_masks.append(input_mask.cpu())
                all_ids.append(input_ids.cpu())
                predictions.append(logits.argmax(dim=1).cpu())
    finally:
        for handle in handles:
            handle.remove()

    x_all = torch.cat(all_x).numpy()
    x_valid = torch.cat(valid_x).numpy()
    masks = torch.cat(all_masks).numpy()
    input_ids = torch.cat(all_ids).numpy()
    predicted = torch.cat(predictions).numpy()
    accuracy = float(np.mean(predicted == labels.numpy()))

    np.save(args.output_dir / "X_all_positions.npy", x_all)
    np.save(args.output_dir / "X_valid_tokens.npy", x_valid)
    np.savetxt(args.output_dir / "X_all_positions.csv", x_all, delimiter=",", fmt="%.9g")
    np.savetxt(args.output_dir / "X_valid_tokens.csv", x_valid, delimiter=",", fmt="%.9g")

    valid_rows = np.argwhere(masks == 1)
    valid_ids = input_ids[masks == 1]
    valid_index = np.column_stack((valid_rows, valid_ids))
    np.save(args.output_dir / "valid_token_index.npy", valid_index)
    with (args.output_dir / "valid_token_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("example_index_0_based", "token_position_0_based", "input_id"))
        writer.writerows(valid_index.tolist())

    sequence_length = int(masks.shape[1])
    metadata = {
        "source_checkpoint": str(args.model_path),
        "checkpoint_kind": "clean",
        "task": args.task_name,
        "split": "dev",
        "layer_1_based": args.layer,
        "attention_arch": int(model.config.arch),
        "input_construction": "reshape hidden [312] to [12,26], then mean across 12 heads",
        "qkv_share_identical_input": qkv_max_difference == 0.0,
        "qkv_input_max_abs_difference": qkv_max_difference,
        "examples": int(len(data)),
        "sequence_length": sequence_length,
        "all_positions_shape": list(x_all.shape),
        "valid_tokens_shape": list(x_valid.shape),
        "dtype": str(x_all.dtype),
        "row_order_all_positions": "row = example_index * sequence_length + token_position",
        "row_order_valid_tokens": "example-major, then token-position; see valid_token_index files",
        "clean_dev_accuracy": accuracy,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# SST-2 clean inference Q/K/V input vectors

These activations were captured from Layer {args.layer} of the clean
`{args.model_path}` checkpoint while running the {len(data)} SST-2 dev
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

- `X_all_positions`: shape **{x_all.shape[0]}x26**. This includes all
  {sequence_length} positions per example, including padding, matching the
  model's dense Linear calls. Row `i*{sequence_length}+j` is example `i`,
  position `j`.
- `X_valid_tokens`: shape **{x_valid.shape[0]}x26**. This removes padding and
  keeps `[CLS]`, `[SEP]`, and text tokens.
- `valid_token_index`: maps every row of `X_valid_tokens` to its zero-based
  example index, token position, and tokenizer input ID.

Both X matrices are available as `.npy` and `.csv`. The captured Q/K/V input
maximum absolute difference was **{qkv_max_difference:.3e}**. Clean dev
accuracy during export was **{accuracy:.12g}**.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
