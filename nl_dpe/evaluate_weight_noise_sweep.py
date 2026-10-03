"""Evaluate 26x26 attention matrices under normalized Gaussian weight noise.

The same standard-normal noise samples and the same clean-weight reference
scales are used for every checkpoint.  This makes checkpoint comparisons a
paired experiment: the only intended difference is the nominal checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from nl_dpe.evaluate import convert_examples_to_features, get_task, get_tensor_data
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


MATRIX_NAMES = ("query", "key", "value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean and noise-aware 26x26 attention matrices under "
            "Gaussian noise scaled to the clean matrices' maximum weights."
        )
    )
    parser.add_argument("--clean-model", type=Path, required=True)
    parser.add_argument(
        "--task-name",
        choices=("cola", "sst-2", "mrpc"),
        default="cola",
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Checkpoint to evaluate. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1, help="1-based encoder layer.")
    parser.add_argument("--head", type=int, default=12, help="1-based attention head.")
    parser.add_argument(
        "--noise-percentages",
        default="0,2,4,6,8,10,12,14,16,18,20",
        help="Comma-separated Gaussian standard deviations as percentages of max |clean weight|.",
    )
    parser.add_argument(
        "--seeds",
        default="1001,1002,1003,1004,1005",
        help="Comma-separated seeds used for every nonzero noise percentage.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_number_list(value: str, cast: type) -> list:
    result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one comma-separated value.")
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_model(
    path: Path,
    device: torch.device,
    num_labels: int,
) -> TinyBertForSequenceClassification:
    model = TinyBertForSequenceClassification.from_pretrained(
        str(path), num_labels=num_labels
    )
    model.to(device)
    model.eval()
    return model


def selected_matrices(
    model: TinyBertForSequenceClassification,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    config = model.config
    if not 1 <= layer <= config.num_hidden_layers:
        raise ValueError(f"Layer {layer} is outside [1, {config.num_hidden_layers}].")
    if not 1 <= head <= config.num_attention_heads:
        raise ValueError(f"Head {head} is outside [1, {config.num_attention_heads}].")
    head_dim = config.hidden_size // config.num_attention_heads
    start = (head - 1) * head_dim
    end = start + head_dim
    attention = model.bert.encoder.layer[layer - 1].attention.self
    matrices: dict[str, torch.Tensor] = {}
    for name in MATRIX_NAMES:
        linear = getattr(attention, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected {name} to be nn.Linear, got {type(linear).__name__}.")
        matrix = linear.weight[start:end, :]
        if tuple(matrix.shape) != (26, 26):
            raise ValueError(f"Expected a 26x26 {name} matrix, got {tuple(matrix.shape)}.")
        matrices[name] = matrix
    return matrices


def make_loader(
    task_name: str,
    clean_model_path: Path,
    data_dir: Path,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, np.ndarray, int]:
    task = get_task(task_name, str(data_dir))
    tokenizer = BertTokenizer.from_pretrained(str(clean_model_path), do_lower_case=True)
    features = convert_examples_to_features(task.get_dev_examples(), tokenizer, task)
    data, labels = get_tensor_data(task.output_mode, features)
    loader = DataLoader(
        data,
        sampler=SequentialSampler(data),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, labels.numpy(), task.num_labels()


def evaluate_model(
    model: TinyBertForSequenceClassification,
    loader: DataLoader,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    logits_parts: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids, input_mask, segment_ids, label_ids, _ = (
                tensor.to(device) for tensor in batch
            )
            logits, _, _ = model(input_ids, segment_ids, input_mask)
            loss_sum += F.cross_entropy(logits, label_ids, reduction="sum").item()
            logits_parts.append(logits.cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    predictions = logits.argmax(axis=1)
    return {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "accuracy": float(np.mean(predictions == labels)),
        "eval_loss": float(loss_sum / len(labels)),
        "positive_rate": float(np.mean(predictions == 1)),
    }


def standard_noise(seed: int, matrix_index: int, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100 + matrix_index)
    return torch.randn(shape, generator=generator, dtype=dtype)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((row["model"], row["noise_percent"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (model, percentage), rows in groups.items():
        item: dict[str, Any] = {
            "model": model,
            "noise_percent": percentage,
            "runs": len(rows),
        }
        for metric in ("mcc", "accuracy", "eval_loss", "positive_rate"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_min"] = float(values.min())
            item[f"{metric}_max"] = float(values.max())
        summary.append(item)
    return sorted(summary, key=lambda row: (row["noise_percent"], row["model"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    percentages = parse_number_list(args.noise_percentages, float)
    seeds = parse_number_list(args.seeds, int)
    if any(percentage < 0 for percentage in percentages):
        raise ValueError("Noise percentages must be nonnegative.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    loader, labels, num_labels = make_loader(
        args.task_name,
        args.clean_model,
        args.data_dir,
        args.eval_batch_size,
        args.workers,
    )
    print(f"Loaded {len(labels)} {args.task_name} validation examples.", flush=True)

    reference_model = load_model(args.clean_model, torch.device("cpu"), num_labels)
    reference_matrices = selected_matrices(reference_model, args.layer, args.head)
    reference_max_abs = {
        name: float(matrix.detach().abs().max()) for name, matrix in reference_matrices.items()
    }
    del reference_model

    raw_rows: list[dict[str, Any]] = []
    model_specs = [(label, Path(path)) for label, path in args.model]
    for label, path in model_specs:
        print(f"Loading {label}: {path}", flush=True)
        model = load_model(path, device, num_labels)
        matrices = selected_matrices(model, args.layer, args.head)
        originals = {name: matrix.detach().clone() for name, matrix in matrices.items()}
        model_max_abs = {name: float(matrix.detach().abs().max()) for name, matrix in matrices.items()}
        print(f"  max |W|: {model_max_abs}", flush=True)

        for percentage in percentages:
            run_seeds = [0] if percentage == 0 else seeds
            for seed in run_seeds:
                with torch.no_grad():
                    for matrix_index, name in enumerate(MATRIX_NAMES):
                        noise = standard_noise(
                            seed=seed,
                            matrix_index=matrix_index,
                            shape=originals[name].shape,
                            dtype=originals[name].dtype,
                        ).to(device)
                        sigma = percentage / 100.0 * reference_max_abs[name]
                        matrices[name].copy_(originals[name] + sigma * noise)

                metrics = evaluate_model(model, loader, labels, device)
                row = {
                    "model": label,
                    "checkpoint": str(path),
                    "noise_percent": percentage,
                    "seed": seed,
                    **metrics,
                }
                raw_rows.append(row)
                print(
                    f"  noise={percentage:>5g}% seed={seed:>4d} "
                    f"MCC={metrics['mcc']:.6f} accuracy={metrics['accuracy']:.6f}",
                    flush=True,
                )

        with torch.no_grad():
            for name in MATRIX_NAMES:
                matrices[name].copy_(originals[name])
        del model

    summary_rows = summarize(raw_rows)
    write_csv(args.output_dir / "raw_results.csv", raw_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    metadata = {
        "task": args.task_name,
        "primary_metric": "accuracy" if args.task_name == "sst-2" else "mcc",
        "layer": args.layer,
        "head": args.head,
        "matrix_names": list(MATRIX_NAMES),
        "matrix_shape": [26, 26],
        "noise_distribution": "additive Gaussian",
        "noise_definition": (
            "epsilon ~ Normal(0, (noise_percent / 100 * max_abs_clean_matrix)^2); "
            "the same standard-normal samples and clean reference scales are used for every checkpoint"
        ),
        "bias_noised": False,
        "clean_reference_model": str(args.clean_model),
        "clean_reference_max_abs": reference_max_abs,
        "models": [{"label": label, "path": str(path)} for label, path in model_specs],
        "noise_percentages": percentages,
        "seeds": seeds,
        "validation_examples": int(len(labels)),
        "device": str(device),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
