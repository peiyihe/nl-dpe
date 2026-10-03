"""Replay external differential-pair Q and K results in SST-2 inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from nl_dpe.evaluate import convert_examples_to_features, get_task, get_tensor_data
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--x-file", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--external-q", type=Path, required=True)
    parser.add_argument("--external-k", type=Path, required=True)
    parser.add_argument("--conductance-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--head", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def evaluate(
    model: TinyBertForSequenceClassification,
    loader: DataLoader,
    labels: np.ndarray,
    modules: dict[str, torch.nn.Linear],
    start: int,
    end: int,
    replacements_without_bias: dict[str, np.ndarray] | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    cursors = {name: 0 for name in modules}
    handles = []
    if replacements_without_bias:
        tensors = {
            name: torch.from_numpy(value.astype(np.float32, copy=False))
            for name, value in replacements_without_bias.items()
        }

        def make_hook(name: str):
            module = modules[name]
            replacement = tensors[name]

            def replace(_module, _inputs, output):
                count = output.shape[0] * output.shape[1]
                cursor = cursors[name]
                block = replacement[cursor : cursor + count].reshape(
                    output.shape[0], output.shape[1], end - start
                )
                block = block.to(device=output.device, dtype=output.dtype)
                bias = module.bias[start:end].to(device=output.device, dtype=output.dtype)
                result = output.clone()
                result[:, :, start:end] = block + bias
                cursors[name] += count
                return result

            return replace

        handles = [
            modules[name].register_forward_hook(make_hook(name))
            for name in replacements_without_bias
        ]

    logits_parts: list[np.ndarray] = []
    loss_sum = 0.0
    try:
        with torch.no_grad():
            for batch in loader:
                input_ids, input_mask, segment_ids, label_ids, _seq_lengths = batch
                logits, _, _ = model(input_ids, segment_ids, input_mask)
                loss_sum += F.cross_entropy(logits, label_ids, reduction="sum").item()
                logits_parts.append(logits.cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    if replacements_without_bias:
        for name, replacement in replacements_without_bias.items():
            if cursors[name] != len(replacement):
                raise RuntimeError(
                    f"Consumed {cursors[name]} {name} rows, expected {len(replacement)}"
                )
    logits = np.concatenate(logits_parts)
    predictions = logits.argmax(axis=1)
    metrics = {
        "accuracy": float(np.mean(predictions == labels)),
        "eval_loss": float(loss_sum / len(labels)),
    }
    return metrics, predictions, logits


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = actual - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "max_abs_error": float(np.max(np.abs(error))),
        "reference_std": float(np.std(reference)),
        "nrmse_over_reference_std": float(rmse / np.std(reference)),
        "pearson_correlation": float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1]),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    x = np.load(args.x_file).astype(np.float64)
    conductance = json.loads(args.conductance_metadata.read_text(encoding="utf-8"))
    external_files = {"query": args.external_q, "key": args.external_k}
    short_names = {"query": "WQ", "key": "WK"}
    external: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    projection_metrics: dict[str, dict[str, float]] = {}
    for name in ("query", "key"):
        short = short_names[name]
        paired = np.load(external_files[name]).astype(np.float64)
        weight = np.load(args.weights_dir / f"{short}.npy").astype(np.float64)
        if x.shape != (55808, 26) or paired.shape != (55808, 52) or weight.shape != (26, 26):
            raise ValueError(
                f"Unexpected {name} shapes: X={x.shape}, paired={paired.shape}, W={weight.shape}"
            )
        scale = float(conductance["matrices"][short]["scale_uS_per_weight"])
        external[name] = (paired[:, 0::2] - paired[:, 1::2]) / (scale * 1e-6)
        reference[name] = x @ weight
        projection_metrics[short] = {
            "scale_uS_per_weight": scale,
            **error_metrics(external[name], reference[name]),
        }
        np.save(
            args.output_dir / f"{name.upper()}_external_weight_domain.npy",
            external[name].astype(np.float32),
        )
        np.save(
            args.output_dir / f"{name.upper()}_software_reference.npy",
            reference[name].astype(np.float32),
        )

    task = get_task("sst-2", str(args.data_dir))
    tokenizer = BertTokenizer.from_pretrained(str(args.model_path), do_lower_case=True)
    features = convert_examples_to_features(task.get_dev_examples(), tokenizer, task)
    data, label_tensor = get_tensor_data(task.output_mode, features)
    loader = DataLoader(data, sampler=SequentialSampler(data), batch_size=args.batch_size)
    labels = label_tensor.numpy()
    model = TinyBertForSequenceClassification.from_pretrained(
        str(args.model_path), num_labels=task.num_labels()
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    start = (args.head - 1) * head_dim
    end = start + head_dim
    attention = model.bert.encoder.layer[args.layer - 1].attention.self
    modules = {"query": attention.query, "key": attention.key}

    baseline, baseline_predictions, baseline_logits = evaluate(
        model, loader, labels, modules, start, end, None
    )
    ideal_qk, ideal_predictions, ideal_logits = evaluate(
        model, loader, labels, modules, start, end, reference
    )
    external_q_only, q_only_predictions, q_only_logits = evaluate(
        model, loader, labels, modules, start, end, {"query": external["query"]}
    )
    external_qk, qk_predictions, qk_logits = evaluate(
        model, loader, labels, modules, start, end, external
    )

    results = {
        "source_checkpoint": str(args.model_path),
        "external_files": {name: str(path) for name, path in external_files.items()},
        "task": "sst-2",
        "split": "dev",
        "examples": int(len(labels)),
        "layer_1_based": args.layer,
        "head_1_based": args.head,
        "replacement": "external WQ and WK; WV and all other computation remain clean software",
        "external_outputs_already_x_scale_restored": True,
        "clean_query_and_key_biases_added_after_external_matmul": True,
        "projection_error_metrics_before_bias": projection_metrics,
        "baseline_clean": baseline,
        "ideal_qk_replay": ideal_qk,
        "external_q_only_replay": external_q_only,
        "external_qk_replay": external_qk,
        "ideal_qk_vs_baseline_logit_max_abs_difference": float(
            np.max(np.abs(ideal_logits - baseline_logits))
        ),
        "external_q_only_vs_baseline_logit_max_abs_difference": float(
            np.max(np.abs(q_only_logits - baseline_logits))
        ),
        "external_qk_vs_baseline_logit_max_abs_difference": float(
            np.max(np.abs(qk_logits - baseline_logits))
        ),
        "external_q_only_predictions_changed_vs_baseline": int(
            np.count_nonzero(q_only_predictions != baseline_predictions)
        ),
        "external_qk_predictions_changed_vs_baseline": int(
            np.count_nonzero(qk_predictions != baseline_predictions)
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ("example_index", "label", "baseline_prediction", "external_q_prediction", "external_qk_prediction")
        )
        writer.writerows(
            zip(
                range(len(labels)),
                labels,
                baseline_predictions,
                q_only_predictions,
                qk_predictions,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
