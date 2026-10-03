"""Replay an external differential-pair WQ result in SST-2 inference."""

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
    parser.add_argument("--wq-file", type=Path, required=True)
    parser.add_argument("--external-paired-output", type=Path, required=True)
    parser.add_argument("--conductance-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-x-scale-restored",
        action="store_true",
        help="The paired output was already multiplied by max(abs(X)).",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--head", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def evaluate(
    model: TinyBertForSequenceClassification,
    loader: DataLoader,
    labels: np.ndarray,
    query_module: torch.nn.Linear,
    start: int,
    end: int,
    replacement_without_bias: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    cursor = 0
    handle = None
    if replacement_without_bias is not None:
        replacement = torch.from_numpy(replacement_without_bias.astype(np.float32, copy=False))

        def replace_query(_module, _inputs, output):
            nonlocal cursor
            count = output.shape[0] * output.shape[1]
            block = replacement[cursor : cursor + count].reshape(
                output.shape[0], output.shape[1], end - start
            )
            if block.shape[:2] != output.shape[:2]:
                raise RuntimeError("External Q rows do not match the current inference batch")
            block = block.to(device=output.device, dtype=output.dtype)
            bias = query_module.bias[start:end].to(device=output.device, dtype=output.dtype)
            result = output.clone()
            result[:, :, start:end] = block + bias
            cursor += count
            return result

        handle = query_module.register_forward_hook(replace_query)

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
        if handle is not None:
            handle.remove()

    if replacement_without_bias is not None and cursor != len(replacement_without_bias):
        raise RuntimeError(f"Consumed {cursor} external rows, expected {len(replacement_without_bias)}")
    logits = np.concatenate(logits_parts)
    predictions = logits.argmax(axis=1)
    metrics = {
        "accuracy": float(np.mean(predictions == labels)),
        "eval_loss": float(loss_sum / len(labels)),
    }
    return metrics, predictions, logits


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    x = np.load(args.x_file).astype(np.float64)
    wq = np.load(args.wq_file).astype(np.float64)
    paired = np.load(args.external_paired_output).astype(np.float64)
    if x.shape != (55808, 26) or wq.shape != (26, 26) or paired.shape != (55808, 52):
        raise ValueError(f"Unexpected shapes: X={x.shape}, WQ={wq.shape}, paired={paired.shape}")

    conductance = json.loads(args.conductance_metadata.read_text(encoding="utf-8"))
    weight_scale_uS = float(conductance["matrices"]["WQ"]["scale_uS_per_weight"])
    x_scale = float(np.abs(x).max())
    # The SuperT notebook uses X_eff=X/scale_x and conductances in siemens.
    # A corrected export may already have multiplied the output by scale_x.
    restore_x_factor = 1.0 if args.input_x_scale_restored else x_scale
    q_external = (
        (paired[:, 0::2] - paired[:, 1::2])
        * restore_x_factor
        / (weight_scale_uS * 1e-6)
    )
    q_reference = x @ wq
    error = q_external - q_reference
    q_metrics = {
        "x_global_scale": x_scale,
        "wq_scale_uS_per_weight": weight_scale_uS,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "reference_std": float(np.std(q_reference)),
        "nrmse_over_reference_std": float(np.sqrt(np.mean(error**2)) / np.std(q_reference)),
        "pearson_correlation": float(np.corrcoef(q_external.ravel(), q_reference.ravel())[0, 1]),
    }
    np.save(args.output_dir / "Q_external_weight_domain.npy", q_external.astype(np.float32))
    np.save(args.output_dir / "Q_software_reference.npy", q_reference.astype(np.float32))

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
    query_module = model.bert.encoder.layer[args.layer - 1].attention.self.query

    baseline, baseline_predictions, baseline_logits = evaluate(
        model, loader, labels, query_module, start, end, None
    )
    ideal_replay, ideal_predictions, ideal_logits = evaluate(
        model, loader, labels, query_module, start, end, q_reference
    )
    external, external_predictions, external_logits = evaluate(
        model, loader, labels, query_module, start, end, q_external
    )

    results = {
        "source_checkpoint": str(args.model_path),
        "external_file": str(args.external_paired_output),
        "task": "sst-2",
        "split": "dev",
        "examples": int(len(labels)),
        "layer_1_based": args.layer,
        "head_1_based": args.head,
        "replacement": "external WQ only; WK and WV remain clean software outputs",
        "external_paired_output_already_x_scale_restored": args.input_x_scale_restored,
        "external_conversion": (
            "Q=(paired[:,0::2]-paired[:,1::2])/(WQ_scale_uS_per_weight*1e-6); "
            "paired output already includes the restored X scale"
            if args.input_x_scale_restored
            else "Q=(paired[:,0::2]-paired[:,1::2])*max_abs(X)/(WQ_scale_uS_per_weight*1e-6)"
        ),
        "clean_query_bias_added_after_external_matmul": True,
        "q_error_metrics_before_bias": q_metrics,
        "baseline_clean": baseline,
        "ideal_wq_replay": ideal_replay,
        "external_wq_replay": external,
        "ideal_vs_baseline_logit_max_abs_difference": float(
            np.max(np.abs(ideal_logits - baseline_logits))
        ),
        "external_vs_baseline_logit_max_abs_difference": float(
            np.max(np.abs(external_logits - baseline_logits))
        ),
        "external_predictions_changed_vs_baseline": int(
            np.count_nonzero(external_predictions != baseline_predictions)
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("example_index", "label", "baseline_prediction", "external_wq_prediction"))
        writer.writerows(
            zip(range(len(labels)), labels, baseline_predictions, external_predictions)
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
