"""Noise-aware fine-tuning for the 26x26 TinyBERT attention heads.

This entry point connects the conductance-dependent DPE noise model from
``TCAD/models/utils/noise.py`` to one attention head in the TinyBERT model used
by the top-level NL-DPE inference code.  The selected Q, K, and V matrices are
noised during every forward pass while the nominal model parameters are
optimized and exported for hardware programming.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from tqdm import tqdm

from nl_dpe.evaluate import (
    OutputMode,
    compute_metrics,
    convert_examples_to_features,
    get_task,
    get_tensor_data,
)
from TCAD.models.utils.noise import DPENoise
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


ATTENTION_MATRIX_NAMES = ("query", "key", "value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-aware fine-tuning for a 26x26 NL-DPE TinyBERT attention head."
    )
    parser.add_argument("--task-name", choices=("cola", "sst-2", "mrpc"), default="cola")
    parser.add_argument("--model-path", type=Path, default=Path("models/blushing-dove-984"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/glue/CoLA"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1, help="1-based encoder layer index.")
    parser.add_argument("--head", type=int, default=12, help="1-based attention head index.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-norm", type=float, default=1e-2)
    parser.add_argument("--g-min", type=float, default=0.0)
    parser.add_argument("--g-max", type=float, default=150.0)
    parser.add_argument("--std-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-analog-slicing",
        dest="analog_slicing",
        action="store_false",
        help=(
            "Disable the residual correction cell in DPENoise. Each weight is then "
            "represented by one continuous-conductance cell in the simulator."
        ),
    )
    parser.set_defaults(analog_slicing=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional deterministic subset size, intended for smoke tests.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false.")
    return device


def _noise_aware_linear_forward(linear: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    """Apply DPE noise to one output slice while retaining gradients to the nominal weight."""
    if linear._naf_noise_enabled:  # type: ignore[attr-defined]
        start = linear._naf_start_index  # type: ignore[attr-defined]
        end = linear._naf_end_index  # type: ignore[attr-defined]
        noisy_head = linear._naf_dpe_noise(linear.weight[start:end])  # type: ignore[attr-defined]
        weight = torch.cat((linear.weight[:start], noisy_head, linear.weight[end:]), dim=0)
    else:
        weight = linear.weight
    return F.linear(inputs, weight, linear.bias)


def attach_dpe_noise(
    model: TinyBertForSequenceClassification,
    layer_index: int,
    head_index: int,
    g_min: float,
    g_max: float,
    std_scale: float,
    analog_slicing: bool = True,
) -> dict[str, nn.Linear]:
    config = model.config
    if not 1 <= layer_index <= config.num_hidden_layers:
        raise ValueError(f"Layer must be in [1, {config.num_hidden_layers}], got {layer_index}.")
    if not 1 <= head_index <= config.num_attention_heads:
        raise ValueError(f"Head must be in [1, {config.num_attention_heads}], got {head_index}.")

    head_dim = config.hidden_size // config.num_attention_heads
    start = (head_index - 1) * head_dim
    end = start + head_dim
    attention = model.bert.encoder.layer[layer_index - 1].attention.self
    linears: dict[str, nn.Linear] = {}
    for name in ATTENTION_MATRIX_NAMES:
        linear = getattr(attention, name)
        expected_shape = (config.hidden_size, head_dim)
        if not isinstance(linear, nn.Linear) or tuple(linear.weight.shape) != expected_shape:
            raise ValueError(
                f"Expected {name} to be nn.Linear with weight shape {expected_shape}, "
                f"got {type(linear).__name__} with {tuple(linear.weight.shape)}."
            )
        linear.add_module(
            "_naf_dpe_noise",
            DPENoise(
                g_min=g_min,
                g_max=g_max,
                std_scale=std_scale,
                weight_shape=(head_dim, head_dim),
                slicing=analog_slicing,
            ),
        )
        linear._naf_start_index = start  # type: ignore[attr-defined]
        linear._naf_end_index = end  # type: ignore[attr-defined]
        linear._naf_noise_enabled = True  # type: ignore[attr-defined]
        linear.forward = types.MethodType(_noise_aware_linear_forward, linear)
        linears[name] = linear
    return linears


def set_noise_enabled(linears: dict[str, nn.Linear], enabled: bool) -> None:
    for linear in linears.values():
        linear._naf_noise_enabled = enabled  # type: ignore[attr-defined]


def build_dataloaders(args: argparse.Namespace, tokenizer: BertTokenizer):
    task = get_task(args.task_name, str(args.data_dir))
    train_features = convert_examples_to_features(task.get_train_examples(), tokenizer, task)
    eval_features = convert_examples_to_features(task.get_dev_examples(), tokenizer, task)
    train_data, _ = get_tensor_data(task.output_mode, train_features)
    eval_data, eval_labels = get_tensor_data(task.output_mode, eval_features)

    if args.max_train_samples is not None:
        sample_count = min(args.max_train_samples, len(train_data))
        train_data = Subset(train_data, range(sample_count))

    train_loader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_data,
        sampler=SequentialSampler(eval_data),
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    return task, train_loader, eval_loader, eval_labels


def evaluate(
    model: TinyBertForSequenceClassification,
    task_name: str,
    output_mode: str,
    eval_loader: DataLoader,
    eval_labels: torch.Tensor,
    num_labels: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids, input_mask, segment_ids, label_ids, _ = (tensor.to(device) for tensor in batch)
            logits, _, _ = model(input_ids, segment_ids, input_mask)
            if output_mode == OutputMode.CLASSIFICATION:
                loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == OutputMode.REGRESSION:
                loss = MSELoss()(logits.view(-1), label_ids.view(-1))
            else:
                raise ValueError(f"Unknown output mode: {output_mode}.")
            total_loss += loss.item()
            predictions.append(logits.detach().cpu().numpy())

    all_predictions = np.concatenate(predictions, axis=0)
    if output_mode == OutputMode.CLASSIFICATION:
        all_predictions = np.argmax(all_predictions, axis=1)
    else:
        all_predictions = np.squeeze(all_predictions)
    metrics = compute_metrics(task_name, all_predictions, eval_labels.numpy())
    metrics["eval_loss"] = total_loss / len(eval_loader)
    return {key: float(value) for key, value in metrics.items()}


def train_epoch(
    model: TinyBertForSequenceClassification,
    linears: dict[str, nn.Linear],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    output_mode: str,
    num_labels: int,
    weight_norm: float,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Training epoch {epoch}")
    for batch in progress:
        input_ids, input_mask, segment_ids, label_ids, _ = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(input_ids, segment_ids, input_mask)
        if output_mode == OutputMode.CLASSIFICATION:
            task_loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == OutputMode.REGRESSION:
            task_loss = MSELoss()(logits.view(-1), label_ids.view(-1))
        else:
            raise ValueError(f"Unknown output mode: {output_mode}.")

        selected_weights = [
            linear.weight[linear._naf_start_index : linear._naf_end_index]  # type: ignore[attr-defined]
            for linear in linears.values()
        ]
        max_weight = torch.stack([weight.abs().max() for weight in selected_weights]).max()
        loss = task_loss + weight_norm * max_weight
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(train_loader)


def export_artifacts(
    args: argparse.Namespace,
    model: TinyBertForSequenceClassification,
    linears: dict[str, nn.Linear],
    history: dict[str, Any],
) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    model_dir = args.output_dir / "model"
    weights_dir = args.output_dir / "weights"
    model_dir.mkdir(parents=True)
    weights_dir.mkdir()

    state_dict = model.state_dict()
    unexpected_noise_keys = [key for key in state_dict if "_naf_dpe_noise" in key]
    if unexpected_noise_keys:
        raise RuntimeError(f"Noise buffers unexpectedly entered the model state: {unexpected_noise_keys}")
    torch.save(state_dict, model_dir / "pytorch_model.bin")
    shutil.copy2(args.model_path / "config.json", model_dir / "config.json")
    shutil.copy2(args.model_path / "vocab.txt", model_dir / "vocab.txt")

    compatible_pickle: dict[str, np.ndarray] = {}
    for short_name, tensor_name in (("WQ", "query"), ("WK", "key"), ("WV", "value")):
        linear = linears[tensor_name]
        start = linear._naf_start_index  # type: ignore[attr-defined]
        end = linear._naf_end_index  # type: ignore[attr-defined]
        matrix = linear.weight[start:end].detach().cpu().numpy()
        bias = linear.bias[start:end].detach().cpu().numpy()
        if matrix.shape != (26, 26):
            raise RuntimeError(f"Expected {short_name} to be 26x26, got {matrix.shape}.")
        np.save(weights_dir / f"{short_name}.npy", matrix)
        np.savetxt(weights_dir / f"{short_name}.csv", matrix, delimiter=",")

        prefix = f"bert.encoder.layer.{args.layer - 1}.attention.self.{tensor_name}"
        compatible_pickle[f"{prefix}.weight"] = matrix
        compatible_pickle[f"{prefix}.bias"] = bias

    with (weights_dir / "attention_matrices.pkl").open("wb") as file:
        pickle.dump(compatible_pickle, file)

    metadata = {
        "task_name": args.task_name,
        "source_model": str(args.model_path),
        "layer": args.layer,
        "head": args.head,
        "matrix_shape": [26, 26],
        "noise_model": "TCAD DPENoise (conductance-dependent write and read noise)",
        "g_min": args.g_min,
        "g_max": args.g_max,
        "std_scale": args.std_scale,
        "analog_slicing": args.analog_slicing,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_norm": args.weight_norm,
        "seed": args.seed,
        "history": history,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    for required_file in ("config.json", "pytorch_model.bin", "vocab.txt"):
        if not (args.model_path / required_file).is_file():
            raise FileNotFoundError(f"Model file is missing: {args.model_path / required_file}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {args.data_dir}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    task = get_task(args.task_name, str(args.data_dir))
    tokenizer = BertTokenizer.from_pretrained(str(args.model_path), do_lower_case=True)
    model = TinyBertForSequenceClassification.from_pretrained(
        str(args.model_path), num_labels=task.num_labels()
    )
    linears = attach_dpe_noise(
        model,
        layer_index=args.layer,
        head_index=args.head,
        g_min=args.g_min,
        g_max=args.g_max,
        std_scale=args.std_scale,
        analog_slicing=args.analog_slicing,
    )
    model.to(device)
    task, train_loader, eval_loader, eval_labels = build_dataloaders(args, tokenizer)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history: dict[str, Any] = {"device": str(device), "epochs": []}

    set_noise_enabled(linears, False)
    history["baseline_clean"] = evaluate(
        model, args.task_name, task.output_mode, eval_loader, eval_labels, task.num_labels(), device
    )
    set_noise_enabled(linears, True)
    history["baseline_noisy"] = evaluate(
        model, args.task_name, task.output_mode, eval_loader, eval_labels, task.num_labels(), device
    )
    print(f"Baseline clean metrics: {history['baseline_clean']}")
    print(f"Baseline noisy metrics: {history['baseline_noisy']}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            linears,
            train_loader,
            optimizer,
            task.output_mode,
            task.num_labels(),
            args.weight_norm,
            device,
            epoch,
        )
        scheduler.step()
        metrics = evaluate(
            model, args.task_name, task.output_mode, eval_loader, eval_labels, task.num_labels(), device
        )
        epoch_result = {"epoch": epoch, "train_loss": train_loss, "noisy_eval": metrics}
        history["epochs"].append(epoch_result)
        print(f"Epoch {epoch}: {epoch_result}")

    set_noise_enabled(linears, False)
    history["final_clean"] = evaluate(
        model, args.task_name, task.output_mode, eval_loader, eval_labels, task.num_labels(), device
    )
    set_noise_enabled(linears, True)
    history["final_noisy"] = evaluate(
        model, args.task_name, task.output_mode, eval_loader, eval_labels, task.num_labels(), device
    )
    print(f"Final clean metrics: {history['final_clean']}")
    print(f"Final noisy metrics: {history['final_noisy']}")

    export_artifacts(args, model, linears, history)
    print(f"Saved noise-aware model and WQ/WK/WV matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
