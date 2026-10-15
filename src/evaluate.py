"""
src/evaluate.py
Module: Quantitative Evaluation for Autoregressive Language Models.

Context Acquisition:
- Responsibility: Evaluation of Cross-Entropy Loss, Perplexity, Top-1 and Top-5 Next-Word Accuracy, and Inference Latency.
- Upstream: PyTorch nn.Module, torch.utils.data.DataLoader, src/models.py, src/data.py.
- Downstream: src/train.py, main.py, notebooks, results/metrics.csv.
- Data Ownership: Quantitative metrics records and summary DataFrames.
- Runtime Constraints: Execution time < 10s per model on CPU.
- Failure Expectations: Fail loudly on empty evaluation sets, guard against log/exp numerical overflow.
"""

import math
import time
from typing import Dict, Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluates loss, perplexity, Top-1 accuracy, and Top-5 accuracy over a DataLoader.

    Returns:
        avg_loss: float
        perplexity: float
        top1_acc: float (percentage 0-100)
        top5_acc: float (percentage 0-100)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)

            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Top-1 accuracy
            preds_top1 = logits.argmax(dim=-1)
            correct_top1 += (preds_top1 == targets).sum().item()

            # Top-5 accuracy
            _, preds_top5 = logits.topk(k=min(5, logits.size(-1)), dim=-1)
            targets_expanded = targets.unsqueeze(-1).expand_as(preds_top5)
            correct_top5 += (preds_top5 == targets_expanded).any(dim=-1).sum().item()

    if total_samples == 0:
        raise ValueError("Cannot evaluate over empty DataLoader.")

    avg_loss = total_loss / total_samples
    try:
        perplexity = math.exp(min(avg_loss, 50.0))  # Prevent numerical overflow
    except OverflowError:
        perplexity = float("inf")

    top1_acc = (correct_top1 / total_samples) * 100.0
    top5_acc = (correct_top5 / total_samples) * 100.0

    return avg_loss, perplexity, top1_acc, top5_acc


def benchmark_inference_latency(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 25,
    num_runs: int = 100,
) -> float:
    """
    Benchmarks single-token autoregressive generation latency in milliseconds.
    """
    model.eval()
    dummy_input = torch.randint(1, 1000, (1, seq_len), dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    return avg_latency_ms


def compile_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Transforms dictionary of model metrics into a structured comparison DataFrame.
    """
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            "Model": model_name,
            "Parameters": int(m.get("total_params", 0)),
            "Train Loss": round(m.get("train_loss", 0.0), 4),
            "Val Loss": round(m.get("val_loss", 0.0), 4),
            "Test Loss": round(m.get("test_loss", 0.0), 4),
            "Test Perplexity": round(m.get("test_ppl", 0.0), 2),
            "Top-1 Accuracy (%)": round(m.get("top1_acc", 0.0), 2),
            "Top-5 Accuracy (%)": round(m.get("top5_acc", 0.0), 2),
            "Training Time (s)": round(m.get("training_time_s", 0.0), 2),
            "Inference Latency (ms)": round(m.get("inference_latency_ms", 0.0), 2),
        })

    df = pd.DataFrame(rows)
    return df
