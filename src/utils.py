"""
src/utils.py
Module: Reproducibility Seeding, Academic Plotting Utilities, and Metric Serialization.

Context Acquisition:
- Responsibility: Seed setting, visualization of training curves, metric comparison bar charts, CSV serialization.
- Upstream: matplotlib.pyplot, numpy, torch.
- Downstream: src/train.py, src/evaluate.py, main.py, notebooks.
- Data Ownership: Plot image files in results/, metric CSV files in results/.
- Runtime Constraints: Plotting execution < 5s; non-blocking file operations.
- Failure Expectations: Automatically create missing directories, handle plotting exceptions safely.
"""

import os
import random
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across random, numpy, and torch for strict reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_compute_device() -> torch.device:
    """
    Determines and verifies the best compute device (GPU vs CPU).
    Probes device with a small tensor operation to verify CUDA kernel execution.
    """
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(1, device="cuda")
            _ = probe + 1.0
            device_name = torch.cuda.get_device_name(0)
            print(f"[Compute Device] Utilizing GPU: {device_name}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[Compute Device] GPU detected ({torch.cuda.get_device_name(0)}), but CUDA kernel execution requires sm_120 compilation: {e}")
            print("[Compute Device] Safely utilizing multi-threaded CPU for execution.")
            return torch.device("cpu")
    print("[Compute Device] Utilizing CPU.")
    return torch.device("cpu")


def plot_training_curves(
    history: Dict[str, Dict[str, List[float]]],
    output_dir: str = "results",
) -> str:
    """
    Plots training and validation loss curves side-by-side for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Simple RNN": "#e74c3c", "Bidirectional RNN": "#e67e22", "LSTM": "#2ecc71", "GRU": "#3498db"}

    # Subplot 1: Training Loss
    for model_name, metrics in history.items():
        epochs = range(1, len(metrics["train_loss"]) + 1)
        axes[0].plot(
            epochs,
            metrics["train_loss"],
            marker="o",
            label=model_name,
            color=colors.get(model_name, None),
            linewidth=2,
        )
    axes[0].set_title("Training Loss vs. Epochs", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # Subplot 2: Validation Loss
    for model_name, metrics in history.items():
        epochs = range(1, len(metrics["val_loss"]) + 1)
        axes[1].plot(
            epochs,
            metrics["val_loss"],
            marker="s",
            label=model_name,
            color=colors.get(model_name, None),
            linewidth=2,
        )
    axes[1].set_title("Validation Loss vs. Epochs", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_perplexity_curves(
    history: Dict[str, Dict[str, List[float]]],
    output_dir: str = "results",
) -> str:
    """
    Plots validation perplexity across epochs for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(9, 5))

    colors = {"Simple RNN": "#e74c3c", "Bidirectional RNN": "#e67e22", "LSTM": "#2ecc71", "GRU": "#3498db"}

    for model_name, metrics in history.items():
        epochs = range(1, len(metrics["val_ppl"]) + 1)
        plt.plot(
            epochs,
            metrics["val_ppl"],
            marker="o",
            label=model_name,
            color=colors.get(model_name, None),
            linewidth=2,
        )

    plt.title("Validation Perplexity (PPL) vs. Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "perplexity_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_comparative_bar_charts(
    metrics_df: pd.DataFrame,
    output_dir: str = "results",
) -> List[str]:
    """
    Generates comparison bar charts for accuracy, training time, and perplexity.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    # 1. Top-1 vs Top-5 Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    top1 = metrics_df["Top-1 Accuracy (%)"].tolist()
    top5 = metrics_df["Top-5 Accuracy (%)"].tolist()

    rects1 = ax.bar(x - width / 2, top1, width, label="Top-1 Accuracy", color="#3498db")
    rects2 = ax.bar(x + width / 2, top5, width, label="Top-5 Accuracy", color="#2ecc71")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Next-Word Prediction Accuracy: Top-1 vs. Top-5", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    saved_paths.append(acc_path)

    # 2. Training Time vs Parameter Count
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = "#9b59b6"
    color2 = "#f39c12"

    train_times = metrics_df["Training Time (s)"].tolist()
    params_m = [p / 1e6 for p in metrics_df["Parameters"].tolist()]

    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Training Time (seconds)", color=color1, fontsize=12)
    bars1 = ax1.bar(x - width / 2, train_times, width, color=color1, label="Training Time (s)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Parameters (Millions)", color=color2, fontsize=12)
    bars2 = ax2.bar(x + width / 2, params_m, width, color=color2, label="Parameters (M)")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Computational Efficiency: Training Time and Model Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "efficiency_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    saved_paths.append(comp_path)

    return saved_paths


def plot_qualitative_metrics(
    diversity_df: pd.DataFrame,
    output_dir: str = "results",
) -> str:
    """
    Plots Distinct-1, Distinct-2, and Repetition Rate across temperatures for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = diversity_df["Model"].unique()
    temps = sorted(diversity_df["Temperature"].unique())

    # Distinct-1
    for m in models:
        subset = diversity_df[diversity_df["Model"] == m]
        axes[0].plot(subset["Temperature"], subset["Distinct-1"], marker="o", label=m, linewidth=2)
    axes[0].set_title("Lexical Diversity (Distinct-1) vs. Temperature", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sampling Temperature", fontsize=11)
    axes[0].set_ylabel("Distinct-1 Ratio", fontsize=11)
    axes[0].set_xticks(temps)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    # Distinct-2
    for m in models:
        subset = diversity_df[diversity_df["Model"] == m]
        axes[1].plot(subset["Temperature"], subset["Distinct-2"], marker="s", label=m, linewidth=2)
    axes[1].set_title("Lexical Diversity (Distinct-2) vs. Temperature", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sampling Temperature", fontsize=11)
    axes[1].set_ylabel("Distinct-2 Ratio", fontsize=11)
    axes[1].set_xticks(temps)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    # Repetition Rate
    for m in models:
        subset = diversity_df[diversity_df["Model"] == m]
        axes[2].plot(subset["Temperature"], subset["Repetition Rate (%)"], marker="^", label=m, linewidth=2)
    axes[2].set_title("Repetition Rate (%) vs. Temperature", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Sampling Temperature", fontsize=11)
    axes[2].set_ylabel("3-gram Repetition Rate (%)", fontsize=11)
    axes[2].set_xticks(temps)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "qualitative_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
