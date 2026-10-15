"""
main.py
CLI Orchestrator for Autoregressive Text Generation System (ICT4403 Assignment 2).
Executes data preprocessing, comparable training across Simple RNN, BiRNN, LSTM, and GRU,
quantitative evaluation, temperature sampling generation, and academic visualization export.

Context Acquisition:
- Responsibility: End-to-end execution pipeline orchestration and CLI interface.
- Upstream: src.data, src.models, src.train, src.evaluate, src.generate, src.utils.
- Downstream: saved_models/, results/.
- Data Ownership: Orchestrates persistence of all metrics, checkpoints, text samples, and plots.
- Runtime Constraints: Configurable epochs, default ~5-10 minutes on CPU.
- Failure Expectations: Fail loudly on invalid arguments or missing data.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

from src.data import prepare_dataset_and_dataloaders
from src.eda import run_full_eda
from src.models import get_model
from src.train import train_model
from src.evaluate import evaluate_model, benchmark_inference_latency, compile_metrics_table
from src.generate import run_sampling_experiment, DEFAULT_SEEDS
from src.utils import (
    set_seed,
    get_compute_device,
    plot_training_curves,
    plot_perplexity_curves,
    plot_comparative_bar_charts,
    plot_qualitative_metrics,
)


def run_pipeline(
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 25,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    lr: float = 0.001,
    seed: int = 42,
    output_dir: str = "results",
    model_dir: str = "saved_models",
):
    print("=" * 78)
    print("ICT4403 Assignment 2: Autoregressive Text Generation Comparative System")
    print("=" * 78)

    set_seed(seed)
    device = get_compute_device()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Dataset Preprocessing & Ingestion
    print("\n--- Task 1: Dataset Acquisition & Sliding Window Sequence Construction ---")
    vocab, train_loader, val_loader, test_loader, meta = prepare_dataset_and_dataloaders(
        seq_len=seq_len,
        step_size=2,
        batch_size=batch_size,
        min_freq=2,
        max_vocab_size=6000,
    )

    print(f"Corpus: The Adventures of Sherlock Holmes (Project Gutenberg)")
    print(f"Total Tokens: {meta['total_tokens']:,} | Vocabulary Size: {meta['vocab_size']:,}")
    print(f"Causal Window Length (L): {meta['seq_len']} words")
    print(f"Samples: Train={meta['train_samples']:,} | Val={meta['val_samples']:,} | Test={meta['test_samples']:,}")

    # 1.5 Research-Grade Exploratory Data Analysis (EDA)
    print("\n--- Task 1.5: Research-Grade Exploratory Data Analysis & Diagnostics ---")
    run_full_eda()

    # 2. Model Training & Validation under Comparable Conditions
    model_names = ["Simple RNN", "Bidirectional RNN", "LSTM", "GRU"]
    model_keys = ["rnn", "birnn", "lstm", "gru"]

    trained_models = {}
    training_histories = {}
    evaluation_records = {}

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n--- Task 2: Model Training under Comparable Conditions ---")
    for name, key in zip(model_names, model_keys):
        print(f"\n>> Initializing Architecture: {name}")
        model = get_model(
            model_name=key,
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
        ).to(device)

        params_info = model.count_parameters()
        print(f"   Parameters: {params_info['total_params']:,} trainable")

        save_path = os.path.join(model_dir, f"{key}.pt")
        history, elapsed = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            lr=lr,
            save_path=save_path,
            device=device,
            verbose=True,
        )

        # Load best checkpoint for testing
        model.load_state_dict(torch.load(save_path, map_location=device))
        trained_models[name] = model
        training_histories[name] = history

        # Evaluate on Test Set
        test_loss, test_ppl, top1, top5 = evaluate_model(model, test_loader, criterion, device)
        latency = benchmark_inference_latency(model, device, seq_len=seq_len)

        evaluation_records[name] = {
            "total_params": params_info["total_params"],
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "top1_acc": top1,
            "top5_acc": top5,
            "training_time_s": elapsed,
            "inference_latency_ms": latency,
        }
        print(f"   [Test Results] Loss: {test_loss:.4f} | PPL: {test_ppl:.2f} | Acc@1: {top1:.2f}% | Acc@5: {top5:.2f}% | Latency: {latency:.2f}ms")

    # 3. Quantitative Summary Table & Plots
    print("\n--- Compiling Quantitative Metrics ---")
    metrics_df = compile_metrics_table(evaluation_records)
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(metrics_df.to_string(index=False))

    plot_training_curves(training_histories, output_dir=output_dir)
    plot_perplexity_curves(training_histories, output_dir=output_dir)
    plot_comparative_bar_charts(metrics_df, output_dir=output_dir)
    print(f"Plots saved to directory: {output_dir}/")

    # 4. Task 3: Autoregressive Generation & Sampling Temperature Analysis
    print("\n--- Task 3: Text Generation & Temperature Sampling Analysis (T in [0.2, 0.7, 1.2]) ---")
    temperatures = [0.2, 0.7, 1.2]
    samples_df, diversity_df = run_sampling_experiment(
        models=trained_models,
        vocab=vocab,
        temperatures=temperatures,
        seeds=DEFAULT_SEEDS,
        max_new_tokens=40,
        device=device,
        seq_len=seq_len,
    )

    samples_csv_path = os.path.join(output_dir, "generated_samples.csv")
    samples_df.to_csv(samples_csv_path, index=False)

    diversity_csv_path = os.path.join(output_dir, "diversity_metrics.csv")
    diversity_df.to_csv(diversity_csv_path, index=False)

    plot_qualitative_metrics(diversity_df, output_dir=output_dir)
    print("\nLexical Diversity & Repetition across Temperatures:")
    print(diversity_df.to_string(index=False))

    print("\n--- Sample Generation Highlights ---")
    for name in model_names:
        print(f"\n[{name}]")
        for temp in temperatures:
            subset = samples_df[(samples_df["Model"] == name) & (samples_df["Temperature"] == temp)]
            sample_text = subset.iloc[0]["Generated Text"]
            print(f"  T={temp:0.1f} | \"{sample_text[:120]}...\"")

    print("\n" + "=" * 78)
    print("Execution completed successfully! All checkpoints, logs, and plots ready.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive Text Generation Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=25, help="Context sequence length L")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden state dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Model checkpoint directory")

    args = parser.parse_args()
    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
