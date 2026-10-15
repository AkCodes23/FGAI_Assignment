"""
src/train.py
Module: Training Loop Orchestration and Checkpoint Management.

Context Acquisition:
- Responsibility: Training loop, backpropagation, gradient clipping, validation monitoring, and checkpoint persistence.
- Upstream: PyTorch nn.Module, torch.optim, src/models.py, src/data.py, src/evaluate.py.
- Downstream: main.py, notebooks, saved_models/*.pt.
- Data Ownership: Checkpoint weights and epoch metric histories.
- Runtime Constraints: < 3 min per model for 10 epochs on CPU.
- Failure Expectations: Assert loss is finite (detect NaN/Inf), cleanly persist best checkpoint.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluate import evaluate_model


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> float:
    """
    Trains model for one epoch over the training DataLoader.
    Returns:
        epoch_loss: Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("Loss diverged to NaN or Inf during training.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 5.0,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], float]:
    """
    Orchestrates full training loop across num_epochs with validation monitoring.
    Returns:
        history: Dictionary containing loss and perplexity lists
        total_training_time_s: Total seconds elapsed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "val_top1_acc": [],
        "val_top5_acc": [],
    }

    best_val_loss = float("inf")
    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=max_grad_norm,
        )

        val_loss, val_ppl, val_top1, val_top5 = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        train_ppl = math.exp(min(train_loss, 50.0))
        epoch_elapsed = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_top1_acc"].append(val_top1)
        history["val_top5_acc"].append(val_top5)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                marker = " [Saved Best]"
            else:
                marker = ""
        else:
            marker = ""

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                f"Val Acc@1: {val_top1:.1f}%{marker}"
            )

    total_training_time_s = time.perf_counter() - start_time
    return history, total_training_time_s

