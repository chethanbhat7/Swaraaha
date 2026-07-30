"""
Shared training utilities for Swaraaha.

Checkpointing, logging, learning rate schedulers, and helper functions.
"""

import csv
import json
import os
from typing import Dict, List, Optional

import torch
import numpy as np


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
    scheduler=None,
    extra: Optional[Dict] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: PyTorch model (or wrapper with .model attribute).
        optimizer: PyTorch optimizer.
        epoch: Current epoch number.
        metrics: Dict of metric values to save.
        path: Output path (.pt file).
        scheduler: Optional LR scheduler state.
        extra: Optional extra state to include.
    """
    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Handle wrapped models (e.g. BaseWav2VecClassifier)
    model_state = (
        model.model.state_dict() if hasattr(model, "model") else model.state_dict()
    )

    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        state["extra"] = extra

    torch.save(state, path)


def load_checkpoint(path: str, model=None, optimizer=None, scheduler=None):
    """
    Load a training checkpoint.

    Returns:
        Dict with keys: epoch, metrics, model_state_dict, optimizer_state_dict, etc.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if model is not None:
        model_state = (
            model.model if hasattr(model, "model") else model
        )
        model_state.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class CSVLogger:
    """
    Simple CSV logger for training metrics.

    Usage:
        logger = CSVLogger("training_log.csv", fields=["epoch", "train_loss", "val_f1"])
        logger.log(epoch=1, train_loss=0.42, val_f1=0.85)
        logger.close()
    """

    def __init__(self, path: str, fields: List[str]):
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()
        self._file.flush()

    def log(self, **kwargs) -> None:
        row = {k: kwargs.get(k, "") for k in self.fields}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EarlyStopping:
    """
    Early stopping monitor.

    Tracks a metric and signals when training should stop.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement to count as progress.
            mode: "max" (higher is better) or "min" (lower is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Update with the latest metric score.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def get_warmup_linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    """
    Create a linear warmup then linear decay learning rate schedule.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler.
    """
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
