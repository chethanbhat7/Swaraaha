#!/usr/bin/env python3
"""
Training script for the CNN spectrogram localization model.

Trains a CNN that outputs per-frame dysfluency probabilities from mel-spectrograms.

Usage:
    python -m model.training.train_localizer \
        --data_dir data \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-3 \
        --output_dir model/weights

    # Or with the alternative nn.Module CNN:
    python -m model.training.train_localizer \
        --cnn_type module \
        --data_dir data \
        --epochs 30
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN spectrogram localization model for dysfluency detection."
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing audio/ and labels/.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="model/weights", help="Directory to save trained weights.")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel frequency bins.")
    parser.add_argument("--hop_length", type=int, default=512, help="STFT hop length.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0, help="Max audio length in seconds.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--cnn_type", type=str, default="wrapper", choices=["wrapper", "module"],
                        help="CNN type: 'wrapper' (CNNSpectrogramLocalizer) or 'module' (SpectrogramCNN).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Simple random split (localization labels are per-frame, not per-class)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    return indices[n_val:].tolist(), indices[:n_val].tolist()


class SubsetDataset:
    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def create_frame_loss_weights(frame_labels: np.ndarray, pos_weight: float = 5.0) -> np.ndarray:
    """
    Create per-frame weights to handle class imbalance in localization.
    Dysfluent frames are typically a small fraction of total frames.
    """
    weights = np.ones_like(frame_labels, dtype=np.float32)
    weights[frame_labels == 1] = pos_weight
    return weights


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    import torch

    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for spectrograms, frame_labels in dataloader:
        spectrograms = spectrograms.to(device)  # [B, 1, n_mels, T]
        frame_labels = frame_labels.float().to(device)  # [B, T]

        optimizer.zero_grad()
        logits = model.forward(spectrograms).squeeze(1)  # [B, T]
        loss = criterion(logits, frame_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_localizer(model, dataloader, device, threshold: float = 0.5):
    """
    Evaluate localization model.

    Returns:
        frame_f1, mean_iou, avg_loss, all_true, all_pred_probs
    """
    import torch

    model.model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    num_batches = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for spectrograms, frame_labels in dataloader:
            spectrograms = spectrograms.to(device)
            frame_labels = frame_labels.float().to(device)

            logits = model.forward(spectrograms).squeeze(1)
            loss = criterion(logits, frame_labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            true = frame_labels.cpu().numpy()

            all_true.extend(true)
            all_pred.extend(probs)
            total_loss += loss.item()
            num_batches += 1

    all_true = np.concatenate(all_true) if all_true else np.array([])
    all_pred = np.concatenate(all_pred) if all_pred else np.array([])

    # Frame-level F1
    pred_bin = (all_pred >= threshold).astype(int)
    tp = np.sum((all_true == 1) & (pred_bin == 1))
    fp = np.sum((all_true == 0) & (pred_bin == 1))
    fn = np.sum((all_true == 1) & (pred_bin == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Event-level IoU
    true_regions = _extract_regions(all_true.astype(int))
    pred_regions = _extract_regions(pred_bin)
    mean_iou = _compute_mean_iou(true_regions, pred_regions)

    avg_loss = total_loss / max(num_batches, 1)
    return f1, mean_iou, avg_loss, all_true, all_pred


def _extract_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(mask):
        if v == 1 and not in_region:
            in_region = True
            start = i
        elif v == 0 and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(mask)))
    return regions


def _compute_mean_iou(true_regions, pred_regions) -> float:
    if not true_regions or not pred_regions:
        return 0.0

    ious = []
    for ts, te in true_regions:
        best_iou = 0.0
        for ps, pe in pred_regions:
            inter_start = max(ts, ps)
            inter_end = min(te, pe)
            intersection = max(0, inter_end - inter_start)
            union = (te - ts) + (pe - ps) - intersection
            iou = intersection / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)
        ious.append(best_iou)

    return float(np.mean(ious)) if ious else 0.0


def train(args) -> Dict:
    """Main training function."""
    import torch
    from torch.utils.data import DataLoader

    from model.data.dataset import LocalizationDataset
    from model.training.utils import (
        CSVLogger,
        EarlyStopping,
        save_checkpoint,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Training Localization Model ({args.cnn_type})")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # ---- Dataset ----
    print("\n  Loading dataset...")
    dataset = LocalizationDataset(
        data_dir=args.data_dir,
        sr=16000,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        max_length_seconds=args.max_length_seconds,
    )
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("  ERROR: No samples found. Check data_dir structure.")
        sys.exit(1)

    # ---- Split ----
    train_idx, val_idx = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_dataset = SubsetDataset(dataset, train_idx)
    val_dataset = SubsetDataset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    if args.cnn_type == "wrapper":
        from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
        model = CNNSpectrogramLocalizer(n_mels=args.n_mels, dropout=args.dropout)
        print(f"  Model: CNNSpectrogramLocalizer (n_mels={args.n_mels})")
    else:
        from model.localization.cnn import SpectrogramCNN
        model_wrapper = type("ModelWrapper", (), {
            "model": SpectrogramCNN(dropout=args.dropout),
            "forward": lambda self, x: self.model(x),
        })()
        model = model_wrapper
        print(f"  Model: SpectrogramCNN (nn.Module)")

    model.model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # ---- Loss & Optimizer ----
    # Positive class weighting for frame-level imbalance
    pos_weight = torch.tensor([5.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "localization_training_log.csv")
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_frame_f1", "val_mean_iou", "lr"])

    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    # ---- Training ----
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_frame_f1": [], "val_mean_iou": []}

    print(f"\n  Starting training for {args.epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_f1, val_iou, val_loss, _, _ = evaluate_localizer(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        logger.log(
            epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_frame_f1=f"{val_f1:.4f}", val_mean_iou=f"{val_iou:.4f}", lr=f"{current_lr:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_frame_f1"].append(val_f1)
        history["val_mean_iou"].append(val_iou)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"frame_F1={val_f1:.3f} | IoU={val_iou:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Checkpoint best
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.output_dir, "localizer_best.pt")
            save_checkpoint(model, optimizer, epoch, {"val_f1": val_f1, "val_iou": val_iou}, ckpt_path, scheduler)

        if early_stopping.step(val_f1):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Save final
    final_path = os.path.join(args.output_dir, "localizer_final.pt")
    save_checkpoint(model, optimizer, epoch, {"val_f1": best_f1}, final_path, scheduler)

    total_time = time.time() - start_time
    logger.close()

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best val frame F1: {best_f1:.4f}")
    print(f"  Saved: {final_path}")

    # Save training curves
    _save_training_curves(history, args.output_dir)

    return history


def _save_training_curves(history: Dict, output_dir: str) -> None:
    """Save training curves as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves_dir = os.path.join(output_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Localization — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_frame_f1"], label="Frame F1", marker="o", markersize=3, color="tab:green")
    ax2.plot(epochs, history["val_mean_iou"], label="Mean IoU", marker="s", markersize=3, color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Localization — Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(curves_dir, "localization_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
