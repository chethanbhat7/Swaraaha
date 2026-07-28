#!/usr/bin/env python3
"""
Training script for the Wav2Vec2-based localization model.

Trains a Wav2Vec 2.0 backbone + temporal attention head for per-frame
dysfluency probability prediction from raw audio waveforms.

Usage:
    python -m model.training.train_wav2vec2_localizer \
        --data_dir data/train \
        --epochs 20 \
        --batch_size 4 \
        --lr 3e-5 \
        --output_dir model/weights

    # With backbone freezing for first 5 epochs:
    python -m model.training.train_wav2vec2_localizer \
        --data_dir data/train \
        --freeze_backbone_epochs 5 \
        --epochs 20
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Wav2Vec2 localization model for dysfluency detection."
    )
    parser.add_argument("--data_dir", type=str, default="data/train",
                        help="Data directory containing audio/ and labels/.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (smaller due to W2V2 memory).")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="model/weights",
                        help="Directory to save trained weights.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0,
                        help="Max audio length in seconds.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in temporal head.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for temporal head.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5,
                        help="Freeze W2V2 backbone for first N epochs (0=never).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers.")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base",
                        help="HuggingFace Wav2Vec2 model name.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation split ratio.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps.")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
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


def collate_wav2vec2(batch):
    """Collate function for Wav2Vec2 dataset — handles variable-length waveforms."""
    import torch
    waveforms, labels = zip(*batch)

    # All waveforms should be same length (padded in dataset)
    waveforms = torch.tensor(np.stack(waveforms), dtype=torch.float32)
    labels = torch.tensor(np.stack(labels), dtype=torch.long)

    return waveforms, labels


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch. Returns average loss."""
    import torch

    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for waveforms, frame_labels in dataloader:
        waveforms = waveforms.to(device)
        frame_labels = frame_labels.float().to(device)

        optimizer.zero_grad()
        logits = model.forward(waveforms).squeeze(1)  # (B, T)
        loss = criterion(logits, frame_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_model(model, dataloader, device, threshold: float = 0.5):
    """
    Evaluate localization model.

    Returns:
        frame_f1, avg_loss, all_true, all_pred_probs
    """
    import torch

    model.model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for waveforms, frame_labels in dataloader:
            waveforms = waveforms.to(device)
            frame_labels = frame_labels.float().to(device)

            logits = model.forward(waveforms).squeeze(1)
            loss = criterion(logits, frame_labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            true = frame_labels.cpu().numpy()

            all_true.extend(true)
            all_pred.extend(probs)
            total_loss += loss.item()
            num_batches += 1

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Frame-level F1
    pred_bin = (all_pred >= threshold).astype(int)
    tp = np.sum((all_true == 1) & (pred_bin == 1))
    fp = np.sum((all_true == 0) & (pred_bin == 1))
    fn = np.sum((all_true == 1) & (pred_bin == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_loss = total_loss / max(num_batches, 1)
    return f1, avg_loss, all_true, all_pred


def train(args) -> Dict:
    """Main training function."""
    import torch
    from torch.utils.data import DataLoader

    from model.localization.wav2vec2_localizer import Wav2Vec2Localizer
    from model.localization.wav2vec2_dataset import Wav2Vec2LocalizationDataset
    from model.training.utils import CSVLogger, EarlyStopping

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Training Wav2Vec2 Localization Model")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Freeze backbone: {args.freeze_backbone_epochs} epochs")

    # ---- Dataset ----
    print("\n  Loading dataset...")
    dataset = Wav2Vec2LocalizationDataset(
        data_dir=args.data_dir,
        sr=16000,
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

    from model.data.augmentation import AugmentedDataset, AudioAugmentor
    from model.config.defaults import AUGMENTATION_ENABLED
    if AUGMENTATION_ENABLED:
        train_dataset = AugmentedDataset(train_dataset, augmentor=AudioAugmentor())
        print(f"  Augmentation: ON")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_wav2vec2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_wav2vec2,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = Wav2Vec2Localizer(
        model_name=args.model_name,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    model.model.to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Freeze backbone initially
    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        print(f"  Backbone frozen for {args.freeze_backbone_epochs} epochs")

    # ---- Loss & Optimizer ----
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Linear warmup + cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "w2v2_localization_training_log.csv")
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_frame_f1", "lr"])
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_frame_f1": []}

    print(f"\n  Starting training for {args.epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warmup
        if (epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0):
            model.unfreeze_backbone()
            print(f"  Epoch {epoch}: Unfreezing backbone")
            optimizer = torch.optim.AdamW(
                model.model.parameters(), lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_f1, val_loss, _, _ = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        # Log
        logger.log(
            epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_frame_f1=f"{val_f1:.4f}", lr=f"{current_lr:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_frame_f1"].append(val_f1)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"frame_F1={val_f1:.3f} | lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Checkpoint best
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.output_dir, "w2v2_localizer_best.pt")
            model.save(ckpt_path)

        if early_stopping.step(val_f1):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Save final
    final_path = os.path.join(args.output_dir, "w2v2_localizer_final.pt")
    model.save(final_path)

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
    ax1.set_title("Wav2Vec2 Localization — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_frame_f1"], label="Frame F1", marker="o", markersize=3, color="tab:green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Wav2Vec2 Localization — Frame F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(curves_dir, "w2v2_localization_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
