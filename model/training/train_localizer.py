#!/usr/bin/env python3
"""
Training script for the CNN spectrogram localization model.

Trains a CNN that outputs per-frame dysfluency probabilities from mel-spectrograms.

Usage:
    python -m model.training.train_localizer \
        --data_dir data/train \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-3 \
        --output_dir model/weights
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
    parser.add_argument("--data_dir", type=str, default="data/train", help="Data directory containing audio/ and labels/.")
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

    parser.add_argument("--clean", action="store_true",
                        help="Ignore resume checkpoints and train from scratch.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated dataset sources to train on (e.g. 'uclass'). "
                             "Defaults to all sources in sources.csv.")
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Positive-frame weight for BCE loss (imbalance correction). "
                             "Default: auto-computed from the training split's frame labels.")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false",
                        default=True,
                        help="Disable spectrogram masking augmentation (ablation).")
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
    from tqdm import tqdm

    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for spectrograms, frame_labels in tqdm(dataloader, desc="  Train", leave=False):
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
        frame_f1, mean_iou, auroc, avg_loss, all_true, all_pred_probs
    """
    import torch
    from tqdm import tqdm

    from model.training.utils import compute_event_mean_iou, compute_frame_auroc

    model.model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    num_batches = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for spectrograms, frame_labels in tqdm(dataloader, desc="  Val", leave=False):
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
    mean_iou = compute_event_mean_iou(all_true, pred_bin)

    # Threshold-free score for early stopping / best-checkpoint selection
    auroc = compute_frame_auroc(all_true, all_pred)

    avg_loss = total_loss / max(num_batches, 1)
    return f1, mean_iou, auroc, avg_loss, all_true, all_pred


def train(args) -> Dict:
    """Main training function."""
    import torch
    from torch.utils.data import DataLoader

    from model.data.dataset import LocalizationDataset
    from model.fingerprint import CNN_LOCALIZER_RESUME_KEYS, localizer_fingerprint
    from model.training.utils import (
        CSVLogger,
        EarlyStopping,
        TeeLogger,
        build_localizer_criterion,
        compute_frame_pos_weight,
        maybe_skip_completed,
        save_resume_state,
        try_load_resume,
    )

    set_seed(args.seed)
    fp = localizer_fingerprint(args, "loc")
    os.makedirs(args.output_dir, exist_ok=True)
    tee = TeeLogger(os.path.join(args.output_dir, f"{fp}_training.log"))
    sys.stdout = tee
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Training Localization Model")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # ---- Checkpoint resume ----
    resume_ckpt = try_load_resume(args, device, fp)
    skip_history = maybe_skip_completed(resume_ckpt, args.epochs)
    if skip_history is not None:
        tee.close()
        return skip_history

    # ---- Dataset ----
    print("\n  Loading dataset...")
    dataset = LocalizationDataset(
        data_dir=args.data_dir,
        sr=16000,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        max_length_seconds=args.max_length_seconds,
        sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
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

    from model.data.augmentation import AugmentedDataset, SpectrogramAugmentor
    from model.config.defaults import AUGMENTATION_ENABLED
    if AUGMENTATION_ENABLED and args.augmentation:
        train_dataset = AugmentedDataset(
            train_dataset,
            augmentor=None,
            augment_spectrogram=True,
            spectrogram_augmentor=SpectrogramAugmentor(),
        )
        print(f"  Augmentation: ON (spectrogram masking)")
    else:
        print(f"  Augmentation: OFF")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
    model = CNNSpectrogramLocalizer(n_mels=args.n_mels, dropout=args.dropout)
    print(f"  Model: CNNSpectrogramLocalizer (n_mels={args.n_mels})")

    model.model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    start_epoch = 1
    best_score = 0.0
    history = {"train_loss": [], "val_loss": [], "val_frame_f1": [], "val_mean_iou": [], "val_auroc": []}
    if resume_ckpt is not None:
        model.model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_score = resume_ckpt.get("best_f1", 0.0)
        history = resume_ckpt["history"]
        print(f"  Resuming from epoch {resume_ckpt['epoch']} (best score: {best_score:.4f})")

    # ---- Loss & Optimizer ----
    # Positive class weighting for frame-level imbalance
    if args.pos_weight is None:
        train_samples = [dataset.samples[i] for i in train_idx]
        pos_weight = compute_frame_pos_weight(
            train_samples,
            num_frames=dataset.max_frames,
            sr=16000,
            hop_length=args.hop_length,
        )
        print(f"  Auto pos_weight: {pos_weight} "
              f"(from {len(train_samples)} training clips)")
    else:
        pos_weight = args.pos_weight
        print(f"  pos_weight: {pos_weight} (explicit)")
    criterion = build_localizer_criterion(pos_weight, device)

    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # Always build with the FULL schedule so restored last_epoch maps correctly.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    # ---- Logging ----
    log_path = os.path.join(args.output_dir, f"{fp}_log.csv")
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_frame_f1", "val_mean_iou", "val_auroc", "lr"])

    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    # ---- Training ----
    print(f"\n  Starting training for {args.epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_f1, val_iou, val_auroc, val_loss, all_true, all_pred = evaluate_localizer(
            model, val_loader, device
        )
        scheduler.step()

        from model.evaluation.metrics import find_optimal_threshold
        best_thresh, best_f1 = find_optimal_threshold(all_true, all_pred, metric="f1")

        epoch_time = time.time() - epoch_start

        # Log
        logger.log(
            epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_frame_f1=f"{val_f1:.4f}", val_mean_iou=f"{val_iou:.4f}",
            val_auroc=f"{val_auroc:.4f}", lr=f"{current_lr:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_frame_f1"].append(val_f1)
        history["val_mean_iou"].append(val_iou)
        history["val_auroc"].append(val_auroc)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"frame_F1={val_f1:.3f} | F1@opt={best_f1:.3f}(t={best_thresh:.2f}) | "
            f"IoU={val_iou:.3f} | AUROC={val_auroc:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Save resume checkpoint
        save_resume_state(model, optimizer, scheduler, epoch, best_score, history, args, fp,
                          CNN_LOCALIZER_RESUME_KEYS, completed=False)

        # Checkpoint best (threshold-free AUROC, not frame F1@0.5)
        if val_auroc > best_score:
            best_score = val_auroc
            ckpt_path = os.path.join(args.output_dir, f"{fp}_best.pt")
            model.save(ckpt_path)

        if early_stopping.step(val_auroc):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Mark training as complete so future runs skip this model
    save_resume_state(model, optimizer, scheduler, epoch, best_score, history, args, fp,
                      CNN_LOCALIZER_RESUME_KEYS, completed=True)

    # Save final
    final_path = os.path.join(args.output_dir, f"{fp}_final.pt")
    model.save(final_path)

    total_time = time.time() - start_time
    logger.close()

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best val AUROC: {best_score:.4f}")
    print(f"  Saved: {final_path}")

    # Save training curves
    _save_training_curves(history, fp, args.output_dir)

    return history


def _save_training_curves(history: Dict, fp: str, output_dir: str) -> None:
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
    path = os.path.join(curves_dir, f"{fp}_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
