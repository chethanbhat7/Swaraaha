#!/usr/bin/env python3
"""
Training script for the shared-backbone multitask Wav2Vec2 classifier.

One backbone + five per-class heads trained jointly. Loss = sum of five
FocalLoss(gamma=2.0), one per head over its (B,2) logits.

Usage:
    python -m model.training.train_multitask_classifier \
        --data_dir data/train \
        --epochs 20 \
        --batch_size 16 \
        --output_dir model/weights
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.config.defaults import DYSFLUENCY_CLASSES
from model.fingerprint import (
    MULTITASK_RESUME_KEYS,
    multitask_fingerprint,
)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Train the shared-backbone multitask wav2vec2 classifier."
    )
    parser.add_argument("--data_dir", type=str, default="data/train", help="Data directory containing audio/ and labels/.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="model/weights", help="Directory to save trained weights.")
    parser.add_argument("--max_length_seconds", type=float, default=3.0, help="Max audio length in seconds (pad/truncate).")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of linear warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val macro-F1 improvement).")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 = auto-detect).")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="HuggingFace model name.")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension of each per-class head.")
    parser.add_argument("--class_names", type=str, default=",".join(DYSFLUENCY_CLASSES),
                        help="Comma-separated head class names.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3, help="Freeze backbone for first N epochs (train heads only).")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "cross_entropy"], help="Loss function.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (only used if --loss_type=focal).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over N steps before optimizer update.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for preprocessed audio (auto-derived from data_dir if omitted).")
    parser.add_argument("--clean", action="store_true", help="Ignore checkpoint and start training from scratch.")
    args = parser.parse_args(argv)
    args.class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    return args


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_pos_weights(dataset, class_names=None):
    """Per-class neg/pos ratio from dataset.label_vectors (clip level)."""
    from model.config.defaults import DYSFLUENCY_CLASSES
    if class_names is None:
        class_names = DYSFLUENCY_CLASSES
    label_vectors = getattr(dataset, 'label_vectors', None)
    if label_vectors is None:
        label_vectors = [np.asarray(dataset[i][1], dtype=float) for i in range(len(dataset))]
    label_vectors = np.asarray(label_vectors, dtype=float)
    weights = {}
    for name in class_names:
        col = label_vectors[:, DYSFLUENCY_CLASSES.index(name)]
        n_pos = int(col.sum())
        n_neg = len(col) - n_pos
        weights[name] = round(n_neg / max(n_pos, 1), 4) if n_pos > 0 else 1.0
    return weights


class MultiLabelBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss summed over per-class heads with per-class pos_weight.

    logits: {class_name: (B, 2)}; labels: (B, len(DYSFLUENCY_CLASSES)) multi-hot.
    """

    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, logits, labels):
        import torch.nn.functional as F
        from model.config.defaults import DYSFLUENCY_CLASSES
        device = next(iter(logits.values())).device
        total = torch.zeros((), dtype=torch.float32, device=device)
        for name, logit in logits.items():
            target = labels[:, DYSFLUENCY_CLASSES.index(name)].float().to(device)
            total = total + F.binary_cross_entropy_with_logits(
                logit[:, 1], target,
                pos_weight=torch.tensor(self.pos_weights[name], dtype=torch.float32,
                                        device=device),
            )
        return total


def multitask_loss(logits: Dict[str, "torch.Tensor"], labels: "torch.Tensor",
                   criterion, device="cpu", class_names=None) -> "torch.Tensor":
    """Sum of per-head losses over the (5,) multi-hot label vector."""
    import torch

    if class_names is None:
        class_names = DYSFLUENCY_CLASSES
    total = torch.zeros((), dtype=torch.float32, device=device)
    for name in class_names:
        target = labels[:, DYSFLUENCY_CLASSES.index(name)].long().to(device)
        total = total + criterion(logits[name], target)
    return total


def _partition_model_params(model) -> Tuple[List, List]:
    """Split the shared-backbone model into head vs backbone parameters.

    Head params live under ``model.model.heads.*``. The check uses a
    substring match (not ``startswith``) because ``torch.compile`` renames
    every parameter with an ``_orig_mod.`` prefix; the heads must still be
    found after compilation (regression: optimizer got an empty list).
    """
    backbone_params = []
    head_params = []
    for name, param in model.model.named_parameters():
        if "heads." in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def stratified_split(
    dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Stratified split by the first positive class.

    Uses ``dataset.label_vectors`` when available (cheap: reads label files
    only) instead of materialising samples via ``__getitem__``.
    """
    rng = np.random.RandomState(seed)
    label_vectors = getattr(dataset, 'label_vectors', None)
    labels = []
    if label_vectors is not None:
        for label_vec in label_vectors:
            label_vec = np.asarray(label_vec)
            positives = np.where(label_vec > 0)[0]
            labels.append(int(positives[0]) if len(positives) > 0 else -1)
    else:
        for i in range(len(dataset)):
            _, label_vec = dataset[i]
            label_vec = np.asarray(label_vec)
            positives = np.where(label_vec > 0)[0]
            labels.append(int(positives[0]) if len(positives) > 0 else -1)

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)
    train_idx: list = []
    val_idx: list = []
    for label, indices in sorted(groups.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class SubsetDataset:
    """Wrapper that exposes a subset of a dataset by index list."""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps=1, class_names=None):
    """Train for one epoch. Returns average loss."""
    import warnings
    import torch
    from tqdm import tqdm

    warnings.filterwarnings("ignore", "Detected call of.*lr_scheduler.step.*before.*optimizer.step")

    use_amp = device.type == "cuda"
    model.model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for i, (audio, labels) in enumerate(tqdm(dataloader, desc="  Train", leave=False)):
        audio = audio.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model.forward(audio)
            loss = multitask_loss(logits, labels, criterion, device,
                                  class_names=class_names)
            loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    if num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return total_loss / max(num_batches, 1)


def evaluate_multitask(model, dataloader, device, class_names=None) -> Tuple[float, float, float]:
    """Evaluate all heads.

    Returns:
        accuracy, macro_f1, avg_loss
    """
    import torch
    from tqdm import tqdm

    if class_names is None:
        class_names = DYSFLUENCY_CLASSES

    model.model.eval()
    all_true = {name: [] for name in class_names}
    all_pred = {name: [] for name in class_names}
    total_loss = 0.0
    num_batches = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for audio, labels in tqdm(dataloader, desc="  Val", leave=False):
            audio = audio.to(device)
            logits = model.forward(audio)

            for name in class_names:
                target = labels[:, DYSFLUENCY_CLASSES.index(name)].long().to(device)
                total_loss += criterion(logits[name], target).item()

                probs = torch.softmax(logits[name], dim=-1)
                preds = (probs[:, 1] >= 0.5).cpu().numpy()
                all_true[name].extend(target.cpu().numpy().tolist())
                all_pred[name].extend(preds.tolist())

            num_batches += 1

    per_class_f1 = {}
    for name in class_names:
        y_true = np.array(all_true[name])
        y_pred = np.array(all_pred[name])
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = f1

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    all_true_flat = np.concatenate([np.array(all_true[n]) for n in class_names])
    all_pred_flat = np.concatenate([np.array(all_pred[n]) for n in class_names])
    accuracy = float(np.mean(all_true_flat == all_pred_flat)) if len(all_true_flat) > 0 else 0.0

    avg_loss = total_loss / max(num_batches, 1)
    return accuracy, macro_f1, avg_loss


def train(args) -> Dict:
    """Main training function. Returns training history."""
    import torch
    from torch.utils.data import DataLoader

    from model.classification.multitask import MultiTaskWav2VecClassifier
    from model.data.dataset import ClassificationDataset
    from model.training.utils import (
        CSVLogger,
        EarlyStopping,
        FocalLoss,
        TeeLogger,
        get_warmup_linear_schedule,
        maybe_skip_completed,
        save_checkpoint,
        save_resume_state,
        try_load_resume,
    )

    if isinstance(args.class_names, str):
        args.class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]

    set_seed(args.seed)
    fp = multitask_fingerprint(args)
    os.makedirs(args.output_dir, exist_ok=True)
    tee = TeeLogger(os.path.join(args.output_dir, f"{fp}_training.log"))
    sys.stdout = tee

    print(f"\n  Configuration:")
    for k in MULTITASK_RESUME_KEYS:
        print(f"    {k}: {getattr(args, k)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.num_workers == 0:
        args.num_workers = os.cpu_count() or 4

    print(f"\n{'='*60}")
    print(f"  Training MULTITASK classifier (shared backbone, {len(args.class_names)} heads)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    if args.gradient_accumulation_steps > 1:
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")

    # ---- Dataset ----
    print("\n  Loading dataset...")
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(args.data_dir.rstrip("/")),
            "cache",
            os.path.basename(args.data_dir.rstrip("/")),
        )
    dataset = ClassificationDataset(
        data_dir=args.data_dir,
        sr=16000,
        max_length_seconds=args.max_length_seconds,
        cache_dir=cache_dir,
    )
    print(f"  Total samples: {len(dataset)}")
    print(f"  Cache: {cache_dir}")

    if len(dataset) == 0:
        print("  ERROR: No samples found. Check data_dir structure.")
        sys.exit(1)

    # ---- Split ----
    train_idx, val_idx = stratified_split(dataset, val_ratio=0.2, seed=args.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_dataset = SubsetDataset(dataset, train_idx)
    val_dataset = SubsetDataset(dataset, val_idx)

    from model.data.augmentation import AugmentedDataset, AudioAugmentor
    from model.config.defaults import AUGMENTATION_ENABLED
    if AUGMENTATION_ENABLED:
        train_dataset = AugmentedDataset(train_dataset, augmentor=AudioAugmentor())
        print(f"  Augmentation: ON")

    # ---- Class distribution ----
    all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    train_labels = all_labels[train_idx]
    for i, name in enumerate(DYSFLUENCY_CLASSES):
        n_pos = int(train_labels[:, i].sum())
        print(f"  Positive ratio (train) {name}: "
              f"{n_pos/len(train_labels):.3f} ({n_pos}/{len(train_labels)})")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Loss ----
    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"  Loss: Focal (gamma={args.focal_gamma}) per head, summed over {len(args.class_names)} heads")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print(f"  Loss: CrossEntropy (no weights)")

    # ---- Model ----
    print(f"  Loading pretrained model: {args.model_name}...")
    model = MultiTaskWav2VecClassifier(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        class_names=args.class_names,
    )
    model.model.to(device)
    if device.type == "cuda":
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        torch._dynamo.config.suppress_errors = True
        model._model = torch.compile(model._model)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- Freeze backbone setup ----
    backbone_params, head_params = _partition_model_params(model)

    backbone_frozen = args.freeze_backbone_epochs > 0
    if backbone_frozen:
        for p in backbone_params:
            p.requires_grad = False
        print(f"  Backbone frozen for first {args.freeze_backbone_epochs} epochs")
        print(f"  Head parameters: {sum(p.numel() for p in head_params):,}")

    # ---- Checkpoint resume ----
    resume_ckpt = try_load_resume(args, device, fp)

    skip_history = maybe_skip_completed(resume_ckpt, args.epochs)
    if skip_history is not None:
        tee.close()
        return skip_history

    if resume_ckpt is not None:
        model.model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_f1 = resume_ckpt["best_f1"]
        history = resume_ckpt["history"]
        resumed_backbone_frozen = resume_ckpt["backbone_frozen"]
        print(f"  Resuming from epoch {resume_ckpt['epoch']} (best F1: {best_f1:.4f})")
    else:
        start_epoch = 1
        best_f1 = 0.0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "macro_f1": []}
        resumed_backbone_frozen = backbone_frozen

    if backbone_frozen != resumed_backbone_frozen:
        if resumed_backbone_frozen:
            for p in backbone_params:
                p.requires_grad = False
        else:
            for p in backbone_params:
                p.requires_grad = True
        backbone_frozen = resumed_backbone_frozen

    trainable_params = head_params if backbone_frozen else model.model.parameters()

    # ---- Optimizer ----
    if resume_ckpt is not None and not backbone_frozen:
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay},
        ])
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay,
        )

    optim_steps_per_epoch = (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    total_optim_steps = optim_steps_per_epoch * args.epochs
    scheduler = get_warmup_linear_schedule(optimizer, args.warmup_steps, total_optim_steps)

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    # ---- Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{fp}_log.csv")
    log_mode = "a" if resume_ckpt is not None else "w"
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_acc", "macro_f1", "lr"], mode=log_mode)

    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    if start_epoch > args.epochs:
        print(f"\n  Training already complete (epoch {start_epoch - 1}/{args.epochs})")
        final_path = os.path.join(args.output_dir, f"{fp}_final.pt")
        save_checkpoint(model, optimizer, args.epochs, {"macro_f1": best_f1}, final_path, scheduler)
        tee.close()
        return history

    print(f"\n  {'Resuming' if resume_ckpt else 'Starting'} training ({start_epoch}/{args.epochs})...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        if backbone_frozen and epoch == args.freeze_backbone_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            optimizer.add_param_group({
                "params": backbone_params,
                "lr": args.lr * 0.1,
                "weight_decay": args.weight_decay,
            })
            scheduler.base_lrs.append(args.lr * 0.1)
            scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
            backbone_frozen = False
            print(f"  >>> Backbone UNFROZEN at epoch {epoch} (head LR={args.lr:.2e}, backbone LR={args.lr*0.1:.2e})")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.gradient_accumulation_steps)
        current_lr = optimizer.param_groups[0]["lr"]

        val_acc, macro_f1, val_loss = evaluate_multitask(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        logger.log(
            epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}", macro_f1=f"{macro_f1:.4f}", lr=f"{current_lr:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["macro_f1"].append(macro_f1)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | macro_F1={macro_f1:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, fp, MULTITASK_RESUME_KEYS, backbone_frozen)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            ckpt_path = os.path.join(args.output_dir, f"{fp}_best.pt")
            save_checkpoint(model, optimizer, epoch, {"macro_f1": macro_f1, "val_acc": val_acc}, ckpt_path, scheduler)

        if early_stopping.step(macro_f1):
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, fp, MULTITASK_RESUME_KEYS, backbone_frozen, completed=True)

    final_path = os.path.join(args.output_dir, f"{fp}_final.pt")
    save_checkpoint(model, optimizer, epoch, {"macro_f1": best_f1}, final_path, scheduler)

    total_time = time.time() - start_time
    logger.close()

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best val macro-F1: {best_f1:.4f}")
    print(f"  Saved: {final_path}")

    tee.close()
    return history


if __name__ == "__main__":
    args = parse_args()
    train(args)
