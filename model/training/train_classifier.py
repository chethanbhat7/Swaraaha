#!/usr/bin/env python3
"""
Training script for Wav2Vec 2.0 binary classifiers.

Trains one binary classifier at a time (e.g., prolongation vs. not-prolongation)
using the StutterDataset from model.data.dataset.

Usage:
    python -m model.training.train_classifier \
        --class_name prolongation \
        --data_dir data/train \
        --epochs 20 \
        --batch_size 8 \
        --lr 3e-5 \
        --output_dir model/weights

    # Train all 5 classifiers:
    for cls in prolongation block soundrep wordrep interjection; do
        python -m model.training.train_classifier --class_name $cls --data_dir data/train --epochs 20
    done
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Args that must match for checkpoint resume
RESUME_KEYS = [
    "class_name", "data_dir", "model_name", "lr", "batch_size",
    "max_length_seconds", "warmup_steps", "weight_decay",
    "freeze_backbone_epochs", "loss_type", "focal_gamma", "seed",
    "gradient_accumulation_steps", "epochs",
]

FINGERPRINT_FMT = "{class_name}_e{epochs}_b{batch_size}_lr{lr}_frz{freeze_backbone_epochs}_{loss_type}_g{focal_gamma}_ga{gradient_accumulation_steps}_wu{warmup_steps}_wd{weight_decay}_ml{max_length_seconds}_s{seed}_{data_short}_{model_short}"

_MODEL_ALIASES = {
    "facebook/wav2vec2-base": "w2v2base",
    "facebook/wav2vec2-large": "w2v2large",
}


def _fmt_fp(v):
    if isinstance(v, float):
        s = f"{v:.10g}"
        s = re.sub(r'e([+-])0(\d)', r'e\1\2', s)
        return s
    return str(v)


def fingerprint(args) -> str:
    values = {k: _fmt_fp(getattr(args, k)) for k in RESUME_KEYS}
    values["data_short"] = os.path.basename(args.data_dir.rstrip("/"))
    values["model_short"] = _MODEL_ALIASES.get(args.model_name, args.model_name.replace("/", "_"))
    return FINGERPRINT_FMT.format(**values)


def parse_fingerprint(fp: str) -> dict:
    """Parse a fingerprint string back into a dict of params."""
    pattern = (
        r'^(?P<class_name>\w+)'
        r'_e(?P<epochs>\d+)'
        r'_b(?P<batch_size>\d+)'
        r'_lr(?P<lr>[\d.e\-]+)'
        r'_frz(?P<freeze_backbone_epochs>\d+)'
        r'_(?P<loss_type>\w+)'
        r'_g(?P<focal_gamma>[\d.e\-]+)'
        r'_ga(?P<gradient_accumulation_steps>\d+)'
        r'_wu(?P<warmup_steps>\d+)'
        r'_wd(?P<weight_decay>[\d.e\-]+)'
        r'_ml(?P<max_length_seconds>[\d.e\-]+)'
        r'_s(?P<seed>\d+)'
        r'_(?P<data_short>\w+)'
        r'_(?P<model_short>\w+)$'
    )
    m = re.match(pattern, fp)
    if not m:
        raise ValueError(f"Cannot parse fingerprint: {fp}")
    d = m.groupdict()
    for k in ("epochs", "batch_size", "freeze_backbone_epochs",
              "gradient_accumulation_steps", "warmup_steps", "seed"):
        d[k] = int(d[k])
    for k in ("lr", "focal_gamma", "weight_decay", "max_length_seconds"):
        d[k] = float(d[k])
    _MODEL_SHORT = {v: k for k, v in _MODEL_ALIASES.items()}
    ms = d.pop("model_short")
    d["model_name"] = _MODEL_SHORT.get(ms, ms)
    return d


def compute_pos_weight(labels: np.ndarray) -> float:
    """Compute pos_weight for BCEWithLogitsLoss from binary labels."""
    labels = np.asarray(labels).flatten()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return round(n_neg / n_pos, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Wav2Vec 2.0 binary classifier for stuttering dysfluency detection."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        choices=["prolongation", "block", "soundrep", "wordrep", "interjection"],
        help="Dysfluency class to train for.",
    )
    parser.add_argument("--data_dir", type=str, default="data/train", help="Data directory containing audio/ and labels/.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="model/weights", help="Directory to save trained weights.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0, help="Max audio length in seconds (pad/truncate).")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of linear warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val F1 improvement).")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 = auto-detect).")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="HuggingFace model name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3, help="Freeze backbone for first N epochs (train head only).")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "cross_entropy"], help="Loss function.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (only used if --loss_type=focal).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over N steps before optimizer update.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for preprocessed audio (auto-derived from data_dir if omitted).")
    parser.add_argument("--clean", action="store_true", help="Ignore checkpoint and start training from scratch.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_index(class_name: str) -> int:
    from model.classification import DYSFLUENCY_CLASSES
    return DYSFLUENCY_CLASSES.index(class_name)


def stratified_split(
    dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/val splits from a ClassificationDataset.

    Each sample's label_vector is reduced to a single class index for stratification.
    If a sample has multiple classes, the first positive class is used.
    """
    rng = np.random.RandomState(seed)

    labels = []
    for i in range(len(dataset)):
        _, label_vec = dataset[i]
        label_vec = np.asarray(label_vec)
        positives = np.where(label_vec > 0)[0]
        labels.append(int(positives[0]) if len(positives) > 0 else -1)

    labels = np.array(labels)
    indices = np.arange(len(dataset))

    train_indices, val_indices = [], []

    for cls_label in np.unique(labels):
        cls_indices = indices[labels == cls_label]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


class SubsetDataset:
    """Wrapper that exposes a subset of a dataset by index list."""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps=1):
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
        class_idx = model.class_idx
        binary_labels = labels[:, class_idx].long().to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model.forward(audio)
            loss = criterion(logits, binary_labels)
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


def evaluate_classifier(model, dataloader, device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate classifier on a dataset.

    Returns:
        accuracy, macro_f1, loss, all_true_labels, all_pred_labels
    """
    import torch
    from tqdm import tqdm

    model.model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for audio, labels in tqdm(dataloader, desc="  Val", leave=False):
            audio = audio.to(device)
            class_idx = model.class_idx
            binary_labels = labels[:, class_idx].long().to(device)

            logits = model.forward(audio)
            loss = criterion(logits, binary_labels)

            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] >= 0.5).cpu().numpy()
            true = binary_labels.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(true.tolist())
            total_loss += loss.item()
            num_batches += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(np.mean(all_preds == all_labels)) if len(all_labels) > 0 else 0.0

    tp = np.sum((all_labels == 1) & (all_preds == 1))
    fp = np.sum((all_labels == 0) & (all_preds == 1))
    fn = np.sum((all_labels == 1) & (all_preds == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_loss = total_loss / max(num_batches, 1)
    return accuracy, f1, avg_loss, all_labels, all_preds


class TeeLogger:
    """Write to both stdout and a file."""

    def __init__(self, path):
        self.file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, text):
        self._stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self._stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self._stdout


def _resume_checkpoint_path(args) -> str:
    return os.path.join(args.output_dir, f"{fingerprint(args)}_checkpoint.pt")


def _save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, backbone_frozen):
    import torch
    path = _resume_checkpoint_path(args)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_f1": best_f1,
        "history": history,
        "args": {k: getattr(args, k) for k in RESUME_KEYS},
        "backbone_frozen": backbone_frozen,
    }
    torch.save(ckpt, path)


def _try_load_resume(args, device):
    path = _resume_checkpoint_path(args)
    if args.clean or not os.path.isfile(path):
        return None

    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def train(args) -> Dict:
    """Main training function. Returns training history."""
    import torch
    from torch.utils.data import DataLoader

    from model.classification import DYSFLUENCY_CLASSES
    from model.data.dataset import ClassificationDataset
    from model.training.utils import (
        CSVLogger,
        EarlyStopping,
        FocalLoss,
        get_warmup_linear_schedule,
        save_checkpoint,
    )

    set_seed(args.seed)
    fp = fingerprint(args)
    os.makedirs(args.output_dir, exist_ok=True)
    tee = TeeLogger(os.path.join(args.output_dir, f"{fp}_training.log"))
    sys.stdout = tee

    print(f"\n  Configuration:")
    for k in RESUME_KEYS:
        print(f"    {k}: {getattr(args, k)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.num_workers == 0:
        args.num_workers = os.cpu_count() or 4

    class_idx = get_class_index(args.class_name)
    class_names = DYSFLUENCY_CLASSES

    print(f"\n{'='*60}")
    print(f"  Training Classifier: {args.class_name.upper()}")
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
    all_labels = np.array([dataset[i][1][class_idx] for i in range(len(dataset))])
    train_labels = all_labels[train_idx]
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    print(f"  Positive ratio (train): {n_pos/len(train_labels):.3f} ({n_pos}/{len(train_labels)})")

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
        print(f"  Loss: Focal (gamma={args.focal_gamma})")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print(f"  Loss: CrossEntropy (no weights)")

    # ---- Model ----
    from model.classification import DYSFLUENCY_CLASSES as _CLASSES
    cls_map = {
        "prolongation": "model.classification.prolongation.ProlongationClassifier",
        "block": "model.classification.block.BlockClassifier",
        "soundrep": "model.classification.soundrep.SoundRepClassifier",
        "wordrep": "model.classification.wordrep.WordRepClassifier",
        "interjection": "model.classification.interjection.InterjectionClassifier",
    }
    module_path, cls_name_str = cls_map[args.class_name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    ClassifierCls = getattr(mod, cls_name_str)

    print(f"  Loading pretrained model: {args.model_name}...")
    model = ClassifierCls(model_name=args.model_name)
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
    backbone_params = []
    head_params = []
    for name, param in model.model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_frozen = args.freeze_backbone_epochs > 0
    if backbone_frozen:
        for p in backbone_params:
            p.requires_grad = False
        print(f"  Backbone frozen for first {args.freeze_backbone_epochs} epochs")
        print(f"  Head parameters: {sum(p.numel() for p in head_params):,}")

    # ---- Checkpoint resume ----
    resume_ckpt = _try_load_resume(args, device)

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
        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        resumed_backbone_frozen = backbone_frozen

    # Ensure freeze state matches where we're resuming
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
    if backbone_frozen:
        scheduler = get_warmup_linear_schedule(optimizer, args.warmup_steps, total_optim_steps)
    else:
        remaining_optim_steps = optim_steps_per_epoch * (args.epochs - start_epoch + 1)
        scheduler = get_warmup_linear_schedule(optimizer, 0, remaining_optim_steps)

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    # ---- Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{fp}_log.csv")
    log_mode = "a" if resume_ckpt is not None else "w"
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "lr"], mode=log_mode)

    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    if start_epoch > args.epochs:
        print(f"\n  Training already complete (epoch {start_epoch - 1}/{args.epochs})")
        final_path = os.path.join(args.output_dir, f"{fp}_final.pt")
        save_checkpoint(model, optimizer, args.epochs, {"val_f1": best_f1}, final_path, scheduler)
        tee.close()
        return history

    print(f"\n  {'Resuming' if resume_ckpt else 'Starting'} training ({start_epoch}/{args.epochs})...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone at the right epoch (preserving optimizer momentum)
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

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.gradient_accumulation_steps)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_acc, val_f1, val_loss, _, _ = evaluate_classifier(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log
        logger.log(
            epoch=epoch, train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}", val_f1=f"{val_f1:.4f}", lr=f"{current_lr:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_F1={val_f1:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Save resume checkpoint
        _save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, backbone_frozen)

        # Checkpoint best
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.output_dir, f"{fp}_best.pt")
            save_checkpoint(model, optimizer, epoch, {"val_f1": val_f1, "val_acc": val_acc}, ckpt_path, scheduler)

        # Early stopping
        if early_stopping.step(val_f1):
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save final model
    final_path = os.path.join(args.output_dir, f"{fp}_final.pt")
    save_checkpoint(model, optimizer, epoch, {"val_f1": best_f1}, final_path, scheduler)

    total_time = time.time() - start_time
    logger.close()

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best val F1: {best_f1:.4f}")
    print(f"  Saved: {final_path}")

    # Save training curves
    _save_training_curves(history, args.class_name, fp, args.output_dir)

    tee.close()
    return history


def _save_training_curves(history: Dict, class_name: str, fp: str, output_dir: str) -> None:
    """Save training loss and F1 curves as PNG."""
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
    ax1.set_title(f"{class_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1"], label="Val F1", marker="o", markersize=3, color="tab:green")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3, color="tab:blue")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{class_name} — Metrics")
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
