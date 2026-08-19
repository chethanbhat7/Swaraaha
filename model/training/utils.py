"""
Shared training utilities for Swaraaha.

Checkpointing, logging, learning rate schedulers, and helper functions.
"""

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.config.defaults import SAMPLE_RATE


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SubsetDataset:
    """Wrapper that exposes a subset of a dataset by index list."""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


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


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Simple random split (localization labels are per-frame, not per-class)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def maybe_compile(model, device):
    """Apply torch.compile to model if on CUDA. Safe no-op on CPU."""
    if device.type == "cuda":
        import warnings
        import logging
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        torch._dynamo.config.suppress_errors = True
        model._model = torch.compile(model._model)


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

    def __init__(self, path: str, fields: List[str], mode: str = "w"):
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, mode, newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        if mode == "w":
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

    def isatty(self):
        return self._stdout.isatty()

    def fileno(self):
        return self._stdout.fileno()

    def close(self):
        self.file.close()
        sys.stdout = self._stdout


def resume_checkpoint_path(args, fp: str) -> str:
    """Path to the resume checkpoint for a fingerprint string."""
    return os.path.join(args.output_dir, f"{fp}_checkpoint.pt")


def save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, fp,
                      resume_keys=None, backbone_frozen=False, completed=False):
    """Persist a resume checkpoint keyed by a fingerprint string."""
    path = resume_checkpoint_path(args, fp)
    if resume_keys is None:
        from model.fingerprint import RESUME_KEYS
        resume_keys = RESUME_KEYS
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_f1": best_f1,
        "history": history,
        "args": {k: getattr(args, k) for k in resume_keys if hasattr(args, k)},
        "backbone_frozen": backbone_frozen,
        "completed": completed,
        "fp": fp,
    }
    torch.save(ckpt, path)


def try_load_resume(args, device, fp: str):
    """Load a resume checkpoint, or None if --clean or the file is missing."""
    path = resume_checkpoint_path(args, fp)
    if getattr(args, "clean", False) or not os.path.isfile(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def maybe_skip_completed(resume_ckpt, epochs: int):
    """Print the skip message and return saved history if training already
    completed; otherwise return None."""
    if resume_ckpt is not None and resume_ckpt.get("completed"):
        print(f"\n  Training already complete (epoch {resume_ckpt['epoch']}/{epochs}, "
              f"best F1: {resume_ckpt.get('best_f1', 0.0):.4f})")
        print("  Skipping — delete the *_checkpoint.pt file or pass --clean to retrain.")
        return resume_ckpt.get("history")
    return None


def align_frame_labels(frame_labels, logits):
    """Truncate or pad frame labels to match the model's output frame count.

    Wav2Vec2's conv feature extractor can emit one fewer frame than the
    dataset's label count (e.g. 499 vs 500 for 160000 samples), which breaks
    the BCE loss shape check. Align labels to the model output before use.
    """
    n = logits.shape[-1]
    if frame_labels.shape[-1] == n:
        return frame_labels
    if frame_labels.shape[-1] > n:
        return frame_labels[..., :n]
    return torch.nn.functional.pad(frame_labels, (0, n - frame_labels.shape[-1]))


def build_localizer_criterion(pos_weight: Optional[float] = None, device: str = "cpu"):
    """Build the BCEWithLogitsLoss criterion for localizer training.

    Localizer frame labels are heavily imbalanced (dysfluent frames are rare),
    so an unweighted loss under-penalizes missed frames. When ``pos_weight`` is
    given the criterion weights positive frames by that factor.
    """
    if pos_weight is not None:
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    return torch.nn.BCEWithLogitsLoss()


def compute_frame_auroc(y_true, y_scores) -> float:
    """Threshold-free frame-level score used for localizer early stopping.

    Frame F1 at a fixed threshold of 0.5 is a dead metric for rare positives
    (all-negative predictions score near-perfect), so early stopping must use a
    threshold-independent ranking score instead. Returns 0.5 (chance) when only
    one class is present, keeping the metric well-defined on degenerate splits.
    """
    from model.evaluation.metrics import compute_binary_metrics

    return float(compute_binary_metrics(y_true, y_scores)["auroc"])


def extract_regions(mask: np.ndarray) -> List[tuple]:
    """Find contiguous runs of 1s in a binary frame mask.

    Returns a list of (start, end) frame index pairs (end exclusive).
    """
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


def compute_mean_iou(true_regions, pred_regions) -> float:
    """Mean best-match IoU between true and predicted regions (frame units).

    For each true region, take the highest IoU against any predicted region;
    return the mean across true regions. 0.0 if either list is empty.
    """
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


def compute_event_mean_iou(y_true: np.ndarray, pred_bin: np.ndarray) -> float:
    """Mean event-level IoU from a binary frame mask and a thresholded mask."""
    return compute_mean_iou(
        extract_regions(y_true.astype(int)),
        extract_regions(pred_bin.astype(int)),
    )


def compute_frame_pos_weight(
    samples: List[Dict],
    num_frames: int,
    sr: int = SAMPLE_RATE,
    hop_length: int = 512,
) -> float:
    """Compute inverse-frequency pos_weight from per-clip label CSVs.

    Frame labels are built from the label intervals only (no audio is loaded),
    so the ratio can be computed once at training start. The masks use the same
    frame convention as ``create_frame_labels``: frame ``i`` covers audio
    samples ``i * hop_length`` onwards.

    Args:
        samples: List of sample dicts with a ``label_path`` key.
        num_frames: Fixed frame count per clip (the padded training length).
        sr: Sample rate used for frame placement.
        hop_length: Frames per spectrogram hop.

    Returns:
        pos_weight = round(n_neg / n_pos, 4), or 1.0 when no positive frames.
    """
    from model.data.dataset import load_label_csv
    from model.data.preprocessing import create_frame_labels

    total_frames = len(samples) * num_frames
    total_pos = 0
    for sample in samples:
        intervals = load_label_csv(sample["label_path"])
        mask = create_frame_labels(
            [(start, end) for start, end, _ in intervals],
            num_frames=num_frames, sr=sr, hop_length=hop_length,
        )
        total_pos += int(mask.sum())
    n_pos = total_pos
    n_neg = total_frames - total_pos
    if n_pos == 0:
        return 1.0
    return round(n_neg / n_pos, 4)


def find_latest_localizer(output_dir: str, pipeline: str) -> Optional[str]:
    """Return the most recently modified {pipeline} best-checkpoint path, or None."""
    import glob

    prefix = "cnnloc_" if pipeline == "loc" else "w2v2loc_"
    paths = glob.glob(os.path.join(output_dir, f"{prefix}*_best.pt"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def update_registry_localizers(registry_path: str, output_dir: str) -> Dict[str, str]:
    """Scan output_dir for the newest localizer checkpoints and write them into
    the registry.json localization section. Returns the new localization mapping
    (may be empty if no checkpoints were found)."""
    mapping = {}
    for pipeline, key in (("loc", "cnn"), ("wav2vec", "wav2vec2")):
        best = find_latest_localizer(output_dir, pipeline)
        if best:
            mapping[key] = os.path.relpath(best, os.path.dirname(os.path.dirname(registry_path)))

    with open(registry_path) as f:
        registry = json.load(f)
    registry["localization"] = mapping
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")
    return mapping
