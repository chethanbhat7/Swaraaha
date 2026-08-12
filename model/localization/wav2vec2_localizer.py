"""
Wav2Vec 2.0-based dysfluency localization model for Swaraaha.

Uses Wav2Vec 2.0 contextual embeddings as backbone, with a temporal
localization head that predicts per-frame dysfluency probabilities.

Architecture:
    Wav2Vec2Model → Classifier → (B, 1, T_frames)

Input:  (B, max_samples,)  — raw waveform at 16kHz
Output: (B, 1, T_frames)  — per-frame dysfluency logits

Frame resolution: ~20ms per frame (Wav2Vec2 internal subsampling factor = 320 samples)
"""

import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


def _wav2vec2_model_class():
    """Resolve the Wav2Vec2 backbone class (lazily, to avoid a heavy import)."""
    from transformers import Wav2Vec2Model

    return Wav2Vec2Model


class Wav2Vec2Localizer:
    """
    Wav2Vec 2.0-based model for localizing dysfluency events in audio.

    Uses Wav2Vec 2.0 embeddings as a rich acoustic backbone, with a temporal
    attention pooling + classifier head to produce per-frame probabilities.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.3,
        freeze_backbone_epochs: int = 5,
        hidden_dim: int = 256,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for Wav2Vec 2.0.
            dropout: Dropout rate in the temporal head.
            freeze_backbone_epochs: Freeze backbone for first N epochs (0 = never freeze).
            hidden_dim: Hidden dimension for temporal head.
        """
        self.model_name = model_name
        self.dropout_rate = dropout
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.hidden_dim = hidden_dim
        self._model = None

    def _build_model(self):
        """Lazy-build the PyTorch model on first use."""
        model_name = self.model_name
        drop = self.dropout_rate
        hdim = self.hidden_dim

        class _Wav2Vec2Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.wav2vec2 = _wav2vec2_model_class().from_pretrained(model_name)
                w2v2_dim = self.wav2vec2.config.hidden_size  # 768 for base

                # Temporal classifier
                self.classifier = nn.Sequential(
                    nn.Linear(w2v2_dim, hdim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop),
                    nn.Linear(hdim, hdim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop),
                    nn.Linear(hdim // 2, 1),
                )

            def forward(self, waveforms):
                """
                Args:
                    waveforms: (B, max_samples,) raw audio at 16kHz

                Returns:
                    logits: (B, 1, T_frames) per-frame dysfluency logits
                """
                outputs = self.wav2vec2(waveforms)
                # outputs.last_hidden_state: (B, T_frames, w2v2_dim)
                hidden = outputs.last_hidden_state

                # Per-frame classification: (B, T_frames, 1) → (B, 1, T_frames)
                logits = self.classifier(hidden)  # (B, T_frames, 1)
                logits = logits.permute(0, 2, 1)  # (B, 1, T_frames)

                return logits

        self._model = _Wav2Vec2Backbone()

    @property
    def model(self):
        """Access the underlying PyTorch nn.Module (builds on first access)."""
        if self._model is None:
            self._build_model()
        return self._model

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            waveforms: (B, max_samples,) raw audio tensor.

        Returns:
            (B, 1, T_frames) logits.
        """
        return self.model(waveforms)

    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Run inference and return per-frame probabilities."""
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(waveforms)
            probs = torch.sigmoid(logits)
        return probs

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        threshold: float = 0.5,
        max_length_seconds: float = 10.0,
    ) -> List[Tuple[float, float, float]]:
        """
        Predict dysfluency regions from a raw audio array.

        Args:
            audio: 1-D float32 numpy array, values in [-1.0, 1.0].
            sr: Sample rate (must be 16000).
            threshold: Detection threshold.
            max_length_seconds: Max audio length to process.

        Returns:
            List of (start_sec, end_sec, confidence) tuples.
        """
        max_samples = int(max_length_seconds * sr)

        actual_samples = len(audio)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            actual_samples = max_samples
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        # Real audio spans only the first `actual_sec`; everything after it is
        # zero-padding and must never produce reported regions.
        actual_sec = actual_samples / sr

        # To tensor: (1, max_samples)
        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        probs = self.predict_proba(tensor)
        probs_np = probs.squeeze().cpu().numpy()

        # Wav2Vec2 subsampling: 320 samples per frame at 16kHz = 20ms/frame
        frame_duration = 320 / sr

        regions = []
        in_region = False
        start_frame = 0
        max_conf = 0.0

        for i, p in enumerate(probs_np):
            if p >= threshold and not in_region:
                in_region = True
                start_frame = i
                max_conf = p
            elif p >= threshold and in_region:
                max_conf = max(max_conf, p)
            elif p < threshold and in_region:
                in_region = False
                start_sec = start_frame * frame_duration
                end_sec = i * frame_duration
                regions.append((start_sec, end_sec, float(max_conf)))

        if in_region:
            start_sec = start_frame * frame_duration
            end_sec = len(probs_np) * frame_duration
            regions.append((start_sec, end_sec, float(max_conf)))

        # Clamp regions to the real audio: drop those starting in the padded
        # tail, and truncate any that extend past the audio's true end.
        return [
            (start, min(end, actual_sec), conf)
            for start, end, conf in regions
            if start < actual_sec
        ]

    def freeze_backbone(self):
        """Freeze Wav2Vec2 backbone parameters."""
        for param in self.model.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze Wav2Vec2 backbone parameters."""
        for param in self.model.wav2vec2.parameters():
            param.requires_grad = True

    def save(self, path: str) -> None:
        """
        Save model weights to a file.

        Args:
            path: Output path (.pt file).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "dropout": self.dropout_rate,
            "hidden_dim": self.hidden_dim,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "Wav2Vec2Localizer":
        """
        Load a trained model from a checkpoint file.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            Wav2Vec2Localizer instance with loaded weights.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            model_name=checkpoint["model_name"],
            dropout=checkpoint["dropout"],
            hidden_dim=checkpoint["hidden_dim"],
            freeze_backbone_epochs=checkpoint.get("freeze_backbone_epochs", 5),
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        return instance

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"Wav2Vec2Localizer(model={self.model_name}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Wav2Vec2 Localizer — Self Test ===")

    model = Wav2Vec2Localizer(model_name="facebook/wav2vec2-base")
    print(f"Model: {model}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Forward pass with random waveform
    batch = torch.randn(2, 160000)  # 2 x 10s at 16kHz
    logits = model.forward(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")

    assert logits.shape[0] == 2, f"Batch dim mismatch: {logits.shape}"
    assert logits.shape[1] == 1, f"Channel dim mismatch: {logits.shape}"

    # Predict probabilities
    probs = model.predict_proba(batch)
    print(f"Prob range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")

    # Single-sample predict
    audio = np.random.randn(160000).astype(np.float32)
    regions = model.predict(audio, threshold=0.3)
    print(f"Detected regions: {len(regions)}")
    for start, end, conf in regions:
        print(f"  {start:.2f}s – {end:.2f}s (conf={conf:.3f})")

    # Save/load roundtrip
    test_path = "/tmp/test_w2v2_localizer.pt"
    model.save(test_path)
    loaded = Wav2Vec2Localizer.from_pretrained(test_path)
    print(f"Loaded model: {loaded}")

    os.remove(test_path)
    print("=== Self test passed ===")
