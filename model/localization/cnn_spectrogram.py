"""
CNN-based dysfluency localization model for Swaraaha.

Takes a mel-spectrogram image and outputs a per-frame probability map
indicating where dysfluency occurs in the audio timeline.

Architecture: small CNN backbone → upsampling → per-frame sigmoid.
Input:  (B, 1, n_mels, T)  — grayscale spectrogram
Output: (B, 1, T)          — per-frame dysfluency probability
"""

import os
from typing import List, Tuple

import numpy as np

from model.config.defaults import SAMPLE_RATE


class CNNSpectrogramLocalizer:
    """
    CNN model for localizing dysfluency events in spectrograms.

    Uses convolutional layers to extract features from the spectrogram,
    then upsamples back to the original time resolution to produce
    a per-frame probability mask.
    """

    def __init__(
        self,
        n_mels: int = 128,
        in_channels: int = 1,
        dropout: float = 0.4,
    ):
        """
        Args:
            n_mels: Height of input spectrogram (mel frequency bins).
            in_channels: Number of input channels (1 for grayscale).
            dropout: Dropout rate between conv layers.
        """
        self.n_mels = n_mels
        self.in_channels = in_channels
        self.dropout_rate = dropout
        self._model = None

    def _build_model(self):
        """Lazy-build the PyTorch model on first use."""
        import torch
        import torch.nn as nn

        n_mels = self.n_mels
        in_ch = self.in_channels
        drop = self.dropout_rate

        class _CNNBackbone(nn.Module):
            """
            CNN that processes a spectrogram and outputs per-frame logits.

            Conv layers reduce the frequency (mel) dimension while
            preserving the time dimension via padding. Final adaptive
            pooling ensures output time resolution matches input.
            """

            def __init__(self):
                super().__init__()
                # Block 1: input (B, 1, n_mels, T) → (B, 32, n_mels/2, T)
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_ch, 32, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1)),  # halve freq, keep time
                    nn.Dropout2d(drop),
                )
                # Block 2: (B, 32, n_mels/2, T) → (B, 64, n_mels/4, T)
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1)),
                    nn.Dropout2d(drop),
                )
                # Block 3: (B, 64, n_mels/4, T) → (B, 128, n_mels/8, T)
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1)),
                    nn.Dropout2d(drop),
                )
                # Block 4: (B, 128, n_mels/8, T) → (B, 128, 1, T)
                self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, None)),  # collapse freq dim
                    nn.Dropout2d(drop),
                )
                # Per-frame classifier: (B, 128, 1, T) → (B, 1, 1, T) → squeeze → (B, 1, T)
                self.classifier = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop),
                    nn.Conv2d(64, 1, kernel_size=(1, 1)),
                )

            def forward(self, x):
                # x: (B, 1, n_mels, T)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                # x shape: (B, 128, 1, T)
                x = self.classifier(x)
                # x shape: (B, 1, 1, T)
                x = x.squeeze(2)
                # x shape: (B, 1, T)
                return x

        self._model = _CNNBackbone()

    @property
    def model(self):
        """Access the underlying PyTorch nn.Module (builds on first access)."""
        if self._model is None:
            self._build_model()
        return self._model

    def forward(self, spectrograms):
        """
        Forward pass through the CNN.

        Args:
            spectrograms: torch.Tensor of shape (B, 1, n_mels, T), float32.

        Returns:
            torch.Tensor of shape (B, 1, T) — raw logits (before sigmoid).
        """
        import torch
        return self.model(spectrograms)

    def predict_proba(self, spectrograms, threshold: float = 0.5):
        """
        Run inference and return per-frame probabilities.

        Args:
            spectrograms: torch.Tensor of shape (B, 1, n_mels, T).
            threshold: Not used here, kept for API consistency.

        Returns:
            torch.Tensor of shape (B, 1, T) — probabilities in [0, 1].
        """
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(spectrograms)
            probs = torch.sigmoid(logits)
        return probs

    def predict(
        self,
        spectrogram: np.ndarray,
        sr: int = SAMPLE_RATE,
        hop_length: int = 512,
        threshold: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        """
        Predict dysfluency regions from a single spectrogram.

        Args:
            spectrogram: numpy array of shape (1, n_mels, T) or (n_mels, T).
            sr: Sample rate of the original audio.
            hop_length: Hop length used during spectrogram generation.
            threshold: Probability threshold for detecting dysfluency.

        Returns:
            List of (start_sec, end_sec, confidence) tuples for each
            detected dysfluency region.
        """
        import torch

        if spectrogram.ndim == 2:
            spectrogram = spectrogram[np.newaxis, ...]  # add channel dim
        if spectrogram.ndim == 3 and spectrogram.shape[0] != 1:
            spectrogram = spectrogram[np.newaxis, ...]

        # To tensor and add batch dim
        tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        probs = self.predict_proba(tensor)
        probs_np = probs.squeeze().cpu().numpy()

        # Threshold and extract contiguous regions
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
                start_sec = start_frame * hop_length / sr
                end_sec = i * hop_length / sr
                regions.append((start_sec, end_sec, float(max_conf)))

        if in_region:
            start_sec = start_frame * hop_length / sr
            end_sec = len(probs_np) * hop_length / sr
            regions.append((start_sec, end_sec, float(max_conf)))

        return regions

    def save(self, path: str) -> None:
        """
        Save model weights to a file.

        Args:
            path: Output path (.pt file).
        """
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_mels": self.n_mels,
            "in_channels": self.in_channels,
            "dropout": self.dropout_rate,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "CNNSpectrogramLocalizer":
        """
        Load a trained model from a checkpoint file.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            CNNSpectrogramLocalizer instance with loaded weights.
        """
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            n_mels=checkpoint["n_mels"],
            in_channels=checkpoint["in_channels"],
            dropout=checkpoint["dropout"],
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        return instance

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"CNNSpectrogramLocalizer(n_mels={self.n_mels}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=== CNN Spectrogram Localizer — Self Test ===")

    model = CNNSpectrogramLocalizer(n_mels=128)
    print(f"Model: {model}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Forward pass with random spectrogram
    batch = torch.randn(2, 1, 128, 200)
    logits = model.forward(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")  # should be (2, 1, 200)

    assert logits.shape == (2, 1, 200), f"Unexpected output shape: {logits.shape}"

    # Predict probabilities
    probs = model.predict_proba(batch)
    print(f"Prob range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")

    # Single-sample predict
    single_spec = np.random.randn(1, 128, 156).astype(np.float32)
    regions = model.predict(single_spec, threshold=0.3)
    print(f"Detected regions: {len(regions)}")
    for start, end, conf in regions:
        print(f"  {start:.2f}s – {end:.2f}s (conf={conf:.3f})")

    # Save/load roundtrip
    test_path = "/tmp/test_localizer.pt"
    model.save(test_path)
    loaded = CNNSpectrogramLocalizer.from_pretrained(test_path)
    print(f"Loaded model: {loaded}")

    # Verify loaded model produces same output (both in eval mode)
    model.model.eval()
    loaded.model.eval()
    logits_orig = model.forward(batch)
    logits_loaded = loaded.forward(batch)
    assert torch.allclose(logits_orig, logits_loaded, atol=1e-5)
    print("Save/load roundtrip passed")

    os.remove(test_path)
    print("=== Self test passed ===")
