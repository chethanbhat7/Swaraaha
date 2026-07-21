"""CNN model for per-frame dysfluency detection on mel-spectrograms."""

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """
    CNN that takes a mel-spectrogram and outputs per-frame dysfluency probabilities.

    Architecture:
        4 conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool2d)
        Global average pooling over frequency
        FC head -> sigmoid per time frame
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self._block(in_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(1, 2),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, n_mels, time_frames] mel-spectrogram.
        Returns:
            [batch, time_frames] per-frame probabilities.
        """
        x = self.conv_blocks(x)
        x = self.head(x)
        return x

    def predict_frames(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> list[list[tuple[float, float]]]:
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        regions = []
        for b in range(probs.shape[0]):
            positive = probs[b] >= threshold
            regions.append(self._extract_regions(positive))
        return regions

    @staticmethod
    def _extract_regions(mask: torch.Tensor) -> list[tuple[float, float]]:
        diffs = mask.int().diff()
        starts = (diffs == 1).nonzero(as_tuple=True)[0]
        ends = (diffs == -1).nonzero(as_tuple=True)[0]

        if mask[0]:
            starts = torch.cat([torch.tensor([0]), starts])
        if mask[-1]:
            ends = torch.cat([ends, torch.tensor([len(mask) - 1])])

        return [(s.item(), e.item()) for s, e in zip(starts, ends)]
