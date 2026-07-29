# Swaraaha - Classification Models
# Wav2Vec 2.0 binary classifiers for dysfluency detection

from typing import Dict, List, Optional, Tuple

import numpy as np


DYSFLUENCY_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]
NUM_CLASSES = len(DYSFLUENCY_CLASSES)


class BaseWav2VecClassifier:
    """
    Base class for Wav2Vec 2.0 binary classifiers.

    All five classifiers (prolongation, block, soundrep, wordrep, interjection)
    inherit from this class. The only difference is the class label and
    potentially class-specific hyperparameters.

    Subclasses should set:
        - class_name: str (e.g. "prolongation")
        - class_idx: int (0-4, index into DYSFLUENCY_CLASSES)
    """

    class_name: str = ""
    class_idx: int = 0

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        """
        Load pretrained Wav2Vec2 + classification head.

        Args:
            model_name: HuggingFace model identifier or local path.
        """
        from transformers import Wav2Vec2ForSequenceClassification

        self.model_name = model_name
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )

    @property
    def model(self):
        """Access the underlying PyTorch model."""
        return self._model

    def forward(self, input_values, attention_mask=None):
        """
        Forward pass through the model.

        Args:
            input_values: torch.Tensor of shape [batch_size, sequence_length]
            attention_mask: Optional torch.Tensor of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor of shape [batch_size, 1] — single logit per sample.
        """
        return self._model(input_values=input_values, attention_mask=attention_mask).logits

    def predict(self, audio_tensor) -> Tuple[int, float]:
        """
        Run inference on a single audio sample.

        Args:
            audio_tensor: torch.Tensor of shape [1, sequence_length] or [sequence_length].

        Returns:
            Tuple of (label: int, confidence: float).
            label: 0 = not present, 1 = present.
            confidence: probability of the predicted class (0.0 - 1.0).
        """
        import torch

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            logits = self.forward(audio_tensor)
            prob = torch.sigmoid(logits[0, 0]).item()
            label = 1 if prob >= 0.5 else 0
            confidence = prob if label == 1 else 1.0 - prob

        return label, confidence

    def save(self, path: str) -> None:
        """
        Save model weights to a file.

        Args:
            path: Output path (.pt file).
        """
        import torch
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "class_name": self.class_name,
            "class_idx": self.class_idx,
            "model_name": self.model_name,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "BaseWav2VecClassifier":
        """
        Load a trained model from a checkpoint file.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            Classifier instance with loaded weights.
        """
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(model_name=checkpoint["model_name"])
        instance._model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return instance

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"


# Concrete classifiers (import from submodules to avoid circular imports):
#   from model.classification.prolongation import ProlongationClassifier
#   from model.classification.block import BlockClassifier
#   from model.classification.soundrep import SoundRepClassifier
#   from model.classification.wordrep import WordRepClassifier
#   from model.classification.interjection import InterjectionClassifier
