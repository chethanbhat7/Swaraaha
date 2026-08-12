"""Shared-backbone multitask Wav2Vec2 classifier.

One Wav2Vec2Model backbone feeds five per-class binary heads (one per
dysfluency class). All heads are trained jointly so the shared representation
helps rare classes and memory scales with one backbone instead of five.

Architecture:
    Wav2Vec2Model -> mean-pool(time) -> (B, 768) -> heads[class] -> (B, 2)

API mirrors BaseWav2VecClassifier (model/classification/__init__.py):
predict() uses the same softmax/threshold/confidence convention.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.config.defaults import DYSFLUENCY_CLASSES


def _wav2vec2_model_class():
    """Resolve the Wav2Vec2 backbone class (lazily, for testability)."""
    from transformers import Wav2Vec2Model

    return Wav2Vec2Model


class MultiTaskWav2VecClassifier:
    """Shared-backbone multitask classifier with one head per dysfluency class."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = 768,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name: HuggingFace Wav2Vec2 model identifier.
            hidden_dim: Hidden dimension of each per-class head.
            class_names: Head keys (default: DYSFLUENCY_CLASSES order).
        """
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.class_names = list(class_names or DYSFLUENCY_CLASSES)
        self._model = None

    def _build_model(self):
        """Lazy-build the PyTorch model on first use."""
        model_name = self.model_name
        hdim = self.hidden_dim
        class_names = list(self.class_names)

        class _MultiTaskBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.wav2vec2 = _wav2vec2_model_class().from_pretrained(model_name)
                w2v2_dim = self.wav2vec2.config.hidden_size  # 768 for base
                self.heads = nn.ModuleDict({
                    name: nn.Sequential(
                        nn.Linear(w2v2_dim, hdim),
                        nn.Tanh(),
                        nn.Linear(hdim, 2),
                    )
                    for name in class_names
                })

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values).last_hidden_state  # (B, T, d)
                pooled = hidden.mean(dim=1)  # (B, d)
                return {name: head(pooled) for name, head in self.heads.items()}

        self._model = _MultiTaskBackbone()
        self.class_names = list(self._model.heads.keys())

    @property
    def model(self) -> nn.Module:
        """Access the underlying PyTorch nn.Module (builds on first access)."""
        if self._model is None:
            self._build_model()
        return self._model

    def forward(self, input_values) -> Dict[str, torch.Tensor]:
        """Return {class_name: (B, 2) logits} for one forward pass."""
        return self.model(input_values)

    def forward_head(self, class_name: str, input_values) -> torch.Tensor:
        """Return (B, 2) logits for a single head."""
        hidden = self.model.wav2vec2(input_values).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.model.heads[class_name](pooled)

    def predict(self, audio_tensor, threshold: float = 0.5) -> Dict[str, Tuple[int, float]]:
        """Classify one audio sample with all heads.

        Returns {class_name: (label, confidence)} using the same convention as
        BaseWav2VecClassifier.predict: softmax over 2 logits, label=1 when
        prob_present >= threshold, confidence = P(predicted class).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits = self.forward(audio_tensor)
            results = {}
            for name, lg in logits.items():
                probs = torch.softmax(lg, dim=-1)
                prob_present = probs[0, 1].item()
                label = 1 if prob_present >= threshold else 0
                confidence = prob_present if label == 1 else 1.0 - prob_present
                results[name] = (label, confidence)
        return results

    def save(self, path: str) -> None:
        """Save weights and architecture metadata to a .pt file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "class_names": self.class_names,
            "hidden_dim": self.hidden_dim,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "MultiTaskWav2VecClassifier":
        """Load a model saved by ``save`` (strict state-dict load)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            model_name=checkpoint["model_name"],
            hidden_dim=checkpoint.get("hidden_dim", 768),
            class_names=checkpoint.get("class_names"),
        )
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        instance.model.load_state_dict(state_dict, strict=True)
        return instance

    def __repr__(self):
        return f"MultiTaskWav2VecClassifier(model={self.model_name}, heads={len(self.class_names)})"
