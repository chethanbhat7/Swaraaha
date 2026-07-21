"""
Hybrid Combiner Model for Swaraaha.

Combines outputs from 5 individual Wav2Vec 2.0 binary classifiers
into a unified multi-class prediction using a learned meta-classifier (MLP).

Design decision: Learned meta-classifier (MLP) over weighted voting or stacking.
Rationale: The MLP can learn non-linear relationships between classifier outputs
and is more flexible than simple weighted voting, while being simpler to train
than full stacking with cross-validation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.classification import DYSFLUENCY_CLASSES, NUM_CLASSES
from model.config.defaults import COMBINER_DROPOUT, COMBINER_HIDDEN_DIM


class CombinerMLP(nn.Module):
    """
    Small MLP that takes concatenated logits from 5 base classifiers
    and produces final predictions for all 5 dysfluency classes.

    Architecture:
        Input:  5 logits (one per base classifier)
        Hidden: COMBINER_HIDDEN_DIM units, ReLU, dropout
        Output: 5 units (one per class), sigmoid
    """

    def __init__(
        self,
        input_dim: int = NUM_CLASSES,
        hidden_dim: int = COMBINER_HIDDEN_DIM,
        output_dim: int = NUM_CLASSES,
        dropout: float = COMBINER_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, 5] concatenated logits from base classifiers.
        Returns:
            [batch_size, 5] probabilities (one per dysfluency class).
        """
        return self.net(logits)


class HybridClassifier:
    """
    Hybrid model that combines 5 trained Wav2Vec 2.0 binary classifiers
    using a learned MLP meta-classifier.

    Pipeline:
        1. Run input audio through each base classifier → 5 logit pairs
        2. Extract the "present" logit from each → 5 logits
        3. Feed into CombinerMLP → 5 probabilities
        4. Threshold at 0.5 for binary predictions
    """

    def __init__(
        self,
        base_classifiers: Optional[List] = None,
        combiner: Optional[CombinerMLP] = None,
        model_name: str = "facebook/wav2vec2-base",
    ):
        """
        Args:
            base_classifiers: List of 5 trained BaseWav2VecClassifier instances,
                              in order of DYSFLUENCY_CLASSES.
            combiner: Trained CombinerMLP instance.
            model_name: Wav2Vec2 model name (used when creating unfitted base classifiers).
        """
        self.model_name = model_name

        if base_classifiers is None:
            from model.classification.block import BlockClassifier
            from model.classification.interjection import InterjectionClassifier
            from model.classification.prolongation import ProlongationClassifier
            from model.classification.soundrep import SoundRepClassifier
            from model.classification.wordrep import WordRepClassifier

            cls_map = {
                "prolongation": ProlongationClassifier,
                "block": BlockClassifier,
                "soundrep": SoundRepClassifier,
                "wordrep": WordRepClassifier,
                "interjection": InterjectionClassifier,
            }
            self.base_classifiers = [cls_map[c](model_name) for c in DYSFLUENCY_CLASSES]
        else:
            assert len(base_classifiers) == NUM_CLASSES
            self.base_classifiers = base_classifiers

        self.combiner = combiner or CombinerMLP()

    def forward(
        self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all base classifiers and the combiner.

        Args:
            input_values: [batch_size, sequence_length] raw waveform at 16kHz.
            attention_mask: Optional attention mask.

        Returns:
            [batch_size, 5] probabilities for each dysfluency class.
        """
        logits_list = []
        for clf in self.base_classifiers:
            logits = clf.forward(input_values, attention_mask)  # [batch, 2]
            logits_list.append(logits[:, 1])  # take "present" logit

        combined = torch.stack(logits_list, dim=-1)  # [batch, 5]
        return self.combiner(combined)

    def predict(
        self, audio_tensor: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, Tuple[int, float]]:
        """
        Run inference on a single audio sample.

        Args:
            audio_tensor: [sequence_length] or [1, sequence_length].
            threshold: Decision threshold for binary prediction.

        Returns:
            Dict mapping class_name → (label, confidence).
            label: 0 = not present, 1 = present.
        """
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        self.combiner.eval()
        for clf in self.base_classifiers:
            clf.model.eval()

        with torch.no_grad():
            probs = self.forward(audio_tensor)  # [1, 5]
            probs = probs.squeeze(0)  # [5]

        results = {}
        for i, cls_name in enumerate(DYSFLUENCY_CLASSES):
            conf = probs[i].item()
            label = 1 if conf >= threshold else 0
            results[cls_name] = (label, conf)

        return results

    def save(self, path: str) -> None:
        """
        Save the hybrid model (all base classifiers + combiner) to a checkpoint.

        Args:
            path: Output path (.pt file).
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        base_state = {}
        for clf in self.base_classifiers:
            base_state[clf.class_name] = {
                "model_state_dict": clf.model.state_dict(),
                "model_name": clf.model_name,
            }

        torch.save({
            "base_classifiers": base_state,
            "combiner_state_dict": self.combiner.state_dict(),
            "model_name": self.model_name,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str) -> "HybridClassifier":
        """
        Load a trained hybrid model from a checkpoint.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            HybridClassifier instance with loaded weights.
        """
        from model.classification.block import BlockClassifier
        from model.classification.interjection import InterjectionClassifier
        from model.classification.prolongation import ProlongationClassifier
        from model.classification.soundrep import SoundRepClassifier
        from model.classification.wordrep import WordRepClassifier

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        cls_map = {
            "prolongation": ProlongationClassifier,
            "block": BlockClassifier,
            "soundrep": SoundRepClassifier,
            "wordrep": WordRepClassifier,
            "interjection": InterjectionClassifier,
        }

        base_classifiers = []
        for cls_name in DYSFLUENCY_CLASSES:
            clf_cls = cls_map[cls_name]
            info = checkpoint["base_classifiers"][cls_name]
            clf = clf_cls(model_name=info["model_name"])
            clf.model.load_state_dict(info["model_state_dict"])
            base_classifiers.append(clf)

        combiner = CombinerMLP()
        combiner.load_state_dict(checkpoint["combiner_state_dict"])

        return cls(
            base_classifiers=base_classifiers,
            combiner=combiner,
            model_name=checkpoint["model_name"],
        )

    def __repr__(self):
        base_names = [c.class_name for c in self.base_classifiers]
        return f"HybridClassifier(base={base_names}, combiner={self.combiner.__class__.__name__})"
