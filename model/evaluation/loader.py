"""
Robust checkpoint loading for Swaraaha evaluation.

Handles every checkpoint format produced by the training scripts and the
model registry:

- Classifiers
    * registry format: saved by ``BaseWav2VecClassifier.save`` and contains
      a ``model_name`` key.
    * training format: saved by ``model.training.utils.save_checkpoint`` and
      contains a ``model_state_dict`` key. Keys may carry a ``_orig_mod.``
      prefix when the model was wrapped with ``torch.compile``; the prefix is
      stripped before loading.
- Localizers
    * own format: saved by ``CNNSpectrogramLocalizer.save`` /
      ``Wav2Vec2Localizer.save``.
    * training format: saved by ``save_checkpoint`` (CNN only).
"""

from typing import Dict, Optional


def _strip_compile_prefix(state_dict: Dict) -> Dict:
    """Remove the ``_orig_mod.`` prefix introduced by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_classifier(class_name: str, model_path: str):
    """
    Load a binary classifier checkpoint in either registry or training format.

    Args:
        class_name: Dysfluency class name (e.g. "prolongation").
        model_path: Path to the ``.pt`` checkpoint.

    Returns:
        A ``BaseWav2VecClassifier`` instance (subclass) with weights loaded.
    """
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_name" in checkpoint:
        from model.classification import get_classifier_class

        return get_classifier_class(class_name).from_pretrained(model_path)

    from transformers import Wav2Vec2ForSequenceClassification

    from model.classification import DYSFLUENCY_CLASSES
    from model.fingerprint import model_name_from_path

    model_name = model_name_from_path(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)

    from model.classification import BaseWav2VecClassifier

    instance = BaseWav2VecClassifier.__new__(BaseWav2VecClassifier)
    instance.model_name = model_name
    instance._model = model
    instance.class_name = class_name
    instance.class_idx = DYSFLUENCY_CLASSES.index(class_name)
    return instance


def load_multitask(model_path):
    """Load a multitask classifier checkpoint.

    Dispatch order:
      1. CNN own format (``aggregator`` key in checkpoint)
      2. wav2vec2 own format (``model_name`` key in checkpoint)
      3. training-format resume checkpoints (``args`` in checkpoint)
    """
    import torch

    from model.classification.multitask import MultiTaskWav2VecClassifier

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'aggregator' in checkpoint:
        from model.classification.cnn_multitask import CNNMultitaskClassifier
        return CNNMultitaskClassifier.from_pretrained(model_path)
    if 'model_name' in checkpoint:
        return MultiTaskWav2VecClassifier.from_pretrained(model_path)
    args_ckpt = checkpoint.get('args', {})
    if args_ckpt.get('aggregator') is not None:
        from model.classification.cnn_multitask import CNNMultitaskClassifier
        instance = CNNMultitaskClassifier(
            n_mels=args_ckpt.get('n_mels', 128),
            hop_length=args_ckpt.get('hop_length', 512),
            n_fft=args_ckpt.get('n_fft', 2048),
            hidden_dim=args_ckpt.get('hidden_dim', 128),
            dropout=args_ckpt.get('dropout', 0.4),
            class_names=args_ckpt.get('class_names'),
            aggregator=args_ckpt['aggregator'],
            num_lstm_layers=args_ckpt.get('num_lstm_layers', 1),
            num_transformer_layers=args_ckpt.get('num_transformer_layers', 1),
        )
        state_dict = _strip_compile_prefix(checkpoint['model_state_dict'])
        instance.model.load_state_dict(state_dict, strict=True)
        return instance
    from model.fingerprint import model_name_from_path, parse_fingerprint_from_path

    model_name = model_name_from_path(model_path)
    try:
        params = parse_fingerprint_from_path(model_path)
        hidden_dim = params.get("hidden_dim", 768)
    except (ValueError, KeyError):
        hidden_dim = 768
    instance = MultiTaskWav2VecClassifier(model_name=model_name, hidden_dim=hidden_dim,
                                          class_names=None)
    state_dict = _strip_compile_prefix(checkpoint['model_state_dict'])
    instance.model.load_state_dict(state_dict, strict=True)
    return instance


def load_localizer(localizer_type: str, model_path: str):
    """
    Load a localization model in either its own or the training format.

    Args:
        localizer_type: "cnn" or "wav2vec2".
        model_path: Path to the ``.pt`` checkpoint.

    Returns:
        A ``CNNSpectrogramLocalizer`` or ``Wav2Vec2Localizer`` instance.
    """
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if localizer_type == "cnn":
        from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer

        if "n_mels" in checkpoint:
            return CNNSpectrogramLocalizer.from_pretrained(model_path)

        # Training-format checkpoint (model_state_dict, no config keys).
        n_mels = checkpoint.get("args", {}).get("n_mels", 128)
        instance = CNNSpectrogramLocalizer(n_mels=n_mels)
        state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
        instance.model.load_state_dict(state_dict, strict=True)
        return instance

    if localizer_type == "wav2vec2":
        from model.localization.wav2vec2_localizer import Wav2Vec2Localizer

        if "model_name" in checkpoint and "hidden_dim" in checkpoint:
            return Wav2Vec2Localizer.from_pretrained(model_path)

        raise ValueError(
            "Wav2Vec2 localizer training-format checkpoints are not supported; "
            "re-save the checkpoint with Wav2Vec2Localizer.save()."
        )

    raise ValueError(f"Unknown localizer type: {localizer_type}")


def registry_paths(registry_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Read model paths from the model registry.

    Args:
        registry_path: Path to ``registry.json`` (default: model/registry.json).

    Returns:
        Dict with "classification" and "localization" sections, each mapping
        model key → resolved absolute path.
    """
    import json
    import os

    path = registry_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "registry.json"
    )
    with open(path) as f:
        registry = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def resolve(section: Dict[str, str]) -> Dict[str, str]:
        return {
            name: os.path.join(project_root, rel) if not os.path.isabs(rel) else rel
            for name, rel in section.items()
        }

    return {
        "classification": resolve(registry.get("classification", {})),
        "localization": resolve(registry.get("localization", {})),
        "thresholds": registry.get("thresholds", {}),
    }


def model_info_from_path(model_path: str) -> Dict:
    """
    Extract the model fingerprint and parsed hyperparameters from a
    fingerprint-encoded checkpoint path.

    Args:
        model_path: Path to a checkpoint file.

    Returns:
        Dict with a "fingerprint" string, the HF "model_name" and every parsed
        hyperparameter (epochs, batch_size, lr, ...). For paths without a
        fingerprint (e.g. localizer_best.pt), returns {"filename": basename}.
    """
    import os

    from model.fingerprint import parse_fingerprint_from_path

    try:
        params = parse_fingerprint_from_path(model_path)
    except ValueError:
        return {"filename": os.path.basename(model_path)}

    info = {"fingerprint": os.path.basename(model_path)}
    for suffix in ("_best.pt", "_final.pt", "_checkpoint.pt", "_log.csv", ".pt"):
        if info["fingerprint"].endswith(suffix):
            info["fingerprint"] = info["fingerprint"][: -len(suffix)]
            break
    info.update(params)
    return info
