"""MultiTaskRunner and CNNMultiTaskRunner — shared-backbone classification."""

from typing import Any, Dict, List, Optional

from model.config.defaults import AUDIO_DURATION_SECONDS, DYSFLUENCY_CLASSES, SAMPLE_RATE

from ._utils import (
    _audio_is_empty,
    _chunk_audio,
    _empty_classifier_output,
    _preprocess_audio,
)


def _get_reg():
    import model.registry
    return model.registry


class MultiTaskRunner:
    """Registry wrapper for the shared-backbone multitask classifier.

    analyze() runs ONE forward pass and returns per-class
    {label, confidence, prob_present, prob_not_present} plus a summary.
    """

    def __init__(self):
        self._model = None
        self._thresholds: Dict[str, float] = {}

    REGISTRY_KEY = "classification_multitask"

    def _load(self) -> None:
        _reg = _get_reg()
        registry = _reg._load_registry()
        self._model, self._thresholds = _reg._load_multitask_registry_entry(
            registry, self.REGISTRY_KEY,
        )

    @staticmethod
    def _empty_result(names: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            name: _empty_classifier_output(False) for name in names
        }
        results["summary"] = {
            "detected": [],
            "primary": names[0] if names else None,
        }
        return results

    def _preprocess(self, audio):
        return _preprocess_audio(audio)

    def analyze(self, audio, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Run one forward pass and classify every class.

        Mirrors ``ClassifierRunner.analyze``: an explicit ``threshold`` overrides
        the loaded per-class thresholds for every class. When ``threshold``
        is None (the default), the loaded per-class thresholds apply
        (registry ``classification_multitask.thresholds``, ``thresholds_path``,
        or a sibling ``multitask_thresholds.json`` next to the model file),
        falling back to 0.5 for any class without a configured threshold.

        Audio longer than the model's max_length_seconds is processed in
        overlapping chunks and the per-class probabilities are merged
        (max prob_present across chunks).

        Returns:
            {class_name: {label, confidence, prob_present, prob_not_present},
             summary: {detected, primary}}
        """
        import numpy as np
        import torch

        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        if _audio_is_empty(audio):
            names = (
                list(self._model.class_names)
                if self._model is not None
                else [n for n in _get_reg()._load_registry().get("classification", {}) if n in DYSFLUENCY_CLASSES]
            )
            return self._empty_result(names)

        if self._model is None:
            self._load()

        from model.data.preprocessing import load_audio_input

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        max_len = getattr(self._model, "max_length_seconds", AUDIO_DURATION_SECONDS)
        audio_sec = len(audio_array) / SAMPLE_RATE

        if audio_sec <= max_len + 0.1:
            return self._analyze_chunk(audio, threshold)

        merged: Dict[str, Dict[str, float]] = {}
        for chunk, start_sample in _chunk_audio(audio_array, max_len, sr=SAMPLE_RATE):
            chunk_result = self._analyze_chunk(chunk, threshold)
            for name in list(chunk_result.keys()):
                if name == "summary":
                    continue
                if name not in merged:
                    merged[name] = chunk_result[name]
                elif chunk_result[name]["prob_present"] > merged[name]["prob_present"]:
                    merged[name] = chunk_result[name]

        detected = [name for name, r in merged.items() if r["label"] == 1]
        primary = max(merged.items(), key=lambda kv: kv[1]["prob_present"])[0] if merged else None
        merged["summary"] = {"detected": detected, "primary": primary}
        return merged

    def _analyze_chunk(self, audio, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Classify a single chunk (up to max_length_seconds)."""
        import torch

        tensor = self._preprocess(audio)
        self._model.model.eval()
        with torch.no_grad():
            logits = self._model.forward(tensor)

        results: Dict[str, Any] = {}
        for name, lg in logits.items():
            probs = torch.softmax(lg, dim=-1)
            prob_present = probs[0, 1].item()
            prob_not_present = probs[0, 0].item()
            thr = threshold if threshold is not None else self._thresholds.get(name, 0.5)
            label = 1 if prob_present >= thr else 0
            confidence = prob_present if label == 1 else prob_not_present
            results[name] = {
                "label": label,
                "confidence": confidence,
                "prob_present": prob_present,
                "prob_not_present": prob_not_present,
            }

        detected = [name for name, r in results.items() if r["label"] == 1]
        primary = max(results.items(), key=lambda kv: kv[1]["prob_present"])[0]
        results["summary"] = {"detected": detected, "primary": primary}
        return results

    def saliency(self, audio, max_length_seconds: Optional[float] = None) -> "torch.Tensor":
        """Per-frame per-class prob_present saliency for the whole audio.

        Audio longer than max_length_seconds is processed in overlapping chunks
        and concatenated. Returns a tensor of shape (1, T, num_classes).
        """
        if _audio_is_empty(audio):
            raise RuntimeError("Cannot compute saliency for empty audio")
        if self._model is None:
            self._load()

        if max_length_seconds is None:
            max_length_seconds = getattr(self._model, "max_length_seconds", AUDIO_DURATION_SECONDS)

        import torch
        from model.data.preprocessing import load_audio_input

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        audio_sec = len(audio_array) / SAMPLE_RATE

        if audio_sec <= max_length_seconds + 0.1:
            return self._saliency_chunk(audio, max_length_seconds)

        chunks = list(_chunk_audio(audio_array, max_length_seconds, sr=SAMPLE_RATE))
        total_samples = len(audio_array)
        total_frames = total_samples // 320
        num_classes = len(getattr(self._model, "class_names", DYSFLUENCY_CLASSES))
        full_sal = torch.zeros(1, total_frames, num_classes)

        for chunk, start_sample in chunks:
            sal = self._saliency_chunk(chunk, max_length_seconds)
            start_frame = start_sample // 320
            sal_np = sal.squeeze(0).cpu().numpy()
            n_frames_chunk = sal_np.shape[0]
            end_frame = min(start_frame + n_frames_chunk, total_frames)
            actual = end_frame - start_frame
            if actual > 0:
                full_sal[0, start_frame:end_frame, :] = torch.tensor(
                    sal_np[:actual], dtype=torch.float32
                )

        return full_sal

    def _saliency_chunk(self, audio, max_length_seconds: float) -> "torch.Tensor":
        """Compute saliency for a single chunk (≤ max_length_seconds)."""
        import librosa
        import torch
        from model.data.preprocessing import load_audio_input

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        _, trim_index = librosa.effects.trim(audio_array, top_db=25)
        trim_offset_samples = trim_index[0]
        trim_offset_frames = int(round(trim_offset_samples / 320))

        tensor = _preprocess_audio(audio, max_length_seconds=max_length_seconds)
        sal = self._model.saliency(tensor)

        if trim_offset_frames > 0:
            shifted = torch.zeros_like(sal)
            if trim_offset_frames < sal.shape[1]:
                shifted[:, trim_offset_frames:, :] = sal[:, :-trim_offset_frames, :]
            sal = shifted

        return sal

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class CNNMultiTaskRunner(MultiTaskRunner):
    """Registry-backed CNN multitask classifier.

    Reuses ``MultiTaskRunner.analyze``; only the preprocessing hook is
    overridden to feed mel-spectrograms instead of raw waveforms.
    """

    REGISTRY_KEY = 'classification_multitask_cnn'

    def _preprocess(self, audio):
        import torch

        from model.data.preprocessing import (
            clean_audio,
            generate_mel_spectrogram,
            load_audio_input,
            pad_to_length,
            spectrogram_to_image_array,
        )

        max_length = getattr(self._model, "max_length_seconds", AUDIO_DURATION_SECONDS)
        max_frames = int(SAMPLE_RATE * max_length) // self._model.hop_length + 1
        audio = clean_audio(load_audio_input(audio, sr=SAMPLE_RATE), sr=SAMPLE_RATE)
        spec = generate_mel_spectrogram(
            audio, sr=SAMPLE_RATE, n_mels=self._model.n_mels,
            hop_length=self._model.hop_length,
            n_fft=getattr(self._model, "n_fft", 2048),
        )
        spec = pad_to_length(spec, max_frames, axis=1, pad_value=float(spec.min()))
        spec = spectrogram_to_image_array(spec)
        return torch.from_numpy(spec).float().unsqueeze(0)
