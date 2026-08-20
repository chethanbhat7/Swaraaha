"""
Model Registry — loads trained models and exposes a clean predict API.

Usage:
    from model.registry import ClassifierRunner, LocalizerRunner, ModelRegistry

    # All classifiers
    clf = ClassifierRunner()
    result = clf.analyze("recording.wav")        # path
    result = clf.analyze(audio_bytes)            # bytes
    result = clf.analyze(np_array, sr=16000)     # numpy array
    # result: {prolongation: {...}, ..., summary: {detected: [...], primary: ...}}

    # Single classifier
    clf = ClassifierRunner("prolongation")
    result = clf.analyze(audio)                  # {label, confidence, prob_present, prob_not_present}

    # Advanced: adds raw logits
    result = clf.analyze_raw(audio)

    # Per-call threshold override
    result = clf.analyze(audio, threshold=0.6)

    # Localizer (regions always; words/syllables when text is provided)
    loc = LocalizerRunner()                      # type comes from registry.json
    result = loc.analyze("recording.wav", text="the cat sat", language="en")
    # result: {regions: [...], words: [...], syllables: [...]}

    # Transcription (Whisper, word-level timestamps + stutter flagging)
    tr = Transcriber()
    result = tr.transcribe("recording.wav")

    # Everything at once — raw audio in, all results out
    m = ModelRegistry()
    all_results = m.run_all("recording.wav", text="the cat sat")
    # all_results: {classification: {...}, localization: {...}, transcription: {...},
    #               multitask: {...}, cnn_multitask: {...}, combined: {...}}
    # combined: localizer regions fused with per-class saliency from the
    # multitask classifier — each region: {start, end, confidence, classes,
    # primary_type, severity, syllables[]}
"""

from model.config.defaults import DYSFLUENCY_CLASSES, FRAME_DURATION, MAX_AUDIO_LENGTH, SAMPLE_RATE
from model.combiner import combine_regions
from model.data.preprocessing import convert_to_wav, generate_mel_spectrogram
from model.localization.ctc_alignment import SimpleForcedAligner
from model.transcription import Transcriber

from ._utils import (
    _REGISTRY_PATH,
    _align_words_syllables,
    _load_classifier,
    _load_multitask_classifier,
    _load_multitask_registry_entry,
    _load_registry,
    _registry_classification_names,
    _resolve_multitask_thresholds,
    _resolve_path,
)
from ._classifier import ClassifierRunner
from ._classifier import ClassifierRunner as Classifier
from ._localizer import LocalizerRunner
from ._localizer import LocalizerRunner as Localizer
from ._multitask import CNNMultiTaskRunner, MultiTaskRunner
from ._multitask import CNNMultiTaskRunner as CNNMultiTaskClassifier
from ._multitask import MultiTaskRunner as MultiTaskClassifier
from ._pipelines import (
    ModelRegistry,
    classify_audio_bytes,
    combine_with_saliency,
    load_audio_16k,
    load_synthesis_config,
    localize_audio_bytes,
    saliency_regions,
)
