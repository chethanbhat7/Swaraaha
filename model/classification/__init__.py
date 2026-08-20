"""Classification models for dysfluency detection."""

from model.classification.base import BaseWav2VecClassifier
from model.config.defaults import DYSFLUENCY_CLASSES

NUM_CLASSES = len(DYSFLUENCY_CLASSES)


def _make_classifier_class(class_name: str) -> type:
    """Create a concrete classifier class for a dysfluency type."""
    idx = DYSFLUENCY_CLASSES.index(class_name)

    class _Classifier(BaseWav2VecClassifier):
        def __init__(self, model_name="facebook/wav2vec2-base"):
            super().__init__(model_name=model_name)

    _Classifier.class_name = class_name
    _Classifier.class_idx = idx
    _Classifier.__name__ = f"{class_name.capitalize()}Classifier"
    _Classifier.__qualname__ = _Classifier.__name__
    _Classifier.__doc__ = f"Binary classifier for {class_name} dysfluency."
    return _Classifier


# Pre-built classifier classes (same API as before)
ProlongationClassifier = _make_classifier_class("prolongation")
BlockClassifier = _make_classifier_class("block")
SoundRepClassifier = _make_classifier_class("soundrep")
WordRepClassifier = _make_classifier_class("wordrep")
InterjectionClassifier = _make_classifier_class("interjection")

_CLASSIFIER_CLASSES = {
    "prolongation": ProlongationClassifier,
    "block": BlockClassifier,
    "soundrep": SoundRepClassifier,
    "wordrep": WordRepClassifier,
    "interjection": InterjectionClassifier,
}


def get_classifier_class(class_name: str):
    """Return the classifier class for a dysfluency class."""
    if class_name not in _CLASSIFIER_CLASSES:
        raise ValueError(
            f"Unknown dysfluency class: {class_name!r}. "
            f"Available: {list(_CLASSIFIER_CLASSES)}"
        )
    return _CLASSIFIER_CLASSES[class_name]
