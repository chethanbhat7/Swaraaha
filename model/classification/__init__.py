# Swaraaha - Classification Models
# Wav2Vec 2.0 binary classifiers for dysfluency detection

from model.classification.base import BaseWav2VecClassifier

from model.config.defaults import DYSFLUENCY_CLASSES

NUM_CLASSES = len(DYSFLUENCY_CLASSES)


def get_classifier_class(class_name: str):
    """Lazily import and return the classifier class for a dysfluency class."""
    _CLASSIFIER_MODULES = {
        "prolongation": "model.classification.prolongation.ProlongationClassifier",
        "block": "model.classification.block.BlockClassifier",
        "soundrep": "model.classification.soundrep.SoundRepClassifier",
        "wordrep": "model.classification.wordrep.WordRepClassifier",
        "interjection": "model.classification.interjection.InterjectionClassifier",
    }
    if class_name not in _CLASSIFIER_MODULES:
        raise ValueError(
            f"Unknown dysfluency class: {class_name!r}. "
            f"Available: {list(_CLASSIFIER_MODULES)}"
        )
    module_path, cls_name_str = _CLASSIFIER_MODULES[class_name].rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_path), cls_name_str)


# Concrete classifiers (import from submodules to avoid circular imports).
# Use get_classifier_class(name) to resolve a classifier by dysfluency class:
#   from model.classification import get_classifier_class
#   Cls = get_classifier_class("prolongation")
