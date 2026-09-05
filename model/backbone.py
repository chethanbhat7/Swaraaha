"""Resolve the HuggingFace model class for a backbone model id.

Training and the runtime registry must build the same backbone architecture
from a ``model_name``. Wav2Vec 2.0, WavLM, and HuBERT are separate
transformers classes, so the model id decides which class to instantiate.
"""


def backbone_model_class(model_name: str):
    """Return the transformers model class for ``model_name``.

    WavLM and HuBERT match on the model id; everything else falls back to
    Wav2Vec2Model. Imports are lazy (importing transformers is slow).
    """
    from transformers import HubertModel, Wav2Vec2Model, WavLMModel

    low = model_name.lower()
    if "wavlm" in low:
        return WavLMModel
    if "hubert" in low:
        return HubertModel
    return Wav2Vec2Model