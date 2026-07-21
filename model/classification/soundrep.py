from model.classification import BaseWav2VecClassifier, DYSFLUENCY_CLASSES


class SoundRepClassifier(BaseWav2VecClassifier):
    """
    Binary Wav2Vec 2.0 classifier for sound repetition dysfluency.

    Sound repetition: repeating a sound or syllable (e.g., "b-b-ball").
    Binary output: sound repetition present (1) vs not present (0).
    """

    class_name: str = "soundrep"
    class_idx: int = DYSFLUENCY_CLASSES.index("soundrep")

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name=model_name)
