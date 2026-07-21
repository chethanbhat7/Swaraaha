from model.classification import BaseWav2VecClassifier, DYSFLUENCY_CLASSES


class InterjectionClassifier(BaseWav2VecClassifier):
    """
    Binary Wav2Vec 2.0 classifier for interjection dysfluency.

    Interjection: filler words/sounds (e.g., "um", "uh", "like" as filler).
    Binary output: interjection present (1) vs not present (0).
    """

    class_name: str = "interjection"
    class_idx: int = DYSFLUENCY_CLASSES.index("interjection")

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name=model_name)
