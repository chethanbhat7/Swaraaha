from model.classification import BaseWav2VecClassifier, DYSFLUENCY_CLASSES


class ProlongationClassifier(BaseWav2VecClassifier):
    """
    Binary Wav2Vec 2.0 classifier for prolongation dysfluency.

    Prolongation: abnormal stretching of a sound (e.g., "sssssnake").
    Binary output: prolongation present (1) vs not present (0).
    """

    class_name: str = "prolongation"
    class_idx: int = DYSFLUENCY_CLASSES.index("prolongation")

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name=model_name)
