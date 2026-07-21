from model.classification import BaseWav2VecClassifier, DYSFLUENCY_CLASSES


class WordRepClassifier(BaseWav2VecClassifier):
    """
    Binary Wav2Vec 2.0 classifier for word repetition dysfluency.

    Word repetition: repeating whole words (e.g., "I-I-I want").
    Binary output: word repetition present (1) vs not present (0).
    """

    class_name: str = "wordrep"
    class_idx: int = DYSFLUENCY_CLASSES.index("wordrep")

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name=model_name)
