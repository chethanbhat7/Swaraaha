from model.classification import BaseWav2VecClassifier, DYSFLUENCY_CLASSES


class BlockClassifier(BaseWav2VecClassifier):
    """
    Binary Wav2Vec 2.0 classifier for block dysfluency.

    Block: involuntary pause where airflow/sound is stopped (e.g., "--ball").
    Binary output: block present (1) vs not present (0).
    """

    class_name: str = "block"
    class_idx: int = DYSFLUENCY_CLASSES.index("block")

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name=model_name)
