"""Model inference wrapper — stub for Phase 1. Will integrate with model/ in Phase 3."""

import numpy as np


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._loaded = False

    def analyze(self, audio: np.ndarray) -> dict:
        """Run classification + localization on audio. Returns structured results."""
        # Placeholder: return mock results for UI development
        return {
            "classifications": {
                "prolongation": (False, 0.12),
                "block": (True, 0.87),
                "soundrep": (False, 0.08),
                "wordrep": (False, 0.05),
                "interjection": (True, 0.72),
            },
            "localizations": [
                (0.5, 1.2, 0.87),
                (3.4, 4.1, 0.72),
            ],
        }
