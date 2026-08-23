"""Pre-download whisper-tiny ASR models into HF_HOME so the image ships warm.

Run during docker build. Keeps cold starts deterministic (no HF Hub fetches
at request time after scale-to-zero container recreation).
"""

from transformers import pipeline

MODELS = (
    "openai/whisper-tiny",
    "vasista22/whisper-kannada-tiny",
    "collabora/whisper-tiny-hindi",
)

for model_id in MODELS:
    print(f"Baking {model_id} ...", flush=True)
    pipeline("automatic-speech-recognition", model=model_id, device="cpu")

print("Whisper cache baked.")
