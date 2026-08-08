"""Audio recording and playback using sounddevice. Pure logic, no Qt imports."""

import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioHandler:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames = []
        self._stream = None
        self._playback_stream = None

    def start_recording(self):
        """Start capturing audio from the default microphone."""
        if self._recording:
            return
        self._frames = []
        self._recording = True

        def callback(indata, frame_count, time_info, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio as a numpy array.

        Leading/trailing silence is trimmed so the audio starts at the
        actual speech onset.
        """
        if not self._recording:
            return np.array([], dtype=np.float32)
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._frames:
            audio = np.concatenate(self._frames, axis=0).flatten()
            return self.trim_audio(audio)
        return np.array([], dtype=np.float32)

    def trim_audio(
        self,
        audio: np.ndarray,
        top_db: int = 25,
        min_duration: float = 0.1,
        margin_ms: float = 50.0,
    ) -> np.ndarray:
        """Trim leading and trailing silence from recorded audio.

        Only audio from the actual start of speech to its end is kept,
        with a small margin around the speech boundaries. If the capture
        contains no real speech (nothing above the silence threshold for
        at least ``min_duration`` seconds), the original audio is returned
        unchanged so the recording is not destroyed.
        """
        if audio is None or len(audio) == 0:
            return audio
        try:
            import librosa

            frame_length = 512
            hop_length = 256
            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]
            max_rms = float(np.max(rms))
            if max_rms <= 0:
                return audio
            threshold = max_rms * (10.0 ** (-top_db / 20.0))
            voiced = np.flatnonzero(rms >= threshold)
            if voiced.size == 0:
                return audio

            margin = int(margin_ms * self.sample_rate / 1000)
            start = max(0, int(voiced[0] * hop_length) - margin)
            end = min(len(audio), int((voiced[-1] + 1) * hop_length) + margin)
            trimmed = audio[start:end]
        except Exception:
            return audio
        if len(trimmed) < int(min_duration * self.sample_rate):
            return audio
        return trimmed

    def play_audio(self, audio: np.ndarray):
        """Play audio through speakers in a background thread."""
        self.stop_playback()

        def _play():
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def stop_playback(self):
        """Stop any ongoing playback."""
        try:
            sd.stop()
        except Exception:
            pass

    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio array to a .wav file."""
        sf.write(path, audio, self.sample_rate)

    def load_audio(self, path: str) -> np.ndarray:
        """Load a .wav file and return audio as numpy array at 16kHz."""
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if samplerate != self.sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=samplerate, target_sr=self.sample_rate)
        return data
