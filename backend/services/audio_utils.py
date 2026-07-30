"""Audio utility functions for conversion."""

import subprocess

def convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert arbitrary audio bytes to standard 16kHz mono WAV format using ffmpeg."""
    try:
        process = subprocess.Popen(
            ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=audio_bytes)
        if process.returncode != 0:
            print(f"ffmpeg error: {stderr.decode('utf-8', errors='ignore')}")
            return audio_bytes
        return stdout
    except Exception as e:
        print(f"Failed to convert audio via ffmpeg: {e}")
        return audio_bytes
