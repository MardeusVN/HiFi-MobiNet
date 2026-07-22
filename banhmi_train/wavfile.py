"""Writes mono 16-bit PCM WAV files using the standard library's `wave`
module -- no third-party WAV codec needed for this one job.
"""
import wave
from pathlib import Path
from typing import Union

import numpy as np


def write(path: Union[str, Path], sample_rate: int, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.astype(np.int16).tobytes())
