"""Normalizes audio and caches its waveform + spectrogram tensors, keyed by
a hash of the source file's path so re-running preprocessing skips work
that's already cached.

No silence trimming: LJSpeech clips are already tightly trimmed by the
dataset's own creators (measured empirically -- VAD found ~0.000s of
leading silence and only chunk-rounding noise at the trailing edge across
a random sample), so a VAD pass here would cost a per-utterance ONNX
inference loop for no actual effect. If a future dataset isn't pre-trimmed,
silence trimming would need to be reintroduced.
"""
import hashlib
from pathlib import Path
from typing import Tuple, Union

import librosa
import torch

from ..mel_processing import spectrogram_torch

_FILTER_LENGTH = 1024
_WINDOW_LENGTH = 1024
_HOP_LENGTH = 256


def _cache_paths(audio_path: Path, cache_dir: Path) -> Tuple[Path, Path]:
    cache_id = hashlib.sha256(str(audio_path.absolute()).encode("utf-8")).hexdigest()
    return cache_dir / f"{cache_id}.pt", cache_dir / f"{cache_id}.spec.pt"


def cache_norm_audio(
    audio_path: Union[str, Path],
    cache_dir: Union[str, Path],
    sample_rate: int,
    ignore_cache: bool = False,
) -> Tuple[Path, Path]:
    audio_path = Path(audio_path)
    cache_dir = Path(cache_dir)
    wav_path, spec_path = _cache_paths(audio_path, cache_dir)

    wav_tensor = None
    if ignore_cache or not wav_path.exists():
        # librosa already normalizes to float32 in [-1, 1]. Uses the
        # FloatTensor constructor rather than torch.from_numpy(): the latter
        # crashes ("Numpy is not available") under a torch/numpy ABI
        # mismatch that FloatTensor's copy-based path tolerates.
        samples, _sr = librosa.load(audio_path, sr=sample_rate)
        wav_tensor = torch.FloatTensor(samples).unsqueeze(0)
        torch.save(wav_tensor, wav_path)

    if ignore_cache or not spec_path.exists():
        if wav_tensor is None:
            wav_tensor = torch.load(wav_path)

        spec_tensor = spectrogram_torch(
            wav_tensor, _FILTER_LENGTH, _HOP_LENGTH, _WINDOW_LENGTH
        ).squeeze(0)
        torch.save(spec_tensor, spec_path)

    return wav_path, spec_path
