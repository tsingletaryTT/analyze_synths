import pytest
import numpy as np
from pathlib import Path


def _make_sine_wave(freq_hz: float = 440.0, duration_s: float = 3.0,
                    sr: int = 22050) -> np.ndarray:
    """Returns a simple sine wave as float32 numpy array."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_jax_extractor_importable():
    """JaxAudioFeatureExtractor can be imported without error."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor
    extractor = JaxAudioFeatureExtractor(sr=22050)
    assert extractor is not None
