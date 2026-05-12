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


def test_filter_shapes():
    """Precomputed filter matrices have correct shapes."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor
    ex = JaxAudioFeatureExtractor(sr=22050, n_fft=2048, n_mels=128, n_mfcc=13)
    n_freqs = 2048 // 2 + 1  # 1025
    assert ex.dft_cos.shape == (n_freqs, 2048), f"dft_cos shape wrong: {ex.dft_cos.shape}"
    assert ex.dft_sin.shape == (n_freqs, 2048), f"dft_sin shape wrong: {ex.dft_sin.shape}"
    assert ex.mel_filterbank.shape == (n_freqs, 128), f"mel shape wrong: {ex.mel_filterbank.shape}"
    assert ex.dct_matrix.shape == (128, 13), f"dct shape wrong: {ex.dct_matrix.shape}"
    assert ex.chroma_filter.shape == (n_freqs, 12), f"chroma shape wrong: {ex.chroma_filter.shape}"
    assert ex.tonnetz_transform.shape == (12, 6), f"tonnetz shape wrong: {ex.tonnetz_transform.shape}"
    assert ex.freq_hz.shape == (n_freqs,), f"freq_hz shape wrong: {ex.freq_hz.shape}"


def test_device_detection_returns_string():
    """_detect_tt_device returns 'tenstorrent' or 'cpu'."""
    from audio_analysis import _detect_tt_device
    result = _detect_tt_device()
    assert result in ('tenstorrent', 'cpu'), f"Unexpected device: {result}"


def test_processing_config_has_device():
    """ProcessingConfig exposes a device field."""
    from audio_analysis.core.parallel_feature_extraction import ProcessingConfig
    config = ProcessingConfig()
    assert hasattr(config, 'device')
    assert config.device in ('tenstorrent', 'cpu')


def test_processing_config_tt_forces_sample_rate():
    """When device='tenstorrent', sample_rate is set to 22050."""
    from audio_analysis.core.parallel_feature_extraction import ProcessingConfig
    config = ProcessingConfig(device='tenstorrent')
    assert config.sample_rate == 22050
