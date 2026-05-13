import numpy as np
import pytest
from audio_analysis.core.tt_stft_kernel import SpectrogramChunk, TTStftKernel


def test_spectrogram_chunk_fields():
    n_frames, n_freqs, n_mels = 100, 1025, 128
    chunk = SpectrogramChunk(
        mag=np.zeros((n_frames, n_freqs), dtype=np.float32),
        mel=np.zeros((n_frames, n_mels), dtype=np.float32),
        timestamps=np.linspace(0, 5.0, n_frames, dtype=np.float32),
    )
    assert chunk.mag.shape == (n_frames, n_freqs)
    assert chunk.mel.shape == (n_frames, n_mels)
    assert chunk.timestamps.shape == (n_frames,)


def test_tt_stft_kernel_instantiates():
    kernel = TTStftKernel(sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    assert kernel.sr == 22050
    assert kernel.n_fft == 2048
    assert kernel.hop_length == 512
    assert kernel.n_mels == 128
    assert kernel.n_freqs == 1025  # n_fft // 2 + 1
