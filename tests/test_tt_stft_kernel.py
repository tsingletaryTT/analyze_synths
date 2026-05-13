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


def test_process_chunk_numpy_returns_spectrogram_chunk():
    """NumPy chunk processing returns SpectrogramChunk with correct shapes."""
    sr = 22050
    n_fft = 2048
    hop = 512
    n_mels = 128
    kernel = TTStftKernel(sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)

    # 2 seconds of audio = 44100 samples
    audio = np.random.randn(44100).astype(np.float32) * 0.1
    chunk = kernel.process_chunk(audio, chunk_start_time=0.0)

    assert isinstance(chunk, SpectrogramChunk)
    n_frames = (len(audio) - n_fft) // hop + 1
    assert chunk.mag.shape == (n_frames, kernel.n_freqs)
    assert chunk.mel.shape == (n_frames, n_mels)
    assert chunk.timestamps.shape == (n_frames,)
    assert chunk.mag.dtype == np.float32
    assert chunk.mel.dtype == np.float32
    assert chunk.timestamps.dtype == np.float32


def test_process_chunk_parity_with_librosa():
    """NumPy DFT-matmul must produce mel spectrogram peak bin within 5% of librosa."""
    import librosa
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    kernel = TTStftKernel(sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    chunk = kernel.process_chunk(audio, chunk_start_time=0.0)

    # Our mel spectrogram peak frame
    our_mel_mean = chunk.mel.mean(axis=0)   # (n_mels,)
    our_peak_bin = int(np.argmax(our_mel_mean))

    # librosa reference
    ref_mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )  # (n_mels, n_frames)
    ref_mel_mean = ref_mel.mean(axis=1)    # (n_mels,)
    ref_peak_bin = int(np.argmax(ref_mel_mean))

    # Allow 5% of n_mels = 6 bins tolerance
    tolerance = max(1, int(0.05 * 128))
    assert abs(our_peak_bin - ref_peak_bin) <= tolerance, (
        f"mel peak bin: ours={our_peak_bin}, librosa={ref_peak_bin}, "
        f"tolerance={tolerance}"
    )


def test_timestamps_are_correct():
    """Frame timestamps must reflect chunk_start_time offset."""
    sr = 22050
    hop = 512
    kernel = TTStftKernel(sr=sr, hop_length=hop)
    audio = np.random.randn(22050).astype(np.float32)  # 1 second

    # Process starting at t=10.0s
    chunk = kernel.process_chunk(audio, chunk_start_time=10.0)

    # First frame center time: 10.0 + n_fft/2/sr
    expected_first = 10.0 + (2048 / 2) / sr
    assert abs(chunk.timestamps[0] - expected_first) < 0.01, (
        f"First timestamp: {chunk.timestamps[0]:.4f}, expected ~{expected_first:.4f}"
    )
    # Timestamp spacing should be hop/sr
    if len(chunk.timestamps) > 1:
        spacing = chunk.timestamps[1] - chunk.timestamps[0]
        expected_spacing = hop / sr
        assert abs(spacing - expected_spacing) < 0.001
