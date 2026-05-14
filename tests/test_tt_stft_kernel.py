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


def test_filter_shapes():
    """Precomputed DFT matrices must have correct shapes for matmul."""
    kernel = TTStftKernel(sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    # cos/sin: (n_fft, n_freqs) for frames @ cos_basis
    assert kernel._cos_basis.shape == (2048, 1025), f"_cos_basis shape: {kernel._cos_basis.shape}"
    assert kernel._sin_basis.shape == (2048, 1025), f"_sin_basis shape: {kernel._sin_basis.shape}"
    # mel_filter: (n_freqs, n_mels) for mag @ mel_filter
    assert kernel._mel_filter.shape == (1025, 128), f"_mel_filter shape: {kernel._mel_filter.shape}"
    assert kernel._hann.shape == (2048,), f"_hann shape: {kernel._hann.shape}"
    # All float32
    assert kernel._cos_basis.dtype == np.float32
    assert kernel._sin_basis.dtype == np.float32
    assert kernel._mel_filter.dtype == np.float32
    assert kernel._hann.dtype == np.float32


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


def test_process_file_returns_full_spectrogram():
    """process_file produces SpectrogramChunk covering full audio duration."""
    sr = 22050
    kernel = TTStftKernel(sr=sr)
    duration = 5.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    chunk = kernel.process_file(audio, sr=sr)

    assert isinstance(chunk, SpectrogramChunk)
    # Should have frames, mel, and timestamps
    assert chunk.mag.ndim == 2
    assert chunk.mag.shape[1] == kernel.n_freqs
    assert chunk.mel.shape == (chunk.mag.shape[0], kernel.n_mels)
    assert chunk.timestamps.shape == (chunk.mag.shape[0],)
    # Timestamps should span the audio duration
    assert chunk.timestamps[0] >= 0.0
    assert chunk.timestamps[-1] < duration + 0.5


def test_process_file_streaming_continuity():
    """
    Streaming: stitched result must match single-chunk result within 1% at boundaries.
    Uses a 5-second file processed with chunk_seconds=3.0, overlap_seconds=1.0
    so that stitching happens at approximately t=3.0s.
    """
    sr = 22050
    hop = 512
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Small chunks to force stitching
    kernel_stream = TTStftKernel(sr=sr, chunk_seconds=3.0, overlap_seconds=1.0,
                                  hop_length=hop)
    result_stream = kernel_stream.process_file(audio, sr=sr)

    # Single-pass reference: process all at once as one chunk
    kernel_single = TTStftKernel(sr=sr, chunk_seconds=100.0, overlap_seconds=1.0,
                                  hop_length=hop)
    result_single = kernel_single.process_file(audio, sr=sr)

    # Both should have the same number of frames (within 2)
    assert abs(result_stream.mag.shape[0] - result_single.mag.shape[0]) <= 2, (
        f"Frame count divergence: stream={result_stream.mag.shape[0]}, "
        f"single={result_single.mag.shape[0]}"
    )

    # Timestamps must be strictly monotonically increasing
    assert np.all(np.diff(result_stream.timestamps) > 0), (
        "Streaming timestamps are not monotonically increasing at stitch boundary"
    )

    # Find the stitch point: first frame at or after t=3.0s (chunk boundary)
    stitch_idx = int(np.searchsorted(result_stream.timestamps, 3.0))

    # Compare frames just AFTER the stitch (5 frames on each side of boundary)
    n_stream = result_stream.mag.shape[0]
    n_single = result_single.mag.shape[0]

    for offset in range(-2, 5):
        si = stitch_idx + offset
        # Find matching frame in single-pass by timestamp
        if si < 0 or si >= n_stream:
            continue
        ts = result_stream.timestamps[si]
        # Find closest timestamp in single-pass result
        di = int(np.argmin(np.abs(result_single.timestamps - ts)))
        if di >= n_single:
            continue

        # Relative error in magnitude at this frame
        stream_frame = result_stream.mag[si]
        single_frame = result_single.mag[di]
        mean_mag = np.abs(single_frame).mean()
        if mean_mag < 1e-9:
            continue
        rel_err = np.abs(stream_frame - single_frame).mean() / mean_mag
        assert rel_err < 0.01, (
            f"Stitch boundary error at t={ts:.3f}s (frame {si}, offset {offset}): "
            f"rel_err={rel_err:.4f} (max 0.01)"
        )


def test_process_file_long_audio_no_oom():
    """Long audio (60 seconds) must not crash or OOM."""
    sr = 22050
    # Allocate 60s of audio
    audio = np.zeros(sr * 60, dtype=np.float32)
    audio[:1000] = 0.1  # some non-zero signal
    kernel = TTStftKernel(sr=sr, chunk_seconds=30.0, overlap_seconds=2.0)
    result = kernel.process_file(audio)  # should not raise
    assert result.mag.shape[0] > 0
