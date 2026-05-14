"""
Hardware parity test: tt_stft_hw.fused_stft_hw vs _process_chunk_numpy.

Skipped if JAX TT backend is unavailable.
"""
import pytest
import numpy as np

try:
    from audio_analysis.core.tt_stft_hw import is_available, fused_stft_hw
    HW_AVAILABLE = is_available()
except Exception:
    HW_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HW_AVAILABLE, reason="TT hardware (JAX PJRT) not available"
)


@pytest.fixture
def kernel():
    from audio_analysis.core.tt_stft_kernel import TTStftKernel
    return TTStftKernel(sr=22050)


def test_hw_mag_shape(kernel):
    """Hardware dispatch returns correct mag/mel shapes."""
    audio = np.random.randn(22050 * 5).astype(np.float32)  # 5s
    chunk = fused_stft_hw(audio, kernel, chunk_start_time=0.0)
    n_frames = (len(audio) - kernel.n_fft) // kernel.hop_length + 1
    assert chunk.mag.shape == (n_frames, kernel.n_freqs)
    assert chunk.mel.shape == (n_frames, kernel.n_mels)
    assert chunk.timestamps.shape == (n_frames,)


def test_hw_parity_with_numpy(kernel):
    """Hardware mag output matches NumPy reference within 2%."""
    np.random.seed(42)
    audio = np.random.randn(22050 * 5).astype(np.float32)

    hw_chunk  = fused_stft_hw(audio, kernel, chunk_start_time=0.0)
    cpu_chunk = kernel._process_chunk_numpy(audio, chunk_start_time=0.0)

    # 2% tolerance: NumPy reference uses float64 intermediate arithmetic
    # (the 1e-8 literal upcasts sqrt operands to float64), while TT hardware
    # stays in float32. float32 matmul over K=2048 accumulates ~1-2% vs float64.
    ratio = np.abs(hw_chunk.mag - cpu_chunk.mag) / (cpu_chunk.mag + 1e-6)
    assert ratio.mean() < 0.02, f"Mean relative error {ratio.mean():.4f} > 2%"


def test_hw_timestamps_correct(kernel):
    """Timestamps are monotonically increasing and match expected values."""
    audio = np.ones(22050 * 3, dtype=np.float32)
    chunk = fused_stft_hw(audio, kernel, chunk_start_time=10.0)
    assert np.all(np.diff(chunk.timestamps) > 0), "timestamps not monotonic"
    expected_t0 = 10.0 + kernel.n_fft / 2 / kernel.sr
    assert abs(chunk.timestamps[0] - expected_t0) < 1e-3


def test_process_file_uses_hw_path(kernel):
    """TTStftKernel.process_file picks hardware path when available."""
    audio = np.random.randn(22050 * 10).astype(np.float32)
    assert kernel._try_hw, "Expected _try_hw=True with TT hardware available"
    chunk = kernel.process_file(audio, sr=22050)
    n_frames = (len(audio) - kernel.n_fft) // kernel.hop_length + 1
    # Within ±1 frame (overlap discard rounding)
    assert abs(chunk.mag.shape[0] - n_frames) <= 1
