"""
Tests for TT-Lang simulator STFT kernel.
These tests use the TT-Lang simulator (no hardware required).

Run with:
    PYTHONPATH=/home/ttuser/code/tt-lang/python pytest tests/test_tt_stft_sim.py -v
"""
import sys
import numpy as np
import pytest

# The simulator requires this path
TTL_PYTHON = "/home/ttuser/code/tt-lang/python"
if TTL_PYTHON not in sys.path:
    sys.path.insert(0, TTL_PYTHON)

# Skip entire module if simulator is not available
try:
    from sim import ttl, ttnn  # noqa: F401
    HAS_SIMULATOR = True
except ImportError:
    HAS_SIMULATOR = False

pytestmark = pytest.mark.skipif(
    not HAS_SIMULATOR,
    reason="TT-Lang simulator not available at /home/ttuser/code/tt-lang/python",
)


def test_simulator_kernel_imports():
    """tt_stft_sim module must import cleanly and expose fused_stft_sim."""
    from audio_analysis.core import tt_stft_sim  # noqa: F401
    assert hasattr(tt_stft_sim, "fused_stft_sim"), (
        "tt_stft_sim must export fused_stft_sim()"
    )


def test_simulator_kernel_returns_spectrogram_chunk():
    """fused_stft_sim must return a SpectrogramChunk with correct shapes."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel, SpectrogramChunk
    from audio_analysis.core.tt_stft_sim import fused_stft_sim

    kernel = TTStftKernel(sr=sr)
    chunk = fused_stft_sim(audio, kernel)

    assert isinstance(chunk, SpectrogramChunk), "Return type must be SpectrogramChunk"
    assert chunk.mag.ndim == 2, f"mag must be 2D, got shape {chunk.mag.shape}"
    assert chunk.mel.ndim == 2, f"mel must be 2D, got shape {chunk.mel.shape}"
    assert chunk.timestamps.ndim == 1, "timestamps must be 1D"
    assert chunk.mag.shape[1] == kernel.n_freqs, (
        f"mag columns must be n_freqs={kernel.n_freqs}, got {chunk.mag.shape[1]}"
    )
    assert chunk.mel.shape[1] == kernel.n_mels, (
        f"mel columns must be n_mels={kernel.n_mels}, got {chunk.mel.shape[1]}"
    )
    assert chunk.mag.shape[0] == chunk.mel.shape[0] == chunk.timestamps.shape[0], (
        "mag, mel, and timestamps must have the same number of frames"
    )


def test_simulator_kernel_parity_with_numpy():
    """
    fused_stft_sim output must match _process_chunk_numpy within 5%.

    Test signal: 440 Hz sine wave (1 second at 22050 Hz).
    The dominant mel bin in the mean spectrum must agree within 6 bins
    (5% of 128 mel bins).
    """
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel
    from audio_analysis.core.tt_stft_sim import fused_stft_sim

    kernel = TTStftKernel(sr=sr)

    # NumPy reference path
    ref = kernel.process_chunk(audio, chunk_start_time=0.0)

    # TT-Lang simulator path
    sim_chunk = fused_stft_sim(audio, kernel, chunk_start_time=0.0)

    # Compare mel spectrograms via mean energy per bin across frames
    n_frames = min(ref.mel.shape[0], sim_chunk.mel.shape[0])
    ref_mean = ref.mel[:n_frames].mean(axis=0)
    sim_mean = sim_chunk.mel[:n_frames].mean(axis=0)

    ref_peak = int(np.argmax(ref_mean))
    sim_peak = int(np.argmax(sim_mean))

    tolerance_bins = max(1, int(0.05 * 128))  # 6 bins = 5% of 128
    assert abs(sim_peak - ref_peak) <= tolerance_bins, (
        f"Simulator mel peak bin {sim_peak} diverges from NumPy {ref_peak} "
        f"by more than {tolerance_bins} bins.\n"
        f"  ref top-5 bins: {np.argsort(ref_mean)[-5:][::-1].tolist()}\n"
        f"  sim top-5 bins: {np.argsort(sim_mean)[-5:][::-1].tolist()}"
    )


def test_simulator_kernel_empty_audio():
    """fused_stft_sim must return an empty SpectrogramChunk for very short audio."""
    sr = 22050
    # Audio shorter than one STFT frame — no valid frames
    audio = np.zeros(100, dtype=np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel
    from audio_analysis.core.tt_stft_sim import fused_stft_sim

    kernel = TTStftKernel(sr=sr)
    chunk = fused_stft_sim(audio, kernel)

    assert chunk.mag.shape[0] == 0, "Short audio must produce 0 frames"
    assert chunk.mel.shape[0] == 0, "Short audio must produce 0 frames"
    assert chunk.timestamps.shape[0] == 0, "Short audio must produce 0 timestamps"


def test_simulator_kernel_dtype():
    """fused_stft_sim must return float32 arrays (Tensix tile dtype)."""
    sr = 22050
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    audio = (0.1 * np.cos(2 * np.pi * 1000 * t)).astype(np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel
    from audio_analysis.core.tt_stft_sim import fused_stft_sim

    kernel = TTStftKernel(sr=sr)
    chunk = fused_stft_sim(audio, kernel)

    if chunk.mag.shape[0] > 0:
        assert chunk.mag.dtype == np.float32, f"mag dtype must be float32, got {chunk.mag.dtype}"
        assert chunk.mel.dtype == np.float32, f"mel dtype must be float32, got {chunk.mel.dtype}"
        assert chunk.timestamps.dtype == np.float32, (
            f"timestamps dtype must be float32, got {chunk.timestamps.dtype}"
        )
