# tests/test_trajectory_analysis.py
import numpy as np
import pytest

from audio_analysis.core.tt_stft_kernel import SpectrogramChunk
from audio_analysis.core.trajectory_analysis import TrajectoryAnalyzer
from audio_analysis.core.narrative_types import TrajectoryPoint

SR = 22050
HOP = 512


def _make_chunk(n_frames: int, sr: int = SR, hop: int = HOP,
                energy_ramp: bool = False):
    """Return (SpectrogramChunk, audio_array) with known properties."""
    n_freqs = 1025
    n_mels = 128
    n_samples = n_frames * hop + 2048

    mag = np.random.rand(n_frames, n_freqs).astype(np.float32) * 0.1
    if energy_ramp:
        ramp = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
        mag = mag + ramp[:, None]
    mel = np.random.rand(n_frames, n_mels).astype(np.float32)
    ts = np.arange(n_frames, dtype=np.float32) * hop / sr

    audio = np.random.randn(n_samples).astype(np.float32) * 0.01
    chunk = SpectrogramChunk(mag=mag, mel=mel, timestamps=ts)
    return chunk, audio


def test_trajectory_has_correct_length():
    n_frames = 300   # ~7 seconds at sr=22050, hop=512
    chunk, audio = _make_chunk(n_frames)
    az = TrajectoryAnalyzer(sr=SR, hop_length=HOP)
    points = az.analyze(chunk, audio)
    # window_frames = ceil(2 * 22050 / 512) = 87
    # expect roughly n_frames // 87 points
    assert len(points) >= 1
    assert all(isinstance(p, TrajectoryPoint) for p in points)


def test_trajectory_energy_rises_with_ramp():
    """On a file where magnitude linearly increases, energy must be monotonically rising."""
    n_frames = 600   # ~14 seconds
    chunk, audio = _make_chunk(n_frames, energy_ramp=True)
    az = TrajectoryAnalyzer(sr=SR, hop_length=HOP)
    points = az.analyze(chunk, audio)
    if len(points) < 2:
        pytest.skip("too few points")
    energies = [p.energy for p in points]
    first_half = np.mean(energies[: len(energies) // 2])
    second_half = np.mean(energies[len(energies) // 2 :])
    assert second_half > first_half, f"Energy should rise: {first_half:.4f} -> {second_half:.4f}"


def test_tension_scores_in_range():
    n_frames = 300
    chunk, audio = _make_chunk(n_frames)
    az = TrajectoryAnalyzer(sr=SR, hop_length=HOP)
    points = az.analyze(chunk, audio)
    for p in points:
        assert 0.0 <= p.tension_score <= 1.0, f"tension_score={p.tension_score} out of [0,1]"


def test_high_roughness_raises_tension():
    """A high-roughness noise segment should have mean tension > 0.4."""
    n_frames = 300
    n_freqs = 1025
    n_samples = n_frames * HOP + 2048
    # White noise in frequency domain — very rough spectrum
    mag = np.random.rand(n_frames, n_freqs).astype(np.float32)
    mel = np.random.rand(n_frames, 128).astype(np.float32)
    ts = np.arange(n_frames, dtype=np.float32) * HOP / SR
    chunk = SpectrogramChunk(mag=mag, mel=mel, timestamps=ts)
    audio = np.random.randn(n_samples).astype(np.float32)  # white noise for high ZCR

    az = TrajectoryAnalyzer(sr=SR, hop_length=HOP)
    points = az.analyze(chunk, audio)
    mean_tension = np.mean([p.tension_score for p in points])
    assert mean_tension > 0.4, f"High roughness should raise tension, got {mean_tension:.3f}"


def test_chroma_peak_is_valid_note():
    n_frames = 200
    chunk, audio = _make_chunk(n_frames)
    az = TrajectoryAnalyzer(sr=SR, hop_length=HOP)
    points = az.analyze(chunk, audio)
    valid = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    for p in points:
        assert p.chroma_peak in valid, f"chroma_peak={p.chroma_peak!r} not a note name"
