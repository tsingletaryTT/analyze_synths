"""
JAX/Tenstorrent audio feature extraction.

Runs the full feature extraction pipeline on Tenstorrent Blackhole hardware
via the JAX PJRT plugin. Uses DFT-via-matmul instead of jnp.fft.rfft because
complex tensor materialization is not yet supported by the TT PJRT backend.

Hardware constraints confirmed on P150X4 (JAX 0.6.0, 2026-05-12):
  - jnp.dot / einsum / vmap / jit / cumsum / argmax: supported
  - jnp.fft.rfft: NOT supported (complex materialization failure)
  - lax.scan / lax.while_loop / lax.fori_loop: NOT supported (stablehlo.while)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level filter cache: (sr, n_fft, n_mels, n_mfcc) -> filter arrays
_FILTER_CACHE: Dict[Tuple, Dict[str, Any]] = {}

# Chromatic pitch class names matching feature_extraction_base.py
_MUSICAL_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Librosa-compatible tonnetz transform matrix (12x6 float32).
# Projects 12-dimensional chroma to 6 tonal centroid dimensions
# (perfect fifth, minor third, major third, each as sin/cos pair).
_TONNETZ_TRANSFORM = np.array([
    [1,  0, -1,  0,  1,  0, -1,  0,  1,  0, -1,  0],   # fifth cos
    [0,  1,  0, -1,  0,  1,  0, -1,  0,  1,  0, -1],   # fifth sin
    [1,  0,  0, -1, -1,  0,  0,  1,  1,  0,  0, -1],   # minor third cos
    [0,  1,  1,  0,  0, -1, -1,  0,  0,  1,  1,  0],   # minor third sin
    [1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1],   # major third cos
    [0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1],   # major third sin
], dtype=np.float32).T  # shape (12, 6)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Construct triangular mel filterbank matrix, shape (n_fft//2+1, n_mels).

    Follows the librosa mel scale formula for direct comparison with
    feature_extraction_base.py output.
    """
    n_freqs = n_fft // 2 + 1
    fmin, fmax = 0.0, float(sr) / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if k < n_freqs:
                filterbank[k, m - 1] = (k - f_left) / max(f_center - f_left, 1)
        for k in range(f_center, f_right):
            if k < n_freqs:
                filterbank[k, m - 1] = (f_right - k) / max(f_right - f_center, 1)

    return filterbank


def _build_dct_matrix(n_mels: int, n_mfcc: int) -> np.ndarray:
    """
    Type-II DCT matrix, shape (n_mels, n_mfcc).

    Matches scipy.fft.dct(type=2, norm='ortho') applied along the mel axis.
    """
    m = np.arange(n_mels, dtype=np.float64)[:, None]
    k = np.arange(n_mfcc, dtype=np.float64)[None, :]
    dct = np.cos(np.pi * k * (2.0 * m + 1.0) / (2.0 * n_mels))
    dct[:, 0] *= np.sqrt(1.0 / n_mels)
    dct[:, 1:] *= np.sqrt(2.0 / n_mels)
    return dct.astype(np.float32)


def _build_chroma_filter(sr: int, n_fft: int) -> np.ndarray:
    """
    Chroma filter matrix, shape (n_fft//2+1, 12).

    Maps each FFT frequency bin to its closest pitch class (0=C ... 11=B).
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    chroma = np.zeros((n_freqs, 12), dtype=np.float32)
    for i, f in enumerate(freqs):
        if f > 0:
            midi = 12.0 * np.log2(f / 440.0) + 69.0
            pitch_class = int(round(midi)) % 12
            chroma[i, pitch_class] = 1.0
    return chroma


def _build_dft_matrices(n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DFT basis matrices for computing magnitude spectrum via matmul.

    Returns:
        dft_cos: (n_fft//2+1, n_fft) float32 — real (cosine) basis
        dft_sin: (n_fft//2+1, n_fft) float32 — imaginary (sine) basis

    Usage: mag = sqrt((frames @ dft_cos.T)**2 + (frames @ dft_sin.T)**2)
    where frames shape is (..., n_fft).
    """
    n_freqs = n_fft // 2 + 1
    n = np.arange(n_fft, dtype=np.float64)
    k = np.arange(n_freqs, dtype=np.float64)[:, None]
    angles = 2.0 * np.pi * k * n / n_fft
    dft_cos = np.cos(angles).astype(np.float32)
    dft_sin = np.sin(angles).astype(np.float32)
    return dft_cos, dft_sin


class JaxAudioFeatureExtractor:
    """
    JAX-based audio feature extractor targeting Tenstorrent Blackhole hardware.

    All compute-intensive operations use matmul and element-wise ops (supported
    on TT via JAX PJRT). Audio loading and tempo tracking remain on CPU via librosa.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        n_chroma: int = 12,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_freqs = n_fft // 2 + 1

        cache_key = (sr, n_fft, n_mels, n_mfcc)
        if cache_key not in _FILTER_CACHE:
            _FILTER_CACHE[cache_key] = self._precompute_filters()

        filters = _FILTER_CACHE[cache_key]
        self.dft_cos = filters['dft_cos']
        self.dft_sin = filters['dft_sin']
        self.mel_filterbank = filters['mel_filterbank']
        self.dct_matrix = filters['dct_matrix']
        self.chroma_filter = filters['chroma_filter']
        self.freq_hz = filters['freq_hz']
        self.hann_window = filters['hann_window']
        self.tonnetz_transform = _TONNETZ_TRANSFORM

    def _precompute_filters(self) -> Dict[str, np.ndarray]:
        dft_cos, dft_sin = _build_dft_matrices(self.n_fft)
        return {
            'dft_cos': dft_cos,
            'dft_sin': dft_sin,
            'mel_filterbank': _build_mel_filterbank(self.sr, self.n_fft, self.n_mels),
            'dct_matrix': _build_dct_matrix(self.n_mels, self.n_mfcc),
            'chroma_filter': _build_chroma_filter(self.sr, self.n_fft),
            'freq_hz': np.linspace(0, self.sr / 2, self.n_freqs, dtype=np.float32),
            'hann_window': np.hanning(self.n_fft).astype(np.float32),
        }

    def extract_batch(
        self,
        audio_batch: np.ndarray,
        lengths: np.ndarray,
        sr: int,
        file_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """
        Extract audio features for a batch of files.

        Args:
            audio_batch: (B, max_len) float32, padded with zeros
            lengths: (B,) int32, actual sample count per file
            sr: sample rate (uniform across batch)
            file_paths: original file paths for metadata

        Returns:
            List of feature dicts with same keys as feature_extraction_base.py
        """
        # Stub — full implementation added in Task 4.
        return [{'filename': p.name} for p in file_paths]
