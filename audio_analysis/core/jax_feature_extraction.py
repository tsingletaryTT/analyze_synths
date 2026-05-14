"""
JAX/Tenstorrent audio feature extraction.

Runs the full feature extraction pipeline on Tenstorrent Blackhole hardware
via the JAX PJRT plugin. Uses DFT-via-matmul instead of jnp.fft.rfft because
complex tensor materialization is not yet supported by the TT PJRT backend.

Hardware constraints confirmed on P300C x4 Blackhole (JAX 0.7.1, 2026-05-12):
  - jnp.dot / einsum / vmap / jit / cumsum: supported
  - jnp.argmax(x, axis=-1): supported ONLY on the LAST axis
  - jnp.argmin(x, axis=n): NOT supported (stablehlo.reduce failure)
    Workaround: use jnp.argmax(-x.T, axis=-1) for argmin over axis=0
  - jnp.fft.rfft: NOT supported (complex tensor materialization failure)
  - lax.scan / lax.while_loop / lax.fori_loop: NOT supported (stablehlo.while)
  - Float32 DFT: 2048-wide matmul accumulates ~10x more off-peak energy
    than CPU float64 reference; spectral centroid biased ~8-10%, peak bin
    location is correct
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Activate PJRT plugin before importing JAX so TT devices are visible.
# Only sets the env var if it's not already set by the environment.
_plugin_path = os.path.expanduser('~/tt-xla/build/lib/libpjrt_tt.so')
if os.path.exists(_plugin_path) and 'PJRT_PLUGIN_LIBRARY_PATH' not in os.environ:
    os.environ['PJRT_PLUGIN_LIBRARY_PATH'] = _plugin_path

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _JAX_AVAILABLE:
    logger.warning("JAX not available — JaxAudioFeatureExtractor will raise on use")

# Module-level filter cache: (sr, n_fft, n_mels, n_mfcc) -> filter arrays
_FILTER_CACHE: Dict[Tuple, Dict[str, Any]] = {}

# Chromatic pitch class names matching feature_extraction_base.py
_MUSICAL_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tonnetz transform matrix (12x6 float32).
# Projects 12-dimensional chroma to 6 tonal centroid dimensions
# (perfect fifth, minor third, major third, each as sin/cos pair).
# Normalised by 1/sqrt(6) to match librosa's tonnetz implementation — without
# this factor the exported values are ~2.45x larger than librosa's output.
_TONNETZ_TRANSFORM = np.array([
    [1,  0, -1,  0,  1,  0, -1,  0,  1,  0, -1,  0],   # fifth cos
    [0,  1,  0, -1,  0,  1,  0, -1,  0,  1,  0, -1],   # fifth sin
    [1,  0,  0, -1, -1,  0,  0,  1,  1,  0,  0, -1],   # minor third cos
    [0,  1,  1,  0,  0, -1, -1,  0,  0,  1,  1,  0],   # minor third sin
    [1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1],   # major third cos
    [0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1],   # major third sin
]) / np.sqrt(6)
_TONNETZ_TRANSFORM = _TONNETZ_TRANSFORM.T.astype(np.float32)  # shape (12, 6)


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
    ):
        # Note: 12-bin chroma (one bin per Western pitch class C–B) is the only
        # supported configuration and is hardcoded throughout the pipeline.
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
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

    def _extract_tempo_cpu(
        self,
        audio_batch: np.ndarray,
        lengths: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Extract tempo and onset density on CPU via librosa.

        lax.scan is not yet supported by the TT PJRT backend (stablehlo.while),
        so this runs in the host process alongside the JAX batch dispatch.
        """
        import librosa  # noqa: PLC0415

        results = []
        for i in range(audio_batch.shape[0]):
            y = audio_batch[i, :lengths[i]]
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
                tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
                onset_density = float(len(beats) / (len(y) / self.sr)) if len(y) > 0 else 0.0
            except Exception as exc:
                logger.debug("beat_track failed for %s: %s", file_paths[i], exc)
                tempo_val = 0.0
                onset_density = 0.0
            results.append({
                'tempo': tempo_val,
                'onset_density': onset_density,
                'beat_count': 0,  # placeholder: lax.scan DP pending TT PJRT support
            })
        return results

    def _extract_from_spectrogram(
        self,
        mag: np.ndarray,
        mel: np.ndarray,
        audio: np.ndarray,
        sr: int,
        file_path,
    ) -> Dict[str, Any]:
        """
        Compute all features from a pre-computed magnitude spectrogram and mel spectrogram.

        All frames are treated as valid (TTStftKernel trims to actual signal length).
        Uses JAX for matmul-heavy ops (MFCC, chroma, tonnetz, spectral stats),
        NumPy for ZCR and RMS (frame-level ops on raw audio).

        Args:
            mag:       (n_frames, n_freqs) float32 — magnitude spectrogram from TTStftKernel
            mel:       (n_frames, n_mels)  float32 — mel spectrogram from TTStftKernel
            audio:     (n_samples,)        float32 — raw audio for ZCR/RMS computation
            sr:        sample rate in Hz
            file_path: Path-like — used for the 'filename' metadata field

        Returns:
            Dict mapping feature name -> Python scalar (float/int/str).
        """
        import jax.numpy as jnp  # noqa: PLC0415 — inner import avoids top-level JAX init

        n_frames, n_freqs = mag.shape
        n_samples = len(audio)

        # Move arrays to JAX once; all heavy reductions run on the JAX backend
        # (TT hardware or CPU, depending on PJRT plugin availability).
        jmag     = jnp.array(mag)                            # (n_frames, n_freqs)
        jmel     = jnp.array(mel)                            # (n_frames, n_mels)
        jdct     = jnp.array(self.dct_matrix)                # (n_mels, n_mfcc)
        jchroma  = jnp.array(self.chroma_filter)             # (n_freqs, 12)
        jtonnetz = jnp.array(self.tonnetz_transform)         # (12, 6)
        jfreq_hz = jnp.array(self.freq_hz)                   # (n_freqs,)

        # --- Spectral features ---
        # Normalise each frame's magnitude so spectral centroid is frequency-weighted
        # mean rather than a sum — makes the metric scale-independent.
        mag_sum  = jnp.sum(jmag, axis=-1, keepdims=True) + 1e-8
        mag_norm = jmag / mag_sum                             # (n_frames, n_freqs)

        centroid = jnp.sum(mag_norm * jfreq_hz, axis=-1)     # (n_frames,)
        centroid_mean = float(jnp.mean(centroid))
        centroid_std  = float(jnp.std(centroid) + 1e-8)

        # 85th-percentile cumulative energy rolloff frequency
        cumsum = jnp.cumsum(jmag, axis=-1)
        total  = cumsum[:, -1:] + 1e-8
        rolloff_mask = (cumsum >= 0.85 * total).astype(jnp.float32)
        rolloff_bin  = jnp.argmax(rolloff_mask, axis=-1).astype(jnp.float32)
        rolloff_hz   = rolloff_bin * (sr / 2.0) / (self.n_freqs - 1)
        rolloff_mean = float(jnp.mean(rolloff_hz))

        # Spectral bandwidth: weighted standard deviation of frequency around centroid
        dev = (jfreq_hz - centroid[:, None]) ** 2
        bandwidth = jnp.sqrt(jnp.sum(mag_norm * dev, axis=-1) + 1e-8)
        bandwidth_mean = float(jnp.mean(bandwidth))

        # --- MFCC ---
        # log-mel → DCT-II → cepstral coefficients
        # Using the pre-computed mel spectrogram from TTStftKernel for consistency.
        log_mel   = jnp.log(jmel + 1e-6)
        mfcc      = jnp.dot(log_mel, jdct)                   # (n_frames, n_mfcc)
        mfcc_mean = np.array(jnp.mean(mfcc, axis=0))         # (n_mfcc,)
        mfcc_std  = np.array(jnp.std(mfcc, axis=0) + 1e-8)  # (n_mfcc,)

        # --- Chroma + key + tonnetz ---
        # Project magnitude spectrum onto 12 pitch-class bins via chroma_filter.
        # Normalise per-frame so that louder frames don't dominate the aggregate.
        chroma      = jnp.dot(jmag, jchroma)                 # (n_frames, 12)
        chroma_norm = chroma / (jnp.sum(chroma, axis=-1, keepdims=True) + 1e-6)
        chroma_mean = np.array(jnp.mean(chroma_norm, axis=0))  # (12,)
        key_index      = int(np.argmax(chroma_mean))
        key_confidence = float(chroma_mean[key_index])
        # Tonnetz: 6-dimensional tonal centroid (3 interval classes × sin/cos)
        tonnetz = np.array(jnp.dot(jnp.array(chroma_mean), jtonnetz))  # (6,)

        # --- Spectral roughness ---
        # Mean absolute difference between adjacent frequency bins.
        # High values → noisy / rough timbres; low values → smooth / tonal spectra.
        spec_diff = jnp.abs(jnp.diff(jmag, axis=-1))         # (n_frames, n_freqs-1)
        spectral_roughness = float(jnp.mean(spec_diff))

        # --- ZCR + RMS from raw audio (NumPy, no JAX dependency) ---
        # Frame the raw audio with the same n_fft / hop_length as the STFT so that
        # ZCR and RMS are computed over the same temporal windows as the spectrogram.
        # Pad to at least n_fft samples so all frames are well-defined.
        if n_samples < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - n_samples))
        n_fft_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        # Align with spectrogram frame count (off-by-one at boundaries is expected)
        n_fft_frames = min(n_fft_frames, n_frames)
        frame_starts = np.arange(n_fft_frames) * self.hop_length
        frame_idx = frame_starts[:, None] + np.arange(self.n_fft)[None, :]
        frames_np = audio[frame_idx]                          # (n_fft_frames, n_fft)

        signs    = np.sign(frames_np)
        zcr_per  = np.mean(np.abs(np.diff(signs, axis=-1)), axis=-1) / 2.0
        zcr_mean = float(np.mean(zcr_per))

        rms_per  = np.sqrt(np.mean(frames_np ** 2, axis=-1))
        rms_mean = float(np.mean(rms_per))
        rms_std  = float(np.std(rms_per) + 1e-8)

        # --- Assemble feature dict ---
        f: Dict[str, Any] = {
            'filename':                file_path.name if hasattr(file_path, 'name') else str(file_path),
            # Duration derived from the original audio length before any padding.
            'duration':                float(n_samples) / float(sr),
            'spectral_centroid_mean':  centroid_mean,
            'spectral_centroid_std':   centroid_std,
            'spectral_rolloff_mean':   rolloff_mean,
            'spectral_bandwidth_mean': bandwidth_mean,
            'zero_crossing_rate_mean': zcr_mean,
            'rms_mean':                rms_mean,
            'rms_std':                 rms_std,
            # 'key' matches field name in feature_extraction_base.py
            'key':                     _MUSICAL_KEYS[key_index % 12],
            'key_confidence':          key_confidence,
            # spectral_roughness: proxy for timbral irregularity
            'spectral_roughness':      spectral_roughness,
        }
        for j in range(self.n_mfcc):
            f[f'mfcc_{j+1}_mean'] = float(mfcc_mean[j])
            f[f'mfcc_{j+1}_std']  = float(mfcc_std[j])
        for j, kname in enumerate(_MUSICAL_KEYS):
            f[f'chroma_{kname}_mean'] = float(chroma_mean[j])
        for j in range(6):
            f[f'tonnetz_{j+1}_mean'] = float(tonnetz[j])

        return f

    def extract_batch(
        self,
        audio_batch: np.ndarray,
        lengths: np.ndarray,
        sr: int,
        file_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """
        Extract audio features for a batch of files.

        Spectrogram computation is delegated to TTStftKernel (streaming, TT-Lang
        backed with NumPy fallback).  Downstream features (MFCC, chroma, tonnetz,
        spectral stats, ZCR, RMS) are computed by _extract_from_spectrogram using JAX
        for matmul-heavy ops.  Tempo tracking remains on CPU via librosa because
        lax.scan is not yet supported by the TT PJRT backend (stablehlo.while).

        Args:
            audio_batch: (B, max_len) float32, zero-padded to uniform length
            lengths:     (B,) int32, actual sample count per file
            sr:          sample rate (uniform across batch; 22050 for TT device)
            file_paths:  original file paths for 'filename' metadata field

        Returns:
            List of B feature dicts with same keys as feature_extraction_base.py
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not importable in this environment. "
                "Install jax[cpu] or activate the p300c-xla-test venv."
            )

        B = audio_batch.shape[0]
        if len(file_paths) != B:
            raise ValueError(f"file_paths length {len(file_paths)} != batch size {B}")

        from audio_analysis.core.tt_stft_kernel import TTStftKernel  # noqa: PLC0415
        stft_kernel = TTStftKernel(
            sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )

        features_list = []
        for i in range(B):
            # Slice to the true signal length (drop zero-padding from batching)
            audio_i = audio_batch[i, :int(lengths[i])].astype(np.float32)
            # TTStftKernel handles streaming in 30-second chunks and stitches results.
            # Returns a SpectrogramChunk with .mag (n_frames, n_freqs) and
            # .mel (n_frames, n_mels) over the full file.
            chunk = stft_kernel.process_file(audio_i, sr=sr)
            f = self._extract_from_spectrogram(
                chunk.mag, chunk.mel, audio_i, sr, file_paths[i]
            )
            features_list.append(f)

        # Tempo uses librosa beat tracker (lax.scan not yet supported on TT PJRT)
        tempo_results = self._extract_tempo_cpu(audio_batch, lengths)
        for i, f in enumerate(features_list):
            f.update(tempo_results[i])

        return features_list


def jax_kmeans(
    features: np.ndarray,
    n_clusters: int,
    n_iter: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering via JAX vmap distance steps, fixed n_iter iterations.

    Uses a Python for-loop (JAX traces/unrolls it) instead of lax.while_loop
    because stablehlo.while is not yet supported by the TT PJRT backend.

    Args:
        features: (n_samples, n_features) float32
        n_clusters: number of clusters
        n_iter: fixed Lloyd's iterations (default 50)
        seed: random seed for k-means++ initialisation (runs on CPU)

    Returns:
        labels: (n_samples,) int32
        centers: (n_clusters, n_features) float32
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not available")

    n_samples, n_feats = features.shape

    # k-means++ initialisation on CPU (fast, one-time)
    rng = np.random.default_rng(seed)
    center_indices = [int(rng.integers(n_samples))]
    for _ in range(1, n_clusters):
        dists = np.min(
            np.sum(
                (features[:, None, :] - features[center_indices][None, :, :]) ** 2,
                axis=-1,
            ),
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-8)
        center_indices.append(int(rng.choice(n_samples, p=probs)))
    init_centers = features[center_indices]

    jfeatures = jnp.array(features)
    centers   = jnp.array(init_centers)

    # Fixed-iteration Lloyd's — Python loop, JAX traces and unrolls the graph
    for _ in range(n_iter):
        # (k, n, d) distances
        diffs    = jfeatures[None, :, :] - centers[:, None, :]
        sq_dists = jnp.sum(diffs ** 2, axis=-1)            # (k, n)
        # TT PJRT constraint: argmin/argmax only supported on the last axis.
        # Workaround: transpose (k,n) -> (n,k), negate, argmax over last dim.
        # Mathematically equivalent to argmin(sq_dists, axis=0).
        labels_j = jnp.argmax(-sq_dists.T, axis=-1)        # (n,)

        # Update centroids via one-hot aggregation
        one_hot = (labels_j[None, :] == jnp.arange(n_clusters)[:, None]).astype(jnp.float32)
        counts  = jnp.sum(one_hot, axis=1, keepdims=True) + 1e-8
        centers = jnp.dot(one_hot, jfeatures) / counts     # (k, d)

    return np.array(labels_j).astype(np.int32), np.array(centers).astype(np.float32)
