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
            except Exception:
                tempo_val = 0.0
                onset_density = 0.0
            results.append({
                'tempo': tempo_val,
                'onset_density': onset_density,
                'beat_count': 0,  # placeholder: lax.scan DP pending TT PJRT support
            })
        return results

    def extract_batch(
        self,
        audio_batch: np.ndarray,
        lengths: np.ndarray,
        sr: int,
        file_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """
        Extract audio features for a batch of files on TT hardware.

        Args:
            audio_batch: (B, max_len) float32, zero-padded
            lengths: (B,) int32, actual sample count per file
            sr: sample rate (uniform across batch; 22050 when device='tenstorrent')
            file_paths: original file paths for filename metadata

        Returns:
            List of feature dicts with same keys as feature_extraction_base.py
        """
        if not _JAX_AVAILABLE:
            raise RuntimeError(
                "JAX is not importable in this environment. "
                "Install jax[cpu] or activate the p300c-xla-test venv."
            )

        B = audio_batch.shape[0]
        assert len(file_paths) == B, f"Batch size mismatch: {B} vs {len(file_paths)}"

        max_len = audio_batch.shape[1]
        n_frames = max(1, (max_len - self.n_fft) // self.hop_length + 1)

        # Build frame index tensor: (n_frames, n_fft)
        frame_starts = np.arange(n_frames) * self.hop_length
        frame_indices = frame_starts[:, None] + np.arange(self.n_fft)[None, :]
        # Clip to max_len so we never index out of bounds on padded audio
        frame_indices = np.clip(frame_indices, 0, max_len - 1)

        # Boolean mask: True for frames whose start sample falls within the
        # actual (non-padded) signal.  Shape: (n_frames,).
        # Stored as float32 (1.0 / 0.0) per file so vmap can take it as
        # a per-sample argument.
        frame_valid_masks = (frame_starts[None, :] < lengths[:, None]).astype(np.float32)
        # shape: (B, n_frames)

        # Move static arrays to JAX once
        jdft_cos  = jnp.array(self.dft_cos)          # (n_freqs, n_fft)
        jdft_sin  = jnp.array(self.dft_sin)          # (n_freqs, n_fft)
        jmel      = jnp.array(self.mel_filterbank)   # (n_freqs, n_mels)
        jdct      = jnp.array(self.dct_matrix)       # (n_mels, n_mfcc)
        jchroma   = jnp.array(self.chroma_filter)    # (n_freqs, 12)
        jtonnetz  = jnp.array(self.tonnetz_transform) # (12, 6)
        jfreq_hz  = jnp.array(self.freq_hz)          # (n_freqs,)
        jhann     = jnp.array(self.hann_window)      # (n_fft,)
        jfi       = jnp.array(frame_indices)         # (n_frames, n_fft)

        jaudio  = jnp.array(audio_batch)             # (B, max_len)
        jmasks  = jnp.array(frame_valid_masks)       # (B, n_frames)

        def extract_one(audio_1d: jnp.ndarray, valid_mask: jnp.ndarray):
            """
            Extract all features for a single audio signal.

            valid_mask: (n_frames,) float32 — 1.0 for valid frames, 0.0 for
            zero-padded frames beyond the true signal length.  Used as
            per-frame weights so padding doesn't bias spectral estimates.
            """
            # --- Frame + window ---
            frames = audio_1d[jfi] * jhann           # (n_frames, n_fft)

            # --- DFT via matmul (rfft not supported on TT PJRT) ---
            real_part = jnp.dot(frames, jdft_cos.T)  # (n_frames, n_freqs)
            imag_part = jnp.dot(frames, jdft_sin.T)  # (n_frames, n_freqs)
            mag = jnp.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)  # (n_frames, n_freqs)

            # Zero-out magnitude for padding frames so they don't bias statistics.
            # valid_mask[:, None] broadcasts to (n_frames, n_freqs).
            mag = mag * valid_mask[:, None]

            # Normalised weight per frame: used for weighted mean/std below.
            # Sum of valid_mask gives count of valid frames (safe: always >= 1).
            n_valid = jnp.sum(valid_mask) + 1e-8           # scalar

            # --- Spectral features ---
            mag_sum  = jnp.sum(mag, axis=-1, keepdims=True) + 1e-8
            mag_norm = mag / mag_sum                 # (n_frames, n_freqs)

            centroid = jnp.sum(mag_norm * jfreq_hz, axis=-1)  # (n_frames,)
            # Weighted mean and std over valid frames only
            centroid_mean = jnp.sum(centroid * valid_mask) / n_valid
            centroid_var  = jnp.sum(((centroid - centroid_mean) ** 2) * valid_mask) / n_valid
            centroid_std  = jnp.sqrt(centroid_var + 1e-8)

            cumsum = jnp.cumsum(mag, axis=-1)        # (n_frames, n_freqs)
            total  = cumsum[:, -1:] + 1e-8
            rolloff_mask = (cumsum >= 0.85 * total).astype(jnp.float32)
            rolloff_bin  = jnp.argmax(rolloff_mask, axis=-1).astype(jnp.float32)
            rolloff_hz   = rolloff_bin * (sr / 2.0) / (self.n_freqs - 1)
            rolloff_mean = jnp.sum(rolloff_hz * valid_mask) / n_valid

            dev       = (jfreq_hz - centroid[:, None]) ** 2        # (n_frames, n_freqs)
            bandwidth = jnp.sqrt(jnp.sum(mag_norm * dev, axis=-1) + 1e-8)
            bandwidth_mean = jnp.sum(bandwidth * valid_mask) / n_valid

            # --- ZCR (from frames, valid frames only) ---
            signs    = jnp.sign(frames)
            # zcr per frame: mean of abs sign-changes over n_fft-1 transitions
            zcr_per_frame = jnp.mean(jnp.abs(jnp.diff(signs, axis=-1)), axis=-1) / 2.0
            zcr_mean = jnp.sum(zcr_per_frame * valid_mask) / n_valid

            # --- RMS (from frames, valid frames only) ---
            rms      = jnp.sqrt(jnp.mean(frames ** 2, axis=-1))     # (n_frames,)
            rms_mean = jnp.sum(rms * valid_mask) / n_valid
            rms_var  = jnp.sum(((rms - rms_mean) ** 2) * valid_mask) / n_valid
            rms_std  = jnp.sqrt(rms_var + 1e-8)

            # --- MFCC ---
            mel     = jnp.dot(mag, jmel)             # (n_frames, n_mels)
            log_mel = jnp.log(mel + 1e-6)
            mfcc    = jnp.dot(log_mel, jdct)         # (n_frames, n_mfcc)
            # Weighted mean/std over valid frames
            mfcc_mean = jnp.sum(mfcc * valid_mask[:, None], axis=0) / n_valid
            mfcc_var  = jnp.sum(((mfcc - mfcc_mean[None, :]) ** 2) * valid_mask[:, None], axis=0) / n_valid
            mfcc_std  = jnp.sqrt(mfcc_var + 1e-8)

            # --- Chroma ---
            chroma      = jnp.dot(mag, jchroma)      # (n_frames, 12)
            chroma_norm = chroma / (jnp.sum(chroma, axis=-1, keepdims=True) + 1e-6)
            chroma_mean = jnp.sum(chroma_norm * valid_mask[:, None], axis=0) / n_valid
            key_index      = jnp.argmax(chroma_mean)
            key_confidence = jnp.max(chroma_mean)

            # --- Tonnetz ---
            tonnetz = jnp.dot(chroma_mean, jtonnetz)  # (6,)

            # --- Spectral roughness ---
            # Mean absolute difference between adjacent frequency bins across all
            # valid frames — a proxy for spectral irregularity.  High values
            # indicate rough or noisy timbres; low values indicate smooth/tonal spectra.
            # spec shape is (n_frames, n_freqs); diff along axis=-1 gives (n_frames, n_freqs-1).
            spec_diff = jnp.abs(jnp.diff(mag, axis=-1))           # (n_frames, n_freqs-1)
            spectral_roughness = jnp.mean(spec_diff)               # scalar

            return (
                centroid_mean, centroid_std,
                rolloff_mean, bandwidth_mean, zcr_mean,
                rms_mean, rms_std,
                mfcc_mean, mfcc_std,
                chroma_mean,
                key_index, key_confidence,
                tonnetz,
                spectral_roughness,
            )

        # vmap over batch dimension (audio signals and their validity masks)
        results = vmap(extract_one)(jaudio, jmasks)

        (
            centroid_mean, centroid_std,
            rolloff_mean, bandwidth_mean, zcr_mean,
            rms_mean, rms_std,
            mfcc_mean, mfcc_std,
            chroma_mean,
            key_indices, key_confidences,
            tonnetz,
            spectral_roughness,
        ) = results

        # Pull to numpy in one transfer
        centroid_mean      = np.array(centroid_mean)
        centroid_std       = np.array(centroid_std)
        rolloff_mean       = np.array(rolloff_mean)
        bandwidth_mean     = np.array(bandwidth_mean)
        zcr_mean           = np.array(zcr_mean)
        rms_mean           = np.array(rms_mean)
        rms_std            = np.array(rms_std)
        mfcc_mean          = np.array(mfcc_mean)
        mfcc_std           = np.array(mfcc_std)
        chroma_mean        = np.array(chroma_mean)
        key_indices        = np.array(key_indices)
        key_confidences    = np.array(key_confidences)
        tonnetz            = np.array(tonnetz)
        spectral_roughness = np.array(spectral_roughness)

        features_list = []
        for i in range(B):
            f: Dict[str, Any] = {
                'filename':               file_paths[i].name,
                # Duration in seconds: actual sample count divided by sample rate.
                # This matches the 'duration' field produced by feature_extraction_base.py
                # and is required by the parallel_analyzer creative-analysis pipeline.
                'duration':               float(lengths[i]) / float(sr),
                'spectral_centroid_mean': float(centroid_mean[i]),
                'spectral_centroid_std':  float(centroid_std[i]),
                'spectral_rolloff_mean':  float(rolloff_mean[i]),
                'spectral_bandwidth_mean': float(bandwidth_mean[i]),
                'zero_crossing_rate_mean': float(zcr_mean[i]),
                'rms_mean':               float(rms_mean[i]),
                'rms_std':                float(rms_std[i]),
                # 'key' matches the field name used in feature_extraction_base.py
                'key':                    _MUSICAL_KEYS[int(key_indices[i]) % 12],
                'key_confidence':         float(key_confidences[i]),
                # spectral_roughness: mean absolute adjacent-bin difference across
                # valid frames — proxy for timbral roughness / spectral irregularity.
                'spectral_roughness':     float(spectral_roughness[i]),
            }
            for j in range(self.n_mfcc):
                f[f'mfcc_{j+1}_mean'] = float(mfcc_mean[i, j])
                f[f'mfcc_{j+1}_std']  = float(mfcc_std[i, j])
            for j, key_name in enumerate(_MUSICAL_KEYS):
                f[f'chroma_{key_name}_mean'] = float(chroma_mean[i, j])
            for j in range(6):
                f[f'tonnetz_{j+1}_mean'] = float(tonnetz[i, j])
            features_list.append(f)

        # Merge CPU tempo results (lax.scan not yet supported on TT PJRT)
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
