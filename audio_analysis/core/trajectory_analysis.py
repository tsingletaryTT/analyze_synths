# audio_analysis/core/trajectory_analysis.py
"""
TrajectoryAnalyzer: compute 2-second feature windows across a full piece.

Produces List[TrajectoryPoint] from a SpectrogramChunk + raw audio.
Tension score is computed in a second pass after min-max normalization
across all windows, so it is relative within the piece.

Window design
-------------
The 2-second non-overlapping window (window_sec=2.0) is chosen to be:
  - Long enough to capture a full LFO cycle at 0.5 Hz
  - Short enough to track fast energy rises and sudden timbral shifts
  - Non-overlapping so that each point is statistically independent

At sr=22050, hop=512: window_frames = ceil(2 * 22050 / 512) = 87 frames.
A 14-second piece therefore yields ~7 trajectory points, giving a
coarse but representative temporal arc.

Tension formula
---------------
After all windows are computed, each of the four raw components
(energy, roughness, chroma_spread, zcr) is min-max normalized to [0,1]
relative to the piece's own dynamic range.  This makes the tension score
a *relative* measure — the tensest moment in any piece scores ~1.0
regardless of absolute loudness.

Tension = 0.35 * energy_norm
        + 0.25 * roughness_norm
        + 0.20 * (1 - chroma_spread_norm)   # narrow chroma = more tense
        + 0.20 * zcr_norm

Narrow chroma spread is more tense because a single sustained tone under
tension sounds more threatening than diffuse noise.  This weight mirrors
how composers use pedal tones and tritones to create tension.
"""
import math
from typing import List

import librosa
import numpy as np

from audio_analysis.core.narrative_types import TrajectoryPoint

# Standard 12 chromatic pitch class names (same order librosa uses: C, C#, ...)
CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class TrajectoryAnalyzer:
    """
    Slide a 2-second non-overlapping window across a SpectrogramChunk,
    computing per-window features and combining them into a tension score.

    Features computed per window
    ----------------------------
    energy        : RMS of magnitude spectrum (perceived loudness proxy)
    brightness    : Spectral centroid in Hz (frequency centre of mass)
    roughness     : Mean absolute difference between adjacent freq bins
    zcr           : Zero-crossing rate from raw audio (noisiness proxy)
    chroma_peak   : Dominant pitch class name at this moment
    chroma_spread : Normalized chroma entropy, 0 (tonal) to 1 (noise-like)
    tension_score : 0–1 composite, computed after full-piece normalization

    Parameters
    ----------
    sr          : int   — sample rate in Hz (default 22050)
    hop_length  : int   — STFT hop size in samples (default 512)
    window_sec  : float — window duration in seconds (default 2.0)
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512,
                 window_sec: float = 2.0):
        self.sr = sr
        self.hop_length = hop_length
        # Number of spectrogram frames per window, rounded up so we never
        # leave a partial second of audio un-analyzed at the window edge.
        self.window_frames = math.ceil(window_sec * sr / hop_length)  # ~87 at default settings

        # n_fft drives both the chroma filter and the frequency axis.
        # 2048 is the standard librosa default; must match the STFT that
        # produced the SpectrogramChunk.mag array (1025 bins = n_fft//2 + 1).
        n_fft = 2048

        # Chroma filter matrix: (12, n_fft//2+1) = (12, 1025)
        # Precomputed once and reused for every window.  librosa.filters.chroma
        # returns the matrix in (n_chroma, n_freqs) layout — perfect for
        # filter @ mean_mag_vector multiplication.
        self._chroma_filter = librosa.filters.chroma(
            sr=sr, n_fft=n_fft, n_chroma=12
        ).astype(np.float32)   # (12, 1025)

        # Frequency axis in Hz corresponding to each STFT bin: (1025,)
        # rfftfreq returns evenly spaced values from 0 Hz to sr/2.
        self._freq_hz = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)  # (1025,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, chunk, audio: np.ndarray) -> List[TrajectoryPoint]:
        """
        Compute a trajectory of feature snapshots across the full piece.

        Parameters
        ----------
        chunk : SpectrogramChunk
            mag shape (n_frames, 1025) float32 — magnitude spectrogram
            timestamps shape (n_frames,) float32 — center time per frame (s)
        audio : np.ndarray
            Raw waveform (n_samples,) float32 — used for ZCR computation.
            Must be at self.sr sample rate.

        Returns
        -------
        List[TrajectoryPoint]
            One point per 2-second window.  The last partial window is
            included only if it is at least half a full window length,
            to avoid noisy estimates from very short tail slices.

        Notes
        -----
        Tension scores are computed in a second pass after all windows
        have been processed, so they are normalized relative to the
        piece's own dynamic range rather than any absolute scale.
        """
        mag = chunk.mag          # (n_frames, n_freqs)
        ts = chunk.timestamps    # (n_frames,)
        n_frames = mag.shape[0]

        if n_frames == 0:
            return []

        # ------ Pass 1: slide windows, collect raw feature values ------
        # Each entry: (center_time, energy, brightness, roughness, zcr,
        #              chroma_peak_name, chroma_spread)
        raw_points = []
        step = self.window_frames  # non-overlapping: advance by full window each time

        for start in range(0, n_frames, step):
            end = min(start + self.window_frames, n_frames)

            # Skip tail windows shorter than half the window length; they
            # would produce unreliable feature estimates, especially for
            # roughness and chroma which benefit from averaging over time.
            if end - start < self.window_frames // 2:
                break

            mag_win = mag[start:end]    # (w, 1025) — slice of the magnitude spectrogram

            # Use the mid-frame timestamp as the "center time" for this point.
            # Integer arithmetic on indices avoids floating-point rounding of
            # the timestamp array.
            center_time = float(ts[(start + end) // 2])

            # --- Spectral features from magnitude spectrum ---
            energy = self._rms(mag_win)
            brightness = self._centroid(mag_win)
            roughness = self._roughness(mag_win)
            chroma_peak, chroma_spread = self._chroma(mag_win)

            # --- ZCR from raw audio ---
            # Map spectrogram frame index to sample index.  The end of the
            # last frame in the window aligns to end * hop_length; we clip
            # to len(audio) to handle edge cases where the audio is slightly
            # shorter than the spectrogram due to padding differences.
            audio_start = start * self.hop_length
            audio_end = min(end * self.hop_length, len(audio))
            audio_win = audio[audio_start:audio_end]
            zcr = self._zcr(audio_win)

            raw_points.append((center_time, energy, brightness,
                                roughness, zcr, chroma_peak, chroma_spread))

        if not raw_points:
            return []

        # ------ Pass 2: min-max normalize then compute tension ------
        # Extract only the four components that feed into tension, packed
        # into a (N, 4) array for vectorized normalization.
        # Column order: energy, roughness, chroma_spread, zcr
        arr = np.array([(e, r, cs, z) for (_, e, _, r, z, _, cs) in raw_points],
                       dtype=np.float32)   # (N, 4)

        def _norm_col(col: np.ndarray) -> np.ndarray:
            """Min-max normalize a column to [0, 1].
            Returns all zeros when the column is constant (avoids division
            by zero and produces a neutral contribution to tension)."""
            lo, hi = col.min(), col.max()
            if hi == lo:
                return np.zeros_like(col)
            return (col - lo) / (hi - lo)

        e_n = _norm_col(arr[:, 0])   # energy (higher = more tense)
        r_n = _norm_col(arr[:, 1])   # roughness (higher = more tense)
        cs_n = _norm_col(arr[:, 2])  # chroma_spread (inverted: narrow = more tense)
        z_n = _norm_col(arr[:, 3])   # ZCR (higher = more tense)

        # Weighted sum.  Weights reflect perceptual importance:
        #   - Energy (0.35): the primary driver of perceived tension
        #   - Roughness (0.25): gritty/noisy textures feel tense
        #   - Chroma narrowness (0.20): sustained single-note pedal = tense
        #   - ZCR (0.20): fast sign changes correlate with noise and agitation
        tension = (0.35 * e_n
                   + 0.25 * r_n
                   + 0.20 * (1.0 - cs_n)   # inverted: narrow chroma = more tense
                   + 0.20 * z_n)
        tension = np.clip(tension, 0.0, 1.0)

        # ------ Assemble TrajectoryPoint objects ------
        points: List[TrajectoryPoint] = []
        for i, (center_time, energy, brightness, roughness, zcr,
                chroma_peak, chroma_spread) in enumerate(raw_points):
            points.append(TrajectoryPoint(
                time=center_time,
                energy=float(energy),
                brightness=float(brightness),
                roughness=float(roughness),
                zcr=float(zcr),
                chroma_peak=chroma_peak,
                chroma_spread=float(chroma_spread),
                tension_score=float(tension[i]),
            ))
        return points

    # ------------------------------------------------------------------
    # Private feature helpers
    # ------------------------------------------------------------------

    def _rms(self, mag_win: np.ndarray) -> float:
        """
        Root mean square of magnitude values across the window.

        Using the magnitude spectrum (not the raw waveform) gives a
        frequency-weighted energy estimate that correlates well with
        perceived loudness, since it reflects spectral power rather than
        clip-level amplitude.

        Parameters
        ----------
        mag_win : (w, n_freqs) float32

        Returns
        -------
        float — RMS energy, always >= 0
        """
        return float(np.sqrt(np.mean(mag_win ** 2)))

    def _centroid(self, mag_win: np.ndarray) -> float:
        """
        Spectral centroid (brightness) in Hz.

        Computed as the frequency-weighted mean over the time-averaged
        magnitude spectrum.  Bright sounds (cymbals, leads, metallic
        percussion) score high; dark pads and sub-bass score low.

        Parameters
        ----------
        mag_win : (w, n_freqs) float32

        Returns
        -------
        float — centroid frequency in Hz, 0 if spectrum is silent
        """
        mean_mag = np.mean(mag_win, axis=0)   # (n_freqs,) — average over time axis
        denom = mean_mag.sum()
        if denom < 1e-9:
            return 0.0
        # dot product of Hz positions and normalized magnitude = weighted average
        return float(np.dot(self._freq_hz, mean_mag) / denom)

    def _roughness(self, mag_win: np.ndarray) -> float:
        """
        Spectral roughness: mean absolute difference between adjacent freq bins.

        A smooth spectrum (pure tones, sine-wave pads) has nearly equal
        neighboring bins and therefore low roughness.  White noise or
        heavily distorted signals create jagged bin-to-bin variations,
        yielding high roughness values.

        np.diff(..., axis=-1) differences along the frequency axis,
        producing a (w, n_freqs-1) array; we then take the mean over
        both time and frequency dimensions.

        Parameters
        ----------
        mag_win : (w, n_freqs) float32

        Returns
        -------
        float — mean absolute inter-bin difference, always >= 0
        """
        diff = np.abs(np.diff(mag_win, axis=-1))   # (w, n_freqs-1)
        return float(diff.mean())

    def _chroma(self, mag_win: np.ndarray) -> tuple:
        """
        Compute chroma features from the window's magnitude spectrum.

        Applies the precomputed chroma filterbank to the time-averaged
        magnitude spectrum to get a 12-dimensional chroma vector, then
        derives:
          - peak note name: the pitch class with the most energy
          - chroma spread: normalized entropy of the chroma distribution
            (0 = single dominant pitch class, 1 = uniform / atonal)

        Entropy-based spread is preferable to raw variance here because
        entropy captures multi-modal distributions (e.g., power chord
        on two pitch classes) that variance would underestimate.

        Parameters
        ----------
        mag_win : (w, n_freqs) float32

        Returns
        -------
        (peak_note_name: str, chroma_spread: float)
        """
        mean_mag = np.mean(mag_win, axis=0)          # (n_freqs,) — time-average
        chroma = self._chroma_filter @ mean_mag      # (12,) — project onto pitch classes
        chroma = np.maximum(chroma, 0.0)             # clamp negatives from filter rounding

        peak_idx = int(np.argmax(chroma))
        peak_name = CHROMA_NAMES[peak_idx]

        # Normalized Shannon entropy: 0 = completely tonal, 1 = uniform noise
        total = chroma.sum()
        if total < 1e-9:
            # Silent window: return a neutral spread and the first note name
            return peak_name, 0.5
        p = chroma / total
        # Small epsilon inside log prevents log(0) while barely affecting entropy
        entropy = -np.sum(p * np.log(p + 1e-9))
        # Maximum possible entropy for 12 equally likely classes = log(12)
        max_entropy = np.log(12.0)
        spread = float(np.clip(entropy / max_entropy, 0.0, 1.0))
        return peak_name, spread

    def _zcr(self, audio_win: np.ndarray) -> float:
        """
        Zero-crossing rate: fraction of consecutive sample pairs with opposite sign.

        ZCR complements roughness by operating on the raw waveform rather
        than the frequency domain.  Noise signals cross zero very frequently
        (high ZCR); sustained low-frequency tones rarely cross zero (low ZCR).

        The result is expressed as crossings per sample (not per second),
        so it is scale-invariant with respect to sample rate.

        Parameters
        ----------
        audio_win : (n_samples,) float32

        Returns
        -------
        float — ZCR in [0, 1], or 0.0 for windows shorter than 2 samples
        """
        if len(audio_win) < 2:
            return 0.0
        signs = np.sign(audio_win)
        # Count pairs where the sign changes (including zero-to-nonzero transitions)
        crossings = np.sum(signs[:-1] != signs[1:])
        return float(crossings) / (len(audio_win) - 1)
