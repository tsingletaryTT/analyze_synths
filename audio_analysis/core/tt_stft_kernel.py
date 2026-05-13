"""
TT-Lang fused STFT kernel wrapper.

Streams audio in 30-second chunks. Each chunk runs a fused pipeline:
  framing → Hann window → DFT (matmul) → magnitude → mel filterbank

When TT hardware is unavailable, falls back to the same computation via NumPy.

SpectrogramChunk is the shared interface between this kernel and the
trajectory analysis layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SpectrogramChunk:
    """
    Output of one STFT chunk (or stitched full-file result).

    mag        : (n_frames, n_fft//2+1) float32 — magnitude spectrum
    mel        : (n_frames, n_mels)      float32 — mel spectrogram
    timestamps : (n_frames,)             float32 — center time of each frame (s)
    """
    mag: np.ndarray
    mel: np.ndarray
    timestamps: np.ndarray


class TTStftKernel:
    """
    Fused STFT kernel wrapper with streaming and TT-Lang dispatch.

    Parameters
    ----------
    sr             : sample rate (default 22050)
    n_fft          : FFT size (default 2048)
    hop_length     : STFT hop in samples (default 512)
    n_mels         : mel filterbank bands (default 128)
    chunk_seconds  : chunk size for streaming (default 30.0)
    overlap_seconds: overlap between chunks (default 2.0)
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_freqs = n_fft // 2 + 1
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        self.chunk_samples   = int(chunk_seconds  * sr)
        self.overlap_samples = int(overlap_seconds * sr)

        # Precompute static matrices (same as in jax_feature_extraction.py)
        self._cos_basis, self._sin_basis, self._mel_filter, self._hann = \
            self._build_stft_constants(sr, n_fft, n_mels)

    @staticmethod
    def _build_stft_constants(sr: int, n_fft: int, n_mels: int):
        """
        Precompute static STFT matrices (computed once at init, reused per chunk).

        All four matrices are float32 to match the Tensix tile dtype and to keep
        memory bandwidth low.  The mel_filter is transposed from librosa's default
        (n_mels, n_freqs) to (n_freqs, n_mels) so that `mag @ mel_filter` works
        without an extra transpose on the hot path.

        Returns
        -------
        cos_basis  : (n_fft, n_freqs) float32 — cosine DFT basis
        sin_basis  : (n_fft, n_freqs) float32 — sine DFT basis
        mel_filter : (n_freqs, n_mels) float32 — mel filterbank (transposed)
        hann       : (n_fft,) float32 — Hann window
        """
        import librosa

        n_freqs = n_fft // 2 + 1
        # DFT basis: cos_basis[t, k] = cos(2π * t * k / n_fft)
        # Using float64 for the phase computation to avoid accumulation error,
        # then cast to float32 once the trig is done.
        freqs = np.arange(n_freqs, dtype=np.float64)
        t_idx = np.arange(n_fft, dtype=np.float64)
        phase = 2.0 * np.pi * np.outer(t_idx, freqs) / n_fft
        cos_basis = np.cos(phase).astype(np.float32)   # (n_fft, n_freqs)
        sin_basis = np.sin(phase).astype(np.float32)   # (n_fft, n_freqs)

        # Mel filterbank: librosa returns (n_mels, n_freqs); transpose to (n_freqs, n_mels)
        mel_filter = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels
        ).T.astype(np.float32)   # (n_freqs, n_mels)

        hann = np.hanning(n_fft).astype(np.float32)    # (n_fft,)
        return cos_basis, sin_basis, mel_filter, hann

    def _process_chunk_numpy(
        self,
        audio_chunk: np.ndarray,
        chunk_start_time: float = 0.0,
    ) -> SpectrogramChunk:
        """
        NumPy fallback: framing → Hann window → DFT matmul → magnitude → mel.

        This is the reference implementation; TT-Lang runs the same math on Tensix.

        The DFT is expressed as two matmuls (cosine and sine projections) rather
        than np.fft.rfft so that the identical operation can be lowered to a pair
        of Tensix matmul tiles, keeping TT-Lang parity exact.

        Parameters
        ----------
        audio_chunk      : (n_samples,) — raw audio (cast to float32 internally)
        chunk_start_time : time in seconds of the first sample in this chunk

        Returns
        -------
        SpectrogramChunk with mag, mel, and timestamps arrays all float32
        """
        n_fft = self.n_fft
        hop   = self.hop_length
        audio = audio_chunk.astype(np.float32)

        # ------ Frame the audio ------
        n_samples = len(audio)
        # Number of complete frames that fit without zero-padding
        n_frames = max(0, (n_samples - n_fft) // hop + 1)
        if n_frames == 0:
            return SpectrogramChunk(
                mag=np.zeros((0, self.n_freqs), dtype=np.float32),
                mel=np.zeros((0, self.n_mels), dtype=np.float32),
                timestamps=np.zeros(0, dtype=np.float32),
            )

        # Build frame matrix (n_frames, n_fft) using stride tricks to avoid copy.
        # We copy afterwards so that the in-place Hann multiply is safe.
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, n_fft),
            strides=(audio.strides[0] * hop, audio.strides[0]),
        ).copy()   # copy so we can modify in-place

        # ------ Apply Hann window ------
        frames *= self._hann  # broadcast (n_frames, n_fft) * (n_fft,)

        # ------ DFT via matmul ------
        # frames @ cos_basis: (n_frames, n_fft) × (n_fft, n_freqs) → (n_frames, n_freqs)
        cos_proj = frames @ self._cos_basis   # (n_frames, n_freqs)
        sin_proj = frames @ self._sin_basis   # (n_frames, n_freqs)
        mag = np.sqrt(cos_proj ** 2 + sin_proj ** 2).astype(np.float32)

        # ------ Mel filterbank ------
        # mag @ mel_filter: (n_frames, n_freqs) × (n_freqs, n_mels) → (n_frames, n_mels)
        mel = (mag @ self._mel_filter).astype(np.float32)  # (n_frames, n_mels)

        # ------ Timestamps: center of each frame ------
        # Frame i starts at sample i*hop; center is at sample i*hop + n_fft//2
        frame_centers = (np.arange(n_frames) * hop + n_fft // 2) / self.sr
        timestamps = (frame_centers + chunk_start_time).astype(np.float32)

        return SpectrogramChunk(mag=mag, mel=mel, timestamps=timestamps)

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        chunk_start_time: float = 0.0,
    ) -> SpectrogramChunk:
        """
        Process one audio chunk. Dispatch to TT-Lang if available, else NumPy.

        Parameters
        ----------
        audio_chunk      : (n_samples,) float32 — raw audio
        chunk_start_time : time in seconds of the first sample

        Returns
        -------
        SpectrogramChunk
        """
        # TT-Lang dispatch wired in Task 5; for now, NumPy path only
        return self._process_chunk_numpy(audio_chunk, chunk_start_time)

    def process_file(
        self,
        audio: np.ndarray,
        sr: int | None = None,
    ) -> SpectrogramChunk:
        """
        Convenience: iterate chunks, stitch, return full-file SpectrogramChunk.

        Parameters
        ----------
        audio : (n_samples,) float32 — full audio array
        sr    : if provided, overrides self.sr (must match self.sr)

        Returns
        -------
        SpectrogramChunk covering the full file
        """
        raise NotImplementedError("process_file implemented in Task 3")
