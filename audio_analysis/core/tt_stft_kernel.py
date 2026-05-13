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
        raise NotImplementedError("process_chunk implemented in Task 2")

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
