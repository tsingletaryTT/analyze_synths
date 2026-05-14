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

TILE = 32  # TT-Lang Tensix tile size — all dimensions must be multiples of this


def _pad_to_tile(n: int) -> int:
    """Round `n` up to the nearest multiple of TILE (32)."""
    return ((n + TILE - 1) // TILE) * TILE


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

        # Detect TT-Lang simulator availability once at init.
        # Cached as a boolean so the import probe is not repeated per chunk.
        self._try_ttl = self._detect_ttl()
        if self._try_ttl:
            log.info("TT-Lang simulator available — using fused STFT kernel")
        else:
            log.info("TT-Lang unavailable — using NumPy STFT fallback")

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

    @staticmethod
    def _detect_ttl() -> bool:
        """
        Return True if the TT-Lang simulator is importable.

        Called once at init; result is cached in ``self._try_ttl``.  Importing
        ``tt_stft_sim`` also triggers the TT-Lang ``sys.path`` injection, so
        the cost is a single import probe rather than repeated path manipulation.
        """
        try:
            from audio_analysis.core import tt_stft_sim  # noqa: F401
            return True
        except ImportError:
            return False

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
        mag = np.sqrt(cos_proj ** 2 + sin_proj ** 2 + 1e-8).astype(np.float32)

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
        Process one audio chunk.

        Dispatch order:
          1. TT-Lang fused STFT (if simulator or hardware available)
          2. NumPy DFT-matmul fallback

        If the TT-Lang dispatch raises any exception on the first attempt,
        ``self._try_ttl`` is flipped to False so subsequent chunks skip the
        import overhead and go straight to NumPy.

        Parameters
        ----------
        audio_chunk      : (n_samples,) float32 — raw audio
        chunk_start_time : time in seconds of the first sample

        Returns
        -------
        SpectrogramChunk
        """
        if self._try_ttl:
            try:
                from audio_analysis.core.tt_stft_sim import fused_stft_sim
                return fused_stft_sim(audio_chunk, self, chunk_start_time)
            except Exception as e:
                log.warning(
                    "TT-Lang dispatch failed (%s); falling back to NumPy", e
                )
                # Disable TT-Lang for all subsequent chunks in this session
                # to avoid repeated import / dispatch overhead on a broken path.
                self._try_ttl = False

        return self._process_chunk_numpy(audio_chunk, chunk_start_time)

    def process_file(
        self,
        audio: np.ndarray,
        sr: int | None = None,
    ) -> SpectrogramChunk:
        """
        Process full audio file by streaming 30-second chunks with 2-second overlap.

        Overlap frames are discarded from the start of each subsequent chunk
        (except the first) to avoid frame duplication at chunk boundaries.

        Chunking layout
        ---------------
        chunk 0: audio[0 .. chunk_samples + overlap_samples]            → all frames kept
        chunk 1: audio[chunk_samples .. 2*chunk_samples + overlap]      → overlap frames discarded
        chunk 2: audio[2*chunk_samples .. 3*chunk_samples + overlap]    → overlap frames discarded
        ...

        Discard strategy
        ----------------
        Rather than using a fixed ceil(overlap_samples / hop_length) discard count,
        we compute exactly which frames are duplicated using each frame's center sample
        in global (full-file) coordinates.

        A frame i in chunk k (starting at global sample `pos`) has its center at:
            center_global = pos + i * hop_length + n_fft // 2

        A frame is a duplicate if its center_global falls at or before the center
        of the last frame produced by chunk k-1 (`prev_last_center`).  The number of
        such frames is:
            discard = floor((prev_last_center - pos - n_fft // 2) / hop_length) + 1

        This gives at most ±1 frame difference versus a single-pass reference,
        regardless of chunk size or audio length.

        Parameters
        ----------
        audio : (n_samples,) float32 — full audio array
        sr    : if provided and differs from self.sr, a warning is logged; the
                kernel always uses self.sr for all time calculations

        Returns
        -------
        SpectrogramChunk covering the full file, with monotonically increasing
        timestamps and no duplicate frames at chunk boundaries
        """
        if sr is not None and sr != self.sr:
            log.warning(
                "sr=%d passed to process_file but kernel sr=%d; using kernel sr",
                sr, self.sr,
            )

        audio = np.asarray(audio, dtype=np.float32)
        n_samples = len(audio)

        # Collect per-chunk arrays before final concatenation
        mags: list[np.ndarray] = []
        mels: list[np.ndarray] = []
        tss:  list[np.ndarray] = []

        pos = 0           # global sample index of the start of this chunk's clean region
        chunk_idx = 0     # 0-based counter; chunk 0 is treated specially
        # Tracks the global-sample center of the last frame we kept in the previous chunk.
        # Used to determine exactly how many frames to discard from the current chunk.
        prev_last_center: int = -1

        while pos < n_samples:
            # Each chunk covers the clean region PLUS the following overlap window.
            # The overlap window ensures that boundary frames have the same STFT
            # framing context they would have if the full file were processed at once.
            end = min(pos + self.chunk_samples + self.overlap_samples, n_samples)
            audio_chunk = audio[pos:end]

            # chunk_start_time is the time of sample `pos` in the full file.
            # process_chunk adds this offset to every frame timestamp, so the returned
            # timestamps correctly reflect positions within the original audio.
            chunk_start_time = pos / self.sr
            chunk = self.process_chunk(audio_chunk, chunk_start_time)

            n_frames = chunk.mag.shape[0]

            if chunk_idx == 0 or n_frames == 0:
                # First chunk (or empty chunk): keep everything, no prior output to deduplicate.
                if n_frames > 0:
                    mags.append(chunk.mag)
                    mels.append(chunk.mel)
                    tss.append(chunk.timestamps)
                    # Record the global-sample center of the last kept frame
                    prev_last_center = pos + (n_frames - 1) * self.hop_length + self.n_fft // 2
            else:
                # Discard all frames whose global center sample is <= prev_last_center.
                # Frame i of this chunk:
                #   center_global = pos + i * hop_length + n_fft // 2
                # Discard when: pos + i * hop + n_fft//2 <= prev_last_center
                #   i <= (prev_last_center - pos - n_fft//2) / hop
                #   discard = floor(...) + 1 (the +1 converts inclusive index to count)
                numerator = prev_last_center - pos - self.n_fft // 2
                if numerator < 0:
                    # No overlap at all (shouldn't happen with valid overlap_seconds > 0)
                    discard = 0
                else:
                    discard = int(np.floor(numerator / self.hop_length)) + 1

                discard = min(discard, n_frames)  # guard against pathological chunks
                kept = n_frames - discard

                if kept > 0:
                    mags.append(chunk.mag[discard:])
                    mels.append(chunk.mel[discard:])
                    tss.append(chunk.timestamps[discard:])
                    # Update the last-center tracker for the next iteration
                    prev_last_center = pos + (discard + kept - 1) * self.hop_length + self.n_fft // 2
                # If kept == 0 the entire chunk was overlap — skip it silently.

            # Advance by chunk_samples (not chunk_samples + overlap) so the next
            # iteration's clean region starts immediately after this one ends.
            pos += self.chunk_samples
            chunk_idx += 1

        # Handle the empty-file edge case
        if not mags:
            return SpectrogramChunk(
                mag=np.zeros((0, self.n_freqs), dtype=np.float32),
                mel=np.zeros((0, self.n_mels), dtype=np.float32),
                timestamps=np.zeros(0, dtype=np.float32),
            )

        return SpectrogramChunk(
            mag=np.concatenate(mags, axis=0),
            mel=np.concatenate(mels, axis=0),
            timestamps=np.concatenate(tss, axis=0),
        )
