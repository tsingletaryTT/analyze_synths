"""
TT-Lang simulator fused STFT kernel.

Provides `fused_stft_sim()` — a function that dispatches two TT-Lang
@ttl.operation kernels through the Tensix simulator to compute:

  framing → Hann window → DFT (cos/sin matmul) → magnitude → mel filterbank

The two-kernel design (`_stft_compute_mag` and `_stft_compute_mel`) avoids
the single-phase DFB limitation: a DataflowBuffer cannot be simultaneously
written by compute and read by the reader in the same @ttl.operation.  By
splitting into two passes — magnitude then mel — each kernel uses clean,
separate DFBs with no cross-phase sharing.

Hardware path:
    When actual TT hardware is available, the kernel dispatch uses the
    hardware backend rather than the simulator.  The Python interface
    (fused_stft_sim) remains unchanged.

Import strategy:
    This module is only imported when the TT-Lang simulator is present.
    The caller (tt_stft_kernel.process_chunk) checks for availability
    before importing.

Tile alignment constraint:
    All ttnn.Tensor dimensions must be multiples of TILE=32.
    - n_fft=2048   : 64 tiles  ✓ (multiple of 32 by construction)
    - n_mels=128   :  4 tiles  ✓ (multiple of 32 by construction)
    - n_freqs=1025 : NOT a multiple → padded to 1056 (33 × 32)
    - n_frames     : varies   → padded to nearest multiple of 32
    Padded columns/rows are zeroed so they contribute nothing to sums.

Loop ordering:
    Reader and compute threads MUST iterate in the same loop order.
    A mismatch causes the DFB pipe to deliver tiles to the wrong compute
    iteration, producing silently wrong results.  In both kernels below
    the outer-to-inner loop order is identical in both reader and compute.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the TT-Lang python path is available
# ---------------------------------------------------------------------------
_TTL_PYTHON = os.environ.get("TT_LANG_PYTHON", "/home/ttuser/code/tt-lang/python")
if _TTL_PYTHON and _TTL_PYTHON not in sys.path:
    sys.path.insert(0, _TTL_PYTHON)

from sim import ttl, ttnn  # noqa: E402  (after sys.path patch)
import torch  # noqa: E402

from audio_analysis.core.tt_stft_kernel import SpectrogramChunk, TILE, _pad_to_tile  # noqa: E402

if TYPE_CHECKING:
    from audio_analysis.core.tt_stft_kernel import TTStftKernel


# ---------------------------------------------------------------------------
# Kernel 1: framing DFT matmuls → magnitude
# ---------------------------------------------------------------------------

@ttl.operation(grid=(1, 1))
def _stft_compute_mag(
    frames_t: ttnn.Tensor,  # (n_frames_pad, n_fft)        — windowed frames
    cos_t: ttnn.Tensor,     # (n_fft, n_freqs_pad)         — cosine DFT basis
    sin_t: ttnn.Tensor,     # (n_fft, n_freqs_pad)         — sine DFT basis
    mag_out: ttnn.Tensor,   # (n_frames_pad, n_freqs_pad)  — magnitude output
) -> None:
    """
    First STFT pass: compute magnitude spectrum.

    For each output tile (ft, nt):
        cos_proj[ft, nt] = sum_kt  frames[ft, kt] @ cos[kt, nt]
        sin_proj[ft, nt] = sum_kt  frames[ft, kt] @ sin[kt, nt]
        mag_out[ft, nt]  = sqrt(cos_proj² + sin_proj²)

    Loop structure (reader == compute order to keep DFB pipe consistent):
        for ft in range(FT):
            for nt in range(NT):
                for kt in range(KT):    ← K-reduction
                    consume one f_blk, c_blk, s_blk tile triple
    """
    FT = frames_t.shape[0] // TILE   # n_frames tiles
    KT = frames_t.shape[1] // TILE   # n_fft tiles (K-reduction axis)
    NT = cos_t.shape[1] // TILE      # n_freqs_pad tiles

    # L1 dataflow buffers — one tile at a time to minimise L1 pressure
    f_dfb   = ttl.make_dataflow_buffer_like(frames_t, shape=(1, 1))
    cos_dfb = ttl.make_dataflow_buffer_like(cos_t,    shape=(1, 1))
    sin_dfb = ttl.make_dataflow_buffer_like(sin_t,    shape=(1, 1))
    mag_dfb = ttl.make_dataflow_buffer_like(mag_out,  shape=(1, 1))

    @ttl.datamovement()
    def reader():
        # Outer-to-inner: ft, nt, kt — must match compute() exactly.
        for ft in range(FT):
            for nt in range(NT):
                for kt in range(KT):
                    with (
                        f_dfb.reserve()   as f_blk,
                        cos_dfb.reserve() as c_blk,
                        sin_dfb.reserve() as s_blk,
                    ):
                        # Transfer one (TILE × TILE) tile from DRAM to L1.
                        ttl.copy(frames_t[ft, kt], f_blk).wait()
                        ttl.copy(cos_t[kt, nt],   c_blk).wait()
                        ttl.copy(sin_t[kt, nt],   s_blk).wait()

    @ttl.compute()
    def compute():
        for ft in range(FT):
            for nt in range(NT):
                with mag_dfb.reserve() as mag_blk:
                    # Initialise two accumulators for the K-reduction.
                    # fill() sets every element of mag_blk to 0 and returns
                    # a temporary Block; we do NOT store it — we accumulate.
                    c_acc = ttl.math.fill(mag_blk, 0.0)
                    s_acc = ttl.math.fill(mag_blk, 0.0)
                    for kt in range(KT):
                        with (
                            f_dfb.wait()   as f_blk,
                            cos_dfb.wait() as c_blk,
                            sin_dfb.wait() as s_blk,
                        ):
                            # GEMM accumulate: f_blk @ c_blk is (TILE × TILE)
                            c_acc = c_acc + f_blk @ c_blk
                            s_acc = s_acc + f_blk @ s_blk
                    # Note: no epsilon here (unlike NumPy path which uses +1e-8).
                    # Near-zero bins produce exactly 0.0 vs ~1e-4 in NumPy.
                    # Downstream log-mel computation should add its own epsilon.
                    mag_blk.store(
                        ttl.math.sqrt(
                            ttl.math.square(c_acc) + ttl.math.square(s_acc)
                        )
                    )

    @ttl.datamovement()
    def writer():
        for ft in range(FT):
            for nt in range(NT):
                with mag_dfb.wait() as mag_blk:
                    # Write completed magnitude tile back to DRAM
                    ttl.copy(mag_blk, mag_out[ft, nt]).wait()


# ---------------------------------------------------------------------------
# Kernel 2: magnitude × mel filterbank → mel spectrogram
# ---------------------------------------------------------------------------

@ttl.operation(grid=(1, 1))
def _stft_compute_mel(
    mag_t: ttnn.Tensor,   # (n_frames_pad, n_freqs_pad)   — magnitude spectrum
    mel_t: ttnn.Tensor,   # (n_freqs_pad, n_mels)         — mel filterbank
    mel_out: ttnn.Tensor, # (n_frames_pad, n_mels)        — mel spectrogram output
) -> None:
    """
    Second STFT pass: project magnitude into mel filterbank.

    For each output tile (ft, mt):
        mel_out[ft, mt] = sum_nt  mag[ft, nt] @ mel_filter[nt, mt]

    Loop structure (reader == compute order):
        for ft in range(FT):
            for mt in range(MT):
                for nt in range(NT):    ← N-reduction (n_freqs tiles)
                    consume one mag_blk, mel_blk tile pair
    """
    FT = mag_t.shape[0]  // TILE   # n_frames tiles
    NT = mag_t.shape[1]  // TILE   # n_freqs_pad tiles (reduction axis)
    MT = mel_t.shape[1]  // TILE   # n_mels tiles

    mag_dfb = ttl.make_dataflow_buffer_like(mag_t,   shape=(1, 1))
    mel_dfb = ttl.make_dataflow_buffer_like(mel_t,   shape=(1, 1))
    out_dfb = ttl.make_dataflow_buffer_like(mel_out, shape=(1, 1))

    @ttl.datamovement()
    def reader():
        # Outer-to-inner: ft, mt, nt — must match compute() exactly.
        for ft in range(FT):
            for mt in range(MT):
                for nt in range(NT):
                    with (
                        mag_dfb.reserve() as mag_blk,
                        mel_dfb.reserve() as m_blk,
                    ):
                        ttl.copy(mag_t[ft, nt],   mag_blk).wait()
                        ttl.copy(mel_t[nt, mt],   m_blk).wait()

    @ttl.compute()
    def compute():
        for ft in range(FT):
            for mt in range(MT):
                with out_dfb.reserve() as out_blk:
                    acc = ttl.math.fill(out_blk, 0.0)
                    for nt in range(NT):
                        with (
                            mag_dfb.wait() as mag_blk,
                            mel_dfb.wait() as m_blk,
                        ):
                            acc = acc + mag_blk @ m_blk
                    out_blk.store(acc)

    @ttl.datamovement()
    def writer():
        for ft in range(FT):
            for mt in range(MT):
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, mel_out[ft, mt]).wait()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fused_stft_sim(
    audio_chunk: np.ndarray,
    kernel: "TTStftKernel",
    chunk_start_time: float = 0.0,
) -> SpectrogramChunk:
    """
    Run the two-pass TT-Lang simulator STFT kernel on one audio chunk.

    Steps
    -----
    1. Frame audio into overlapping windows and apply the Hann window
       (identical to _process_chunk_numpy so results are comparable).
    2. Pad all tensor dimensions to multiples of TILE=32 (Tensix constraint).
    3. Convert padded NumPy arrays to ttnn.Tensor via torch.
    4. Dispatch ``_stft_compute_mag``: frames × DFT basis → magnitude.
    5. Dispatch ``_stft_compute_mel``: magnitude × mel filter → mel.
    6. Convert outputs back to NumPy and trim the padding rows/columns.
    7. Build frame-centre timestamps and return a SpectrogramChunk.

    Parameters
    ----------
    audio_chunk      : (n_samples,) float32 — raw audio samples
    kernel           : TTStftKernel instance.  Provides:
                         ``_cos_basis``  (n_fft, n_freqs) float32
                         ``_sin_basis``  (n_fft, n_freqs) float32
                         ``_mel_filter`` (n_freqs, n_mels) float32
                         ``_hann``       (n_fft,) float32
                         ``n_fft``, ``hop_length``, ``sr``,
                         ``n_freqs``, ``n_mels``
    chunk_start_time : time in seconds of the first audio sample

    Returns
    -------
    SpectrogramChunk
        ``mag``        : (n_frames, n_freqs) float32 — trimmed magnitude
        ``mel``        : (n_frames, n_mels)  float32 — trimmed mel spectrogram
        ``timestamps`` : (n_frames,)         float32 — frame centre times (s)
    """
    n_fft   = kernel.n_fft
    hop     = kernel.hop_length
    n_freqs = kernel.n_freqs   # 1025 (NOT a multiple of 32)
    n_mels  = kernel.n_mels    # 128  (multiple of 32 ✓)
    sr      = kernel.sr

    audio = np.asarray(audio_chunk, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 1. Frame audio                                                       #
    # ------------------------------------------------------------------ #
    n_samples = len(audio)
    n_frames  = max(0, (n_samples - n_fft) // hop + 1)
    if n_frames == 0:
        # Audio shorter than one STFT frame — return empty chunk
        return SpectrogramChunk(
            mag=np.zeros((0, n_freqs), dtype=np.float32),
            mel=np.zeros((0, n_mels),  dtype=np.float32),
            timestamps=np.zeros(0,     dtype=np.float32),
        )

    # Build frame matrix (n_frames, n_fft) using stride tricks to avoid
    # a full copy; copy() afterwards so in-place Hann multiply is safe.
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    ).copy()
    frames *= kernel._hann  # apply Hann window in-place

    # ------------------------------------------------------------------ #
    # 2. Pad all dimensions to multiples of TILE=32                       #
    # ------------------------------------------------------------------ #
    n_frames_pad = _pad_to_tile(n_frames)  # pad n_frames (variable)
    n_freqs_pad  = _pad_to_tile(n_freqs)   # pad 1025 → 1056
    # n_fft=2048 and n_mels=128 are already tile-aligned ✓

    # Frames: (n_frames_pad, n_fft) — zero-pad extra rows
    frames_pad = np.zeros((n_frames_pad, n_fft), dtype=np.float32)
    frames_pad[:n_frames] = frames

    # Cosine DFT basis: (n_fft, n_freqs_pad) — zero-pad extra columns
    cos_pad = np.zeros((n_fft, n_freqs_pad), dtype=np.float32)
    cos_pad[:, :n_freqs] = kernel._cos_basis

    # Sine DFT basis: (n_fft, n_freqs_pad) — zero-pad extra columns
    sin_pad = np.zeros((n_fft, n_freqs_pad), dtype=np.float32)
    sin_pad[:, :n_freqs] = kernel._sin_basis

    # Mel filterbank: (n_freqs_pad, n_mels) — zero-pad extra rows
    mel_pad = np.zeros((n_freqs_pad, n_mels), dtype=np.float32)
    mel_pad[:n_freqs, :] = kernel._mel_filter

    # ------------------------------------------------------------------ #
    # 3. Convert to ttnn tensors                                          #
    # ------------------------------------------------------------------ #
    frames_tt = ttnn.from_torch(torch.from_numpy(frames_pad))
    cos_tt    = ttnn.from_torch(torch.from_numpy(cos_pad))
    sin_tt    = ttnn.from_torch(torch.from_numpy(sin_pad))
    mel_tt    = ttnn.from_torch(torch.from_numpy(mel_pad))

    # Pre-allocate output tensors (uninitialised DRAM buffers)
    mag_tt  = ttnn.empty((n_frames_pad, n_freqs_pad), dtype=torch.float32)
    mel_out = ttnn.empty((n_frames_pad, n_mels),       dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # 4 & 5. Dispatch kernels (synchronous in simulator mode)             #
    # ------------------------------------------------------------------ #
    _stft_compute_mag(frames_tt, cos_tt, sin_tt, mag_tt)
    _stft_compute_mel(mag_tt, mel_tt, mel_out)

    # ------------------------------------------------------------------ #
    # 6. Convert back and trim padding                                     #
    # ------------------------------------------------------------------ #
    mag_np = ttnn.to_torch(mag_tt).numpy()[:n_frames, :n_freqs].astype(np.float32)
    mel_np = ttnn.to_torch(mel_out).numpy()[:n_frames, :].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 7. Timestamps — identical formula to _process_chunk_numpy           #
    # ------------------------------------------------------------------ #
    frame_centers = (np.arange(n_frames) * hop + n_fft // 2) / sr
    timestamps = (frame_centers + chunk_start_time).astype(np.float32)

    return SpectrogramChunk(mag=mag_np, mel=mel_np, timestamps=timestamps)
