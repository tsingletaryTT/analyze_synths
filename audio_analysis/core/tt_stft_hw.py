"""
JAX PJRT hardware dispatch for the fused STFT kernel.

Uses JAX with the TT PJRT backend (Blackhole) to run the two DFT matmuls
(cos/sin projection) and mel filterbank projection on Tensix hardware.

Environment setup (done once at import):
- Adds /home/ttuser/p300c-xla-test/lib/python3.12/site-packages to sys.path
- Activates the pjrt_plugin_tt editable install finder
- Sets PJRT_PLUGIN_LIBRARY_PATH if not already set
- Switches JAX default platform to 'tt'

Dispatch design:
- Basis matrices (cos, sin, mel) are pre-transferred to JAX device arrays
  at the first call to fused_stft_hw() for a given kernel instance.
  Subsequent calls for the same kernel reuse the device-resident arrays.
- Frames are padded to a fixed size (n_frames_pad = ceil(max_frames/32)*32)
  so JIT compiles exactly once per kernel configuration.
- The standard max_frames for a 32s chunk at sr=22050, hop=512:
    (32 * 22050 - 2048) // 512 + 1 = 1375 → padded to 1376
"""
from __future__ import annotations

import functools
import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time environment setup (runs at import)
# ---------------------------------------------------------------------------
_P300C_SITE = "/home/ttuser/p300c-xla-test/lib/python3.12/site-packages"
_PJRT_SO = "/home/ttuser/tt-xla/python_package/pjrt_plugin_tt/pjrt_plugin_tt.so"

if _P300C_SITE not in sys.path:
    sys.path.insert(0, _P300C_SITE)

try:
    import __editable___pjrt_plugin_tt_0_1_260226_dev_c67d612a_finder as _finder
    _finder.install()
except ImportError:
    pass  # already installed or not present

os.environ.setdefault("PJRT_PLUGIN_LIBRARY_PATH", _PJRT_SO)

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "tt")

if TYPE_CHECKING:
    from audio_analysis.core.tt_stft_kernel import TTStftKernel, SpectrogramChunk

# ---------------------------------------------------------------------------
# Cache: per-kernel basis matrices already on TT device
# ---------------------------------------------------------------------------
# Maps id(kernel) → (jax_cos, jax_sin, jax_mel)
_DEVICE_BASES: dict = {}

# Standard padded n_frames for a full 30s+2s chunk at sr=22050, hop=512:
#   n_samples = (30 + 2) * 22050 = 705600
#   n_frames  = (705600 - 2048) // 512 + 1 = 1375
#   padded    = ceil(1375 / 32) * 32       = 1376
_STANDARD_N_FRAMES_PAD = 1376

# ---------------------------------------------------------------------------
# JIT-compiled STFT function (compiled once per unique shape triple)
# ---------------------------------------------------------------------------

@jax.jit
def _stft_jit(frames_j, cos_j, sin_j, mel_j):
    """
    Fused DFT matmul + mel filterbank, JIT-compiled for TT hardware.

    frames_j : (n_frames_pad, n_fft)         float32
    cos_j    : (n_fft, n_freqs_pad)          float32
    sin_j    : (n_fft, n_freqs_pad)          float32
    mel_j    : (n_freqs_pad, n_mels)         float32

    Returns
    -------
    mag_j : (n_frames_pad, n_freqs_pad) float32
    mel_o : (n_frames_pad, n_mels)      float32
    """
    cos_proj = jnp.matmul(frames_j, cos_j)
    sin_proj = jnp.matmul(frames_j, sin_j)
    mag_j    = jnp.sqrt(cos_proj * cos_proj + sin_proj * sin_proj)
    mel_o    = jnp.matmul(mag_j, mel_j)
    return mag_j, mel_o


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if JAX with a TT backend is importable and has devices."""
    try:
        devs = jax.devices("tt")
        return len(devs) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fused_stft_hw(
    audio_chunk: np.ndarray,
    kernel: "TTStftKernel",
    chunk_start_time: float = 0.0,
) -> "SpectrogramChunk":
    """
    Run the fused STFT kernel on TT hardware via JAX PJRT.

    Steps
    -----
    1. Frame audio and apply Hann window (CPU — irregular memory access).
    2. Pad frames to _STANDARD_N_FRAMES_PAD × n_fft (or actual padded size if
       the chunk is longer than the standard 32s).
    3. Transfer padded frames to JAX TT device; use cached device-resident
       basis matrices (transferred once per kernel instance).
    4. Dispatch JIT-compiled matmul chain: frames × cos/sin → mag → mel.
    5. Block until ready, pull back to NumPy, trim padding, return SpectrogramChunk.

    Parameters
    ----------
    audio_chunk      : (n_samples,) float32 — raw audio samples
    kernel           : TTStftKernel instance with precomputed basis matrices.
                       Provides _cos_basis, _sin_basis, _mel_filter, _hann,
                       n_fft, hop_length, sr, n_freqs, n_mels.
    chunk_start_time : time in seconds of the first sample in this chunk

    Returns
    -------
    SpectrogramChunk
        mag        : (n_frames, n_freqs) float32 — magnitude spectrum
        mel        : (n_frames, n_mels)  float32 — mel spectrogram
        timestamps : (n_frames,)         float32 — frame centre times (s)
    """
    from audio_analysis.core.tt_stft_kernel import SpectrogramChunk, TILE, _pad_to_tile

    n_fft   = kernel.n_fft
    hop     = kernel.hop_length
    n_freqs = kernel.n_freqs   # 1025
    n_mels  = kernel.n_mels    # 128
    sr      = kernel.sr

    audio = np.asarray(audio_chunk, dtype=np.float32)
    n_samples = len(audio)
    n_frames  = max(0, (n_samples - n_fft) // hop + 1)

    if n_frames == 0:
        return SpectrogramChunk(
            mag=np.zeros((0, n_freqs), dtype=np.float32),
            mel=np.zeros((0, n_mels),  dtype=np.float32),
            timestamps=np.zeros(0,     dtype=np.float32),
        )

    # ------ Frame + window (CPU) ------
    # Using stride tricks to create frame view, then copy for safe in-place window.
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    ).copy()
    frames *= kernel._hann

    # ------ Pad frames to tile-aligned shape ------
    # Use _STANDARD_N_FRAMES_PAD for normal chunks so JIT sees only one shape.
    # If the chunk is unusually long (edge case), round up to the next tile boundary.
    n_frames_pad = max(_STANDARD_N_FRAMES_PAD, _pad_to_tile(n_frames))
    n_freqs_pad  = _pad_to_tile(n_freqs)  # 1025 → 1056

    frames_pad = np.zeros((n_frames_pad, n_fft), dtype=np.float32)
    frames_pad[:n_frames] = frames

    # ------ Retrieve or build cached device-resident basis matrices ------
    # Basis matrices are transferred to TT device memory exactly once per
    # TTStftKernel instance.  The cache uses id(kernel) as the key; this is
    # safe because TTStftKernel instances are long-lived (created once per
    # analysis run) and basis matrices are immutable after __init__.
    kid = id(kernel)
    if kid not in _DEVICE_BASES:
        log.info("Transferring STFT basis matrices to TT device (once per kernel)")
        cos_pad = np.zeros((n_fft, n_freqs_pad), dtype=np.float32)
        cos_pad[:, :n_freqs] = kernel._cos_basis
        sin_pad = np.zeros((n_fft, n_freqs_pad), dtype=np.float32)
        sin_pad[:, :n_freqs] = kernel._sin_basis
        mel_pad = np.zeros((n_freqs_pad, n_mels), dtype=np.float32)
        mel_pad[:n_freqs, :] = kernel._mel_filter

        _DEVICE_BASES[kid] = (
            jnp.array(cos_pad),
            jnp.array(sin_pad),
            jnp.array(mel_pad),
        )

    cos_j, sin_j, mel_j = _DEVICE_BASES[kid]

    # Transfer frames to TT device for this chunk (basis matrices already resident)
    frames_j = jnp.array(frames_pad)

    # ------ Dispatch to hardware (JIT-compiled, cached after first call) ------
    mag_j, mel_o_j = _stft_jit(frames_j, cos_j, sin_j, mel_j)

    # ------ Pull results back to NumPy and trim padding ------
    # np.array() blocks until the TT device computation completes.
    mag_np = np.array(mag_j)[:n_frames, :n_freqs].astype(np.float32)
    mel_np = np.array(mel_o_j)[:n_frames, :].astype(np.float32)

    # ------ Timestamps: centre of each frame in full-file coordinates ------
    frame_centers = (np.arange(n_frames) * hop + n_fft // 2) / sr
    timestamps = (frame_centers + chunk_start_time).astype(np.float32)

    return SpectrogramChunk(mag=mag_np, mel=mel_np, timestamps=timestamps)
