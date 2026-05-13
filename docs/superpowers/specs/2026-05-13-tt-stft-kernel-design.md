# TT-Lang Fused STFT Kernel Design

**Date:** 2026-05-13
**Hardware:** 2x P300C cards = 4 Blackhole chips (P150X4 mesh)
**TT-Lang env:** `~/code/tt-lang` (ttlang-sim 1.0.0.dev14 for testing)
**JAX env:** `~/p300c-xla-test` (Python 3.12, JAX 0.6.0, PJRT TT plugin)

## Goal

Replace the DFT-via-matmul approach in `jax_feature_extraction.py` with a TT-Lang fused
STFT kernel that keeps all intermediate tensors in Tensix L1 SRAM, writing only the final
mel spectrogram to DRAM. Implement streaming (30s chunks) to eliminate the OOM crash seen
with long files (avg 7.5 min).

## Why TT-Lang over JAX PJRT

The JAX PJRT path writes every intermediate tensor (windowed frames, cos/sin projections,
magnitude spectrum) to DRAM and reads it back. TT-Lang `compute` blocks run on Tensix cores
with explicit L1 SRAM control, fusing the entire pipeline into one kernel dispatch with no
intermediate DRAM traffic.

Known JAX PJRT limitations on Blackhole that this also sidesteps:
- `jnp.fft.rfft` fails (complex tensor materialization)
- Large batches OOM (31 GB attempted for 8 files × 7.5 min)

## Streaming Design

Process audio in 30-second chunks with 2-second overlap (4096 samples at sr=22050) to
handle seamless chunk boundaries. The caller iterates; the kernel is stateless per chunk.

```
audio file (any length)
  ├── chunk 0: samples [0 .. 30s+2s]
  ├── chunk 1: samples [30s .. 60s+2s]
  └── ...
        │
        ▼ TTStftKernel.process_chunk()
        SpectrogramChunk(mag, mel, timestamps)
        │
        ▼ caller stitches chunks, discards overlap frames
```

## Python Interface

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SpectrogramChunk:
    mag: np.ndarray        # (n_frames, n_fft//2+1) float32 — magnitude spectrum
    mel: np.ndarray        # (n_frames, n_mels)      float32 — mel spectrogram
    timestamps: np.ndarray # (n_frames,)             float32 — center time of each frame (s)

class TTStftKernel:
    def __init__(self,
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 chunk_seconds: float = 30.0,
                 overlap_seconds: float = 2.0):
        # Precompute static matrices: cos_basis (n_fft, n_freqs), sin_basis (n_fft, n_freqs),
        # mel_filterbank (n_freqs, n_mels), hann_window (n_fft,).
        # Compile TT-Lang kernel once.

    def process_chunk(self,
                      audio_chunk: np.ndarray,  # (n_samples,) float32
                      chunk_start_time: float = 0.0
                      ) -> SpectrogramChunk:
        # Dispatch to TT hardware, return chunk result.

    def process_file(self,
                     audio: np.ndarray,          # (n_samples,) float32
                     sr: int
                     ) -> SpectrogramChunk:
        # Convenience: iterate chunks, stitch, return full-file SpectrogramChunk.
```

`process_file` is the primary entry point for the analysis pipeline. `process_chunk` is
exposed for streaming use cases (live audio, very large files).

## TT-Lang Kernel: Fused STFT Pipeline

### File: `tt_stft_kernel.ttl`

The kernel operates on one chunk at a time. All steps run in a single `compute` block,
keeping intermediate results in L1 SRAM.

**Static inputs (loaded once at init, treated as constants):**
```
cos_basis    : (n_fft, n_freqs)  float32  — cosine DFT basis
sin_basis    : (n_fft, n_freqs)  float32  — sine DFT basis
mel_filter   : (n_freqs, n_mels) float32  — triangular mel filterbank
hann_window  : (n_fft,)          float32  — Hann window coefficients
```

**Per-chunk input:**
```
frames       : (n_frames, n_fft) float32  — strided slices of audio chunk
```

Framing (strided slicing with hop=512) happens in the Python wrapper before dispatch
since it requires irregular memory access that is more efficient on the CPU side.

**Fused compute in L1 (per tile of frames):**
```
1. Apply window:  frames = frames * hann_window           shape: (n_frames, n_fft)
2. Cos project:   cos_proj = frames @ cos_basis           shape: (n_frames, n_freqs) — L1
3. Sin project:   sin_proj = frames @ sin_basis           shape: (n_frames, n_freqs) — L1
4. Magnitude:     mag = sqrt(cos_proj^2 + sin_proj^2)    shape: (n_frames, n_freqs) — L1
5. Mel project:   mel = mag @ mel_filter                  shape: (n_frames, n_mels)  — DRAM out
   + pass mag to DRAM output                              shape: (n_frames, n_freqs) — DRAM out
```

Steps 1–5 are fused. Steps 2 and 3 can run in parallel on separate Tensix cores
(independent matmuls over disjoint output tiles).

### Why matmul-DFT, not radix-2 FFT

Tensix cores are optimised for matrix multiply. A radix-2 FFT requires irregular memory
access (bit-reversal permutation, stride-2 reads) that maps poorly to the Tensix grid.
The DFT matmul runs at full Tensix FLOP rate with the cos/sin matrices resident in L1.
For n_fft=2048, the L1 footprint of both basis matrices is 2 × 2048 × 1025 × 4 bytes ≈ 16 MB,
which fits in Blackhole L1 SRAM (128 MB per chip).

## Filter Precomputation (Python, done once)

```python
import numpy as np
import librosa

def build_stft_constants(sr, n_fft, n_mels):
    n_freqs = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft) * n_fft  # frequency indices 0..n_fft//2
    cos_basis = np.cos(2 * np.pi * np.outer(np.arange(n_fft), freqs / n_fft)).astype(np.float32)
    sin_basis = np.sin(2 * np.pi * np.outer(np.arange(n_fft), freqs / n_fft)).astype(np.float32)
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).T.astype(np.float32)
    hann = np.hanning(n_fft).astype(np.float32)
    return cos_basis, sin_basis, mel_filter, hann
```

## Integration with JAX Feature Extraction

`jax_feature_extraction.py` currently builds cos/sin matrices and runs them through JAX
vmap. After this change:

1. `TTStftKernel.process_file()` replaces the JAX DFT-via-matmul for the spectrogram.
2. The resulting `SpectrogramChunk.mag` and `SpectrogramChunk.mel` are passed directly
   to the downstream feature extraction (MFCC, chroma, spectral stats) which remain as
   JAX matmul ops (they're small matmuls that work fine on TT PJRT).
3. The filter cache in `jax_feature_extraction.py` is reused — same matrices, different
   dispatch path.

When TT hardware is not available, `TTStftKernel` falls back to running the same
computation with NumPy (no TT-Lang dispatch). The `SpectrogramChunk` interface is
identical either way.

## Fallback Chain

```
TT hardware + TT-Lang kernel compiled successfully
  → fused STFT on TT hardware, streaming chunks      [primary]

TT hardware unavailable OR kernel compilation fails
  → NumPy DFT-via-matmul, same chunk structure       [warn + continue]

NumPy unavailable (shouldn't happen)
  → librosa stft fallback                            [emergency]
```

## Files

| File | Change |
|------|--------|
| `audio_analysis/core/tt_stft_kernel.py` | New — Python wrapper, streaming loop, fallback |
| `tt_stft_kernel.ttl` | New — TT-Lang DSL kernel (same directory) |
| `audio_analysis/core/jax_feature_extraction.py` | Modify — use TTStftKernel instead of JAX DFT matmul |
| `tests/test_tt_stft_kernel.py` | New — unit + parity tests |

## Testing Plan

1. **Parity test**: `TTStftKernel.process_file()` vs `librosa.stft()` on a 440 Hz sine.
   Assert mel spectrogram peak bin within 5%.

2. **Streaming continuity test**: process a 90s file as 3 × 30s chunks with overlap.
   Assert the stitched result matches single-pass result at chunk boundaries (within 1%).

3. **OOM regression**: process all 8 files from `~/samples` (avg 7.5 min). Assert no crash.

4. **TT hardware test**: with PJRT plugin active, assert `TTStftKernel` uses TT device,
   run single-file extraction end-to-end.

5. **Integration**: `JaxAudioFeatureExtractor.extract_batch()` produces same feature keys
   with kernel-backed spectrogram as with matmul-backed spectrogram.
