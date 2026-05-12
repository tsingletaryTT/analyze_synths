# JAX/TT Feature Extraction Design

**Date:** 2026-05-12
**Hardware:** 2x P300C cards = 4 Blackhole chips (P150X4 mesh)
**JAX env:** `~/p300c-xla-test` (Python 3.12, JAX 0.6.0, PJRT TT plugin)

## Goal

Replace the `TenstorrentTensorProcessor` and `_extract_features_vectorized` stubs with a
real JAX/XLA implementation that runs the full audio feature extraction pipeline natively on
Tenstorrent Blackhole hardware. Hardware is selected transparently — no user action required.

## Files Changed

| File | Change |
|------|--------|
| `audio_analysis/core/jax_feature_extraction.py` | New — all JAX compute |
| `audio_analysis/core/tensor_operations.py` | Fill in `TenstorrentTensorProcessor` stub |
| `audio_analysis/core/parallel_feature_extraction.py` | Replace `_extract_features_vectorized` stub |
| `audio_analysis/__init__.py` | Auto-detect TT device at import |

Nothing else changes. CLI flags, API signatures, export formats, mood/character/phase analysis
are all untouched.

## New File: `jax_feature_extraction.py`

### Responsibilities

- Precompute static filter matrices (mel filterbank, chroma projection, DCT) once per
  `(sr, n_fft)` combination and cache them.
- Implement the full feature extraction pipeline as a single `jax.jit`-compiled function
  vmapped over the batch dimension.
- Implement JAX-native beat tracking via `jax.lax.scan` so tempo runs on TT hardware
  with no CPU round-trip.
- Expose one public entry point: `JaxAudioFeatureExtractor.extract_batch()`.

### Filter Precomputation

Filters are static arrays baked into JIT-compiled functions. Computed once, never reloaded.

```
mel_filterbank   : (n_fft//2+1, n_mels)   float32  — triangular mel filters
dct_matrix       : (n_mels, n_mfcc)        float32  — type-II DCT
chroma_filter    : (n_fft//2+1, 12)        float32  — chroma bin projection
freq_weights     : (n_fft//2+1,)           float32  — Hz values for centroid/rolloff
```

Cache key is `(sr, n_fft, n_mels, n_mfcc)`. Stored as a module-level dict so repeated
calls to `extract_batch` with the same parameters skip recomputation.

### Pipeline (vmapped over batch dimension B)

```
Input: audio_batch (B, max_len) float32, lengths (B,) int32, sr int

1. Frame + window
   frames = strided_slice(audio, hop=512, n_fft=2048) + hann_window
   shape: (B, n_frames, n_fft)

2. FFT
   spec = jnp.abs(jnp.fft.rfft(frames, axis=-1))
   shape: (B, n_frames, n_fft//2+1)

3. Spectral features  [from spec, reductions over freq/frame dims]
   spectral_centroid   = sum(freq_weights * spec) / sum(spec)   per frame, then mean/std
   spectral_rolloff    = freq where cumsum(spec) >= 0.85 * total  per frame, then mean
   spectral_bandwidth  = sqrt(sum((freq - centroid)^2 * spec) / sum(spec))  per frame, mean
   spectral_roughness  = mean(abs(diff(spec, axis=-1)))  — spectral irregularity proxy

4. MFCC
   mel = jnp.dot(spec, mel_filterbank)          shape: (B, n_frames, n_mels)
   log_mel = jnp.log(mel + 1e-6)
   mfcc = jnp.dot(log_mel, dct_matrix)          shape: (B, n_frames, n_mfcc)
   output: mean and std per coefficient -> 2*n_mfcc values

5. Chroma
   chroma = jnp.dot(spec, chroma_filter)         shape: (B, n_frames, 12)
   chroma = chroma / (jnp.sum(chroma, axis=-1, keepdims=True) + 1e-6)
   chroma_mean = jnp.mean(chroma, axis=1)        shape: (B, 12)
   key_index   = jnp.argmax(chroma_mean, axis=-1)
   key_confidence = jnp.max(chroma_mean, axis=-1)

6. Temporal features  [from frames before FFT]
   rms = jnp.sqrt(jnp.mean(frames**2, axis=-1))    per frame -> mean/std
   zcr = jnp.mean(jnp.abs(jnp.diff(jnp.sign(frames), axis=-1)), axis=-1)  per frame -> mean

7. Tonnetz  [from chroma, linear transform — static 6x12 matrix]
   tonnetz = jnp.dot(chroma_mean, tonnetz_transform)   shape: (B, 6)

8. Tempo / beat tracking  [lax.scan DP on onset envelope]
   a. onset_env = jnp.mean(jnp.diff(spec, axis=1).clip(0), axis=-1)  shape: (B, n_frames-1)
   b. autocorr  = real(ifft(abs(fft(onset_env))**2))  shape: (B, n_frames-1)
      restrict to lag range corresponding to 30-300 BPM at given sr/hop
   c. DP refinement via lax.scan: forward pass accumulates best-path score
      over the autocorrelation trellis (same algorithm as librosa's beat_track
      but expressed as a scan over frames)
   d. dominant_period = argmax of DP score -> BPM

Output: feature dict with same keys as feature_extraction_base.py
```

### Class Interface

```python
class JaxAudioFeatureExtractor:
    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512,
                 n_mels: int = 128, n_mfcc: int = 13, n_chroma: int = 12):
        # precompute filters, jit-compile pipeline

    def extract_batch(self,
                      audio_batch: np.ndarray,   # (B, max_len) float32
                      lengths: np.ndarray,        # (B,) int32
                      sr: int,
                      file_paths: List[Path]) -> List[Dict[str, Any]]:
        # dispatch to TT, return list of feature dicts
```

`extract_batch` is the only public method. It accepts numpy arrays (from the existing
librosa loading path), moves them to TT via `jax.device_put`, runs the jit-compiled
pipeline, pulls results back to numpy, and returns a list of feature dicts.

## Auto-Detection

`audio_analysis/__init__.py` gains a `_detect_tt_device()` function called once at import:

```python
def _detect_tt_device() -> str:
    try:
        import jax
        # Activate PJRT plugin (sets env vars if not already set)
        _activate_tt_pjrt()
        devices = jax.devices()
        if any(d.platform == 'tt' for d in devices):
            logger.info(f"Tenstorrent hardware detected: {len(devices)} device(s)")
            return 'tenstorrent'
    except Exception as e:
        logger.debug(f"TT device detection failed: {e}")
    return 'cpu'
```

`_activate_tt_pjrt()` sets:
```
PJRT_PLUGIN_LIBRARY_PATH=/home/ttuser/tt-xla/build/lib/libpjrt_tt.so
```
only if not already set, so it doesn't interfere with environments that configure it
externally.

`ProcessingConfig.device` field is added, defaulting to `_detect_tt_device()` result.
`ProcessingConfig.tt_venv_python` points to `~/p300c-xla-test/bin/python` for reference.

## Modified: `TenstorrentTensorProcessor`

Replace the two `logger.info / fallback to CPU` stub methods:

```python
def compute_features(self, audio_tensor, lengths, sample_rates, file_paths):
    extractor = JaxAudioFeatureExtractor()
    # All files in a TT batch are loaded at ProcessingConfig.sample_rate (22050),
    # so sample_rates is uniform. bincount handles the edge case of mixed rates.
    sr = int(np.bincount(sample_rates).argmax())
    return extractor.extract_batch(audio_tensor, lengths, sr, file_paths)

def cluster_features(self, features, n_clusters):
    # JAX Lloyd's algorithm: vmap distance, lax.while_loop convergence
    return _jax_kmeans(features, n_clusters)
```

When `device == 'tenstorrent'`, `ProcessingConfig.sample_rate` defaults to `22050` so
librosa resamples every file to that rate on load, keeping filter matrices consistent.

`_jax_kmeans(features, n_clusters)` is a module-level jit-compiled function:
- Initialize centroids with k-means++ (on CPU, one-time)
- `lax.while_loop`: each iteration does `vmap(l2_distance)` → assignment → centroid update
- Convergence: max centroid movement < 1e-4 or 100 iterations
- Returns `(labels, centers)` as numpy arrays matching sklearn KMeans output shape

## Modified: `_extract_features_vectorized`

Current stub loops over individual files calling `_extract_single_file_features`.
Replace with:

```python
def _extract_features_vectorized(self, batch, tensor_data):
    if self.config.device == 'tenstorrent':
        processor = TenstorrentTensorProcessor()
        return processor.compute_features(
            tensor_data['audio_tensor'],
            tensor_data['lengths'],
            tensor_data['sample_rates'],
            batch.file_paths
        )
    # existing CPU fallback path unchanged
```

## Fallback Chain

```
TT hardware detected + PJRT plugin loads
  -> JAX pipeline on TT hardware                  [primary path]

TT hardware detected but PJRT init fails
  -> JAX pipeline on CPU (jit still compiles)     [warn + continue]
  -> log: "TT device unavailable, using CPU JAX"

JAX not importable (wrong venv)
  -> existing librosa/numpy path unchanged        [silent fallback]
  -> log: "JAX not available, using librosa path"
```

## Feature Key Compatibility

The JAX extractor produces identical dict keys to `feature_extraction_base.py`. Spot-check
list (non-exhaustive):

| Key | Source in JAX pipeline |
|-----|------------------------|
| `spectral_centroid_mean` | step 3, frame mean |
| `spectral_rolloff_mean` | step 3, frame mean |
| `spectral_bandwidth_mean` | step 3, frame mean |
| `spectral_roughness` | step 3 |
| `mfcc_1_mean` ... `mfcc_13_mean` | step 4 |
| `mfcc_1_std` ... `mfcc_13_std` | step 4 |
| `chroma_C_mean` ... `chroma_B_mean` | step 5 |
| `key` | step 5, argmax index mapped to note name |
| `key_confidence` | step 5 |
| `rms_mean`, `rms_std` | step 6 |
| `zero_crossing_rate_mean` | step 6 |
| `tonnetz_1_mean` ... `tonnetz_6_mean` | step 7 |
| `tempo` | step 8 |

Keys produced by `feature_extraction_base.py` but NOT in JAX pipeline:
- `beat_count` (librosa beat_track returns beat frames; JAX DP gives tempo only)
  -> set to `0` with a note in the feature dict; does not affect mood/character thresholds

## Testing Plan

1. **Unit test** `JaxAudioFeatureExtractor.extract_batch()` against
   `feature_extraction_base.extract_features_from_audio()` on the same file.
   Assert all shared keys agree within 5% relative tolerance (float precision differences
   between JAX/librosa implementations are expected).

2. **Integration test** `ParallelAudioAnalyzer` on a small directory (3-5 WAV files).
   Assert output CSV has same columns as the CPU path.

3. **Hardware smoke test** — run with `PJRT_PLUGIN_LIBRARY_PATH` set and verify
   `jax.devices()` returns TT devices, then run a single-file extraction end to end.

## Phase B (out of scope here)

Once this JAX path is working and profiled, the STFT → MFCC sub-pipeline (steps 1-4
above) is the natural candidate for a fused TT-Lang kernel. The `ttl-import` skill can
translate the JAX framing + FFT + filterbank code to TT-Lang DSL, eliminating the
intermediate DRAM writes between steps.
