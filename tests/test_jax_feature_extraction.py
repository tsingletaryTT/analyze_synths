import pytest
import numpy as np
from pathlib import Path


def _make_sine_wave(freq_hz: float = 440.0, duration_s: float = 3.0,
                    sr: int = 22050) -> np.ndarray:
    """Returns a simple sine wave as float32 numpy array."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_jax_extractor_importable():
    """JaxAudioFeatureExtractor can be imported without error."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor
    extractor = JaxAudioFeatureExtractor(sr=22050)
    assert extractor is not None


def test_filter_shapes():
    """Precomputed filter matrices have correct shapes."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor
    ex = JaxAudioFeatureExtractor(sr=22050, n_fft=2048, n_mels=128, n_mfcc=13)
    n_freqs = 2048 // 2 + 1  # 1025
    assert ex.dft_cos.shape == (n_freqs, 2048), f"dft_cos shape wrong: {ex.dft_cos.shape}"
    assert ex.dft_sin.shape == (n_freqs, 2048), f"dft_sin shape wrong: {ex.dft_sin.shape}"
    assert ex.mel_filterbank.shape == (n_freqs, 128), f"mel shape wrong: {ex.mel_filterbank.shape}"
    assert ex.dct_matrix.shape == (128, 13), f"dct shape wrong: {ex.dct_matrix.shape}"
    assert ex.chroma_filter.shape == (n_freqs, 12), f"chroma shape wrong: {ex.chroma_filter.shape}"
    assert ex.tonnetz_transform.shape == (12, 6), f"tonnetz shape wrong: {ex.tonnetz_transform.shape}"
    assert ex.freq_hz.shape == (n_freqs,), f"freq_hz shape wrong: {ex.freq_hz.shape}"


def test_device_detection_returns_string():
    """_detect_tt_device returns 'tenstorrent' or 'cpu'."""
    from audio_analysis import _detect_tt_device
    result = _detect_tt_device()
    assert result in ('tenstorrent', 'cpu'), f"Unexpected device: {result}"


def test_processing_config_has_device():
    """ProcessingConfig exposes a device field."""
    from audio_analysis.core.parallel_feature_extraction import ProcessingConfig
    config = ProcessingConfig()
    assert hasattr(config, 'device')
    assert config.device in ('tenstorrent', 'cpu')


def test_processing_config_tt_forces_sample_rate():
    """When device='tenstorrent', sample_rate is set to 22050."""
    from audio_analysis.core.parallel_feature_extraction import ProcessingConfig
    config = ProcessingConfig(device='tenstorrent')
    assert config.sample_rate == 22050


def test_extract_batch_returns_correct_keys():
    """extract_batch produces all expected feature keys."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor

    sr = 22050
    audio = _make_sine_wave(440.0, 3.0, sr)
    batch = audio[np.newaxis, :]
    lengths = np.array([len(audio)], dtype=np.int32)

    ex = JaxAudioFeatureExtractor(sr=sr)
    features = ex.extract_batch(batch, lengths, sr, [Path('test.wav')])

    assert len(features) == 1
    f = features[0]

    required_keys = [
        'filename', 'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_rolloff_mean', 'spectral_bandwidth_mean', 'zero_crossing_rate_mean',
        'rms_mean', 'rms_std',
        'mfcc_1_mean', 'mfcc_1_std', 'mfcc_13_mean', 'mfcc_13_std',
        'chroma_C_mean', 'chroma_B_mean',
        'detected_key', 'key_confidence',
        'tonnetz_1_mean', 'tonnetz_6_mean',
    ]
    for key in required_keys:
        assert key in f, f"Missing feature key: {key}"


def test_extract_batch_values_are_finite():
    """All numeric feature values are finite (no NaN or Inf)."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor

    sr = 22050
    audio = _make_sine_wave(440.0, 3.0, sr)
    batch = audio[np.newaxis, :]
    lengths = np.array([len(audio)], dtype=np.int32)

    ex = JaxAudioFeatureExtractor(sr=sr)
    features = ex.extract_batch(batch, lengths, sr, [Path('test.wav')])[0]

    for k, v in features.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Non-finite value for key '{k}': {v}"


def test_extract_batch_spectral_centroid_plausible():
    """Spectral centroid of a 440 Hz sine wave is close to 440 Hz."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor

    sr = 22050
    audio = _make_sine_wave(440.0, 3.0, sr)
    batch = audio[np.newaxis, :]
    lengths = np.array([len(audio)], dtype=np.int32)

    ex = JaxAudioFeatureExtractor(sr=sr)
    features = ex.extract_batch(batch, lengths, sr, [Path('test.wav')])[0]

    centroid = features['spectral_centroid_mean']
    assert abs(centroid - 440.0) / 440.0 < 0.15, (
        f"Spectral centroid {centroid:.1f} Hz too far from 440 Hz"
    )


def test_extract_batch_multiple_files():
    """extract_batch handles B > 1 correctly."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor

    sr = 22050
    a1 = _make_sine_wave(440.0, 2.0, sr)
    a2 = _make_sine_wave(880.0, 3.0, sr)
    max_len = max(len(a1), len(a2))
    batch = np.zeros((2, max_len), dtype=np.float32)
    batch[0, :len(a1)] = a1
    batch[1, :len(a2)] = a2
    lengths = np.array([len(a1), len(a2)], dtype=np.int32)

    ex = JaxAudioFeatureExtractor(sr=sr)
    features = ex.extract_batch(batch, lengths, sr,
                                [Path('a.wav'), Path('b.wav')])

    assert len(features) == 2
    assert features[0]['filename'] == 'a.wav'
    assert features[1]['filename'] == 'b.wav'
    assert features[1]['spectral_centroid_mean'] > features[0]['spectral_centroid_mean'], (
        "880 Hz sine should have higher centroid than 440 Hz"
    )


def test_extract_batch_has_tempo_and_onset_density():
    """extract_batch includes tempo and onset_density from CPU librosa path."""
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor

    sr = 22050
    audio = _make_sine_wave(440.0, 4.0, sr)
    batch = audio[np.newaxis, :]
    lengths = np.array([len(audio)], dtype=np.int32)

    ex = JaxAudioFeatureExtractor(sr=sr)
    features = ex.extract_batch(batch, lengths, sr, [Path('test.wav')])[0]

    assert 'tempo' in features, "Missing 'tempo' key"
    assert 'onset_density' in features, "Missing 'onset_density' key"
    assert isinstance(features['tempo'], float)
    assert features['tempo'] >= 0.0
    assert features['beat_count'] == 0


def test_jax_kmeans_output_shapes():
    """jax_kmeans returns labels (n,) and centers (k, d)."""
    from audio_analysis.core.jax_feature_extraction import jax_kmeans

    rng = np.random.default_rng(42)
    features = rng.standard_normal((20, 10)).astype(np.float32)
    labels, centers = jax_kmeans(features, n_clusters=3)

    assert labels.shape == (20,), f"labels shape wrong: {labels.shape}"
    assert centers.shape == (3, 10), f"centers shape wrong: {centers.shape}"
    assert set(labels.tolist()).issubset({0, 1, 2}), f"unexpected label values: {set(labels.tolist())}"


def test_jax_kmeans_separates_clusters():
    """jax_kmeans correctly separates clearly distinct clusters."""
    from audio_analysis.core.jax_feature_extraction import jax_kmeans

    a = np.zeros((10, 2), dtype=np.float32)
    b = np.ones((10, 2), dtype=np.float32) * 10.0
    features = np.vstack([a, b])
    labels, centers = jax_kmeans(features, n_clusters=2)

    assert len(set(labels[:10].tolist())) == 1, "a-cluster not uniform"
    assert len(set(labels[10:].tolist())) == 1, "b-cluster not uniform"
    assert labels[0] != labels[10], "clusters got the same label"


def test_tenstorrent_processor_compute_features():
    """TenstorrentTensorProcessor.compute_features returns feature dicts."""
    from audio_analysis.core.tensor_operations import TenstorrentTensorProcessor

    sr = 22050
    audio = _make_sine_wave(440.0, 3.0, sr)
    audio_batch = audio[np.newaxis, :].astype(np.float32)
    lengths = np.array([len(audio)], dtype=np.int32)
    sample_rates = np.array([sr], dtype=np.int32)

    proc = TenstorrentTensorProcessor()
    result = proc.compute_features(audio_batch, lengths, sample_rates, [Path('test.wav')])

    assert isinstance(result, list)
    assert len(result) == 1
    assert 'spectral_centroid_mean' in result[0]


def test_parity_with_librosa_baseline():
    """
    JAX pipeline spectral values agree with feature_extraction_base within 25%
    relative tolerance.

    Keys compared: spectral centroid/std/rolloff/bandwidth and ZCR — these use
    the same algorithm in both paths (magnitude spectrum → weighted statistics),
    so agreement is expected to be very tight (<1% on real signals).

    Keys excluded (different algorithms/normalisations):
    - tempo, onset_density, beat_count  — different rhythm-tracking algorithms
    - detected_key                      — string label
    - filename, duration                — metadata, not computed by JAX path
    - mfcc_*                            — JAX uses plain DCT-II; librosa applies
                                          HTK mel scaling + liftering, giving
                                          different dynamic ranges
    - chroma_*, key_confidence          — JAX uses raw magnitude chroma; librosa
                                          normalises each chroma frame to unit norm
    - tonnetz_*                         — tonnetz is derived from chroma, so the
                                          same normalisation difference propagates
    - rms_*                             — librosa computes RMS per frame; JAX
                                          computes global RMS over the padded batch
                                          array, giving ~35% difference on short clips
    """
    from audio_analysis.core.jax_feature_extraction import JaxAudioFeatureExtractor
    from audio_analysis.core.feature_extraction_base import FeatureExtractionCore

    sr = 22050
    rng = np.random.default_rng(0)
    # White noise exercises all frequency bins equally
    audio = (rng.standard_normal(sr * 4) * 0.1).astype(np.float32)

    # JAX path
    ex = JaxAudioFeatureExtractor(sr=sr)
    batch = audio[np.newaxis, :]
    lengths = np.array([len(audio)], dtype=np.int32)
    jax_features = ex.extract_batch(batch, lengths, sr, [Path('test.wav')])[0]

    # librosa baseline
    core = FeatureExtractionCore(sample_rate=sr)
    lib_features = core.extract_comprehensive_features(audio, sr, Path('test.wav'), 4.0)

    # Keys that use equivalent algorithms in both paths and should agree closely.
    # MFCC, chroma, tonnetz, and rms are excluded because of known normalisation
    # differences between the JAX custom pipeline and librosa (see docstring).
    comparable_keys = {
        'spectral_centroid_mean',
        'spectral_centroid_std',
        'spectral_rolloff_mean',
        'spectral_bandwidth_mean',
        'zero_crossing_rate_mean',
    }

    # Verify all comparable keys are present in both outputs
    for key in comparable_keys:
        assert key in jax_features, f"JAX output missing key: {key}"
        assert key in lib_features, f"librosa output missing key: {key}"

    assert len(comparable_keys) >= 5, "Sanity check: at least 5 comparable keys required"

    failures = []
    for key in sorted(comparable_keys):
        jv = float(jax_features[key])
        lv = float(lib_features[key])
        if abs(lv) < 1e-6:
            continue  # skip near-zero baseline values
        rel_err = abs(jv - lv) / (abs(lv) + 1e-8)
        if rel_err > 0.25:
            failures.append(
                f"  {key}: jax={jv:.4f} librosa={lv:.4f} rel_err={rel_err:.1%}"
            )

    assert not failures, (
        "Parity failures (>25% relative error) on spectral features "
        "that should use equivalent algorithms:\n" + "\n".join(failures)
    )


def test_tenstorrent_processor_cluster_features():
    """TenstorrentTensorProcessor.cluster_features returns labels and centers."""
    from audio_analysis.core.tensor_operations import TenstorrentTensorProcessor

    rng = np.random.default_rng(0)
    features = rng.standard_normal((15, 8)).astype(np.float32)

    proc = TenstorrentTensorProcessor()
    labels, centers = proc.cluster_features(features, n_clusters=3)

    assert labels.shape == (15,)
    assert centers.shape == (3, 8)


def test_parallel_feature_extractor_uses_tt_path():
    """ParallelFeatureExtractor routes to TT when device='tenstorrent'."""
    from audio_analysis.core.parallel_feature_extraction import (
        ParallelFeatureExtractor, ProcessingConfig, AudioBatch,
    )

    sr = 22050
    config = ProcessingConfig(device='tenstorrent', use_multiprocessing=False)
    extractor = ParallelFeatureExtractor(config)

    audio = _make_sine_wave(440.0, 2.0, sr)
    batch = AudioBatch(
        audio_data=[audio],
        sample_rates=[sr],
        file_paths=[Path('test.wav')],
        durations=[2.0],
    )
    tensor_data = batch.to_tensor_format()
    result = extractor._extract_features_vectorized(batch, tensor_data)

    assert len(result) == 1
    assert 'spectral_centroid_mean' in result[0]
