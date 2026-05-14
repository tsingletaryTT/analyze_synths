# tests/test_cross_piece_similarity.py
import numpy as np
import pytest
from audio_analysis.analysis.cross_piece_similarity import CrossPieceSimilarity
from audio_analysis.core.narrative_types import NarrativeResult


def _make_result(filename: str, section_types: list, tension_scores: list,
                 texture: np.ndarray = None) -> NarrativeResult:
    if texture is None:
        texture = np.random.rand(10).astype(np.float32)
        texture /= np.linalg.norm(texture) + 1e-9
    struct_fp = list(zip(section_types, tension_scores))
    return NarrativeResult(
        filename=filename, duration=120.0, narrative="test",
        sections=[], trajectory=[],
        structure_fingerprint=struct_fp,
        texture_fingerprint=texture,
        similar_to=[],
    )


def test_similarity_identical_is_one():
    sim = CrossPieceSimilarity()
    r = _make_result("a.aif", ["intro", "climax", "outro"], [0.2, 0.8, 0.15])
    score = sim.score(r, r)
    assert abs(score - 1.0) < 0.01, f"identical similarity={score}"


def test_identical_pieces_most_similar():
    """File A and B share same structure; C is different. A-B > A-C."""
    sim = CrossPieceSimilarity()
    struct_ab = ["intro", "rising", "climax", "outro"]
    tension_ab = [0.2, 0.5, 0.8, 0.1]
    texture_ab = np.array([0.1, 0.2, 0.3, 0.1, 0.15, 0.05, 0.4, 0.1, 0.5, 0.2], dtype=np.float32)
    texture_ab /= np.linalg.norm(texture_ab)

    struct_c = ["plateau", "plateau", "plateau"]
    tension_c = [0.5, 0.5, 0.5]
    texture_c = np.array([0.9, 0.05, 0.02, 0.9, 0.05, 0.02, 0.1, 0.9, 0.05, 0.01], dtype=np.float32)
    texture_c /= np.linalg.norm(texture_c)

    r_a = _make_result("a.aif", struct_ab, tension_ab, texture_ab.copy())
    r_b = _make_result("b.aif", struct_ab, tension_ab, texture_ab.copy())
    r_c = _make_result("c.aif", struct_c,  tension_c,  texture_c)

    score_ab = sim.score(r_a, r_b)
    score_ac = sim.score(r_a, r_c)
    assert score_ab > score_ac, f"A-B ({score_ab:.3f}) should be > A-C ({score_ac:.3f})"


def test_compute_library_populates_similar_to():
    sim = CrossPieceSimilarity()
    results = [
        _make_result(f"track_{i}.aif", ["intro", "climax", "outro"], [0.2, 0.8, 0.1])
        for i in range(4)
    ]
    sim.compute_library(results)
    for r in results:
        assert isinstance(r.similar_to, list)
        assert len(r.similar_to) <= 3
        assert r.filename not in r.similar_to


def test_structure_fingerprint_encoding_length():
    sim = CrossPieceSimilarity()
    fp = sim._encode_structure([("intro", 0.2), ("climax", 0.8), ("outro", 0.1)])
    assert fp.shape == (35,), f"expected (35,) got {fp.shape}"
