# tests/test_narrative_types.py
import numpy as np
import pytest
from audio_analysis.core.narrative_types import (
    TrajectoryPoint, SectionMotion, Section, NarrativeResult
)


def test_trajectory_point_fields():
    tp = TrajectoryPoint(
        time=1.0, energy=0.05, brightness=2000.0,
        roughness=0.08, zcr=0.12, chroma_peak="C",
        chroma_spread=0.6, tension_score=0.4,
    )
    assert tp.time == 1.0
    assert tp.chroma_peak == "C"
    assert 0.0 <= tp.tension_score <= 1.0


def test_section_motion_fields():
    m = SectionMotion(direction="rising", rate="gradual", type="swell")
    assert m.direction in ("rising", "falling", "stable", "oscillating")
    assert m.rate in ("fast", "gradual", "instant")
    assert m.type in ("abrupt", "gradual", "fade", "swell")


def test_section_fields():
    from audio_analysis.core.narrative_types import TrajectoryPoint
    tp = TrajectoryPoint(0.0, 0.04, 1800.0, 0.07, 0.10, "G", 0.55, 0.35)
    motion = SectionMotion("stable", "gradual", "gradual")
    sec = Section(
        start=0.0, end=30.0, section_type="intro",
        tension_arc="valley", motion_in=motion, motion_out=motion,
        dominant_mood="spacey", dominant_character="pad_synth",
        instruments=["pad_synth"], tension_score=0.25,
        trajectory=[tp],
    )
    assert sec.section_type == "intro"
    assert len(sec.trajectory) == 1


def test_narrative_result_fields():
    nr = NarrativeResult(
        filename="test.aif", duration=120.0, narrative="A short piece.",
        sections=[], trajectory=[],
        structure_fingerprint=[("intro", 0.2)],
        texture_fingerprint=np.zeros(10, dtype=np.float32),
        similar_to=[],
    )
    assert nr.filename == "test.aif"
    assert nr.texture_fingerprint.shape == (10,)
