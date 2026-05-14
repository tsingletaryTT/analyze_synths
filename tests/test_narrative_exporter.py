# tests/test_narrative_exporter.py
import json
import numpy as np
import pytest
from pathlib import Path
from audio_analysis.exporters.narrative_exporter import NarrativeExporter
from audio_analysis.core.narrative_types import (
    NarrativeResult, Section, SectionMotion, TrajectoryPoint,
)


def _make_result(filename: str = "test_piece.aif") -> NarrativeResult:
    motion = SectionMotion("stable", "gradual", "gradual")
    tp = TrajectoryPoint(1.0, 0.05, 1800.0, 0.07, 0.10, "C", 0.5, 0.3)
    sec = Section(
        start=0.0, end=30.0, section_type="intro",
        tension_arc="valley", motion_in=motion, motion_out=motion,
        dominant_mood="spacey", dominant_character="pad_synth",
        instruments=["pad_synth"], tension_score=0.25, trajectory=[tp],
    )
    return NarrativeResult(
        filename=filename, duration=30.0,
        narrative="Opens quietly with a spacey, ambient texture.",
        sections=[sec], trajectory=[tp],
        structure_fingerprint=[("intro", 0.25)],
        texture_fingerprint=np.zeros(10, dtype=np.float32),
        similar_to=["other_piece.aif"],
    )


def test_narrative_json_is_valid(tmp_path):
    exp = NarrativeExporter(tmp_path)
    result = _make_result()
    exp.export_narrative(result)

    json_path = tmp_path / "test_piece_narrative.json"
    assert json_path.exists(), f"{json_path} not found"

    with open(json_path) as f:
        data = json.load(f)

    assert data["filename"] == "test_piece.aif"
    assert "narrative" in data
    assert "sections" in data
    assert "trajectory" in data
    assert "similar_to" in data
    assert len(data["sections"]) == 1


def test_narrative_json_has_all_section_fields(tmp_path):
    exp = NarrativeExporter(tmp_path)
    result = _make_result()
    exp.export_narrative(result)

    with open(tmp_path / "test_piece_narrative.json") as f:
        data = json.load(f)

    sec = data["sections"][0]
    required = {"start", "end", "section_type", "tension_arc",
                 "dominant_mood", "dominant_character",
                 "instruments", "tension_score"}
    for key in required:
        assert key in sec, f"missing section field: {key}"


def test_narrative_md_created(tmp_path):
    exp = NarrativeExporter(tmp_path)
    result = _make_result()
    exp.export_narrative(result)

    md_path = tmp_path / "test_piece_narrative.md"
    assert md_path.exists()
    content = md_path.read_text()
    assert "spacey" in content or "intro" in content


def test_similarity_matrix_csv(tmp_path):
    exp = NarrativeExporter(tmp_path)
    results = [_make_result("a.aif"), _make_result("b.aif")]
    results[0].similar_to = ["b.aif"]
    results[1].similar_to = ["a.aif"]

    exp.export_similarity_matrix(results)

    csv_path = tmp_path / "similarity_matrix.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "filename_a" in content
    assert "a.aif" in content


def test_export_narrative_filename_with_spaces(tmp_path):
    exp = NarrativeExporter(tmp_path)
    result = _make_result("my piece with spaces.aif")
    exp.export_narrative(result)
    json_path = tmp_path / "my piece with spaces_narrative.json"
    assert json_path.exists()
