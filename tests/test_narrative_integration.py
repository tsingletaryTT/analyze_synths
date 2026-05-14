"""
Integration test: run ParallelAudioAnalyzer on ~/samples and assert
NarrativeResult is produced for every file with at least 1 section.

Skipped if ~/samples doesn't exist or has no audio files.
"""
import os
import pytest
from pathlib import Path

SAMPLES_DIR = Path.home() / "samples"

pytestmark = pytest.mark.skipif(
    not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.glob("*.aif")),
    reason="~/samples directory with .aif files required"
)


def test_parallel_analyzer_produces_narrative_results():
    """Every file in ~/samples should produce a NarrativeResult with >=1 section."""
    from audio_analysis.api.parallel_analyzer import ParallelAudioAnalyzer, ProcessingConfig

    config = ProcessingConfig(
        max_workers=4,
        batch_size=4,
        enable_tensor_optimization=False,
        memory_limit_mb=4096,
    )
    analyzer = ParallelAudioAnalyzer(str(SAMPLES_DIR), config)
    analyzer.analyze_directory()

    narrative_results = analyzer.narrative_results
    assert narrative_results, "No narrative results produced"

    expected_count = len(list(SAMPLES_DIR.glob("*.aif")))
    assert len(narrative_results) == expected_count, (
        f"Expected {expected_count} narrative results, got {len(narrative_results)}"
    )

    for filename, result in narrative_results.items():
        assert len(result.sections) >= 1, (
            f"{filename}: expected at least 1 section, got {len(result.sections)}"
        )
        assert result.narrative, f"{filename}: narrative prose is empty"
        assert result.trajectory, f"{filename}: trajectory is empty"


def test_narrative_exports_created(tmp_path):
    """NarrativeExporter should create .json and .md files for each result."""
    from audio_analysis.api.parallel_analyzer import ParallelAudioAnalyzer, ProcessingConfig
    from audio_analysis.exporters.narrative_exporter import NarrativeExporter

    config = ProcessingConfig(max_workers=2, batch_size=2,
                              enable_tensor_optimization=False, memory_limit_mb=2048)
    analyzer = ParallelAudioAnalyzer(str(SAMPLES_DIR), config)
    analyzer.analyze_directory()

    exporter = NarrativeExporter(tmp_path)
    for result in analyzer.narrative_results.values():
        exporter.export_narrative(result)

    json_files = list(tmp_path.glob("*_narrative.json"))
    assert len(json_files) == len(analyzer.narrative_results), (
        f"Expected {len(analyzer.narrative_results)} JSON files, got {len(json_files)}"
    )

    md_files = list(tmp_path.glob("*_narrative.md"))
    assert len(md_files) == len(analyzer.narrative_results), (
        f"Expected {len(analyzer.narrative_results)} MD files, got {len(md_files)}"
    )
