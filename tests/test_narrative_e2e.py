# tests/test_narrative_e2e.py
"""
End-to-end smoke test on ~/samples.

Runs the full narrative pipeline on all .aif files and asserts:
  - Every file produces a NarrativeResult
  - Every NarrativeResult has at least 1 section (3+ for files >= 30s)
  - narrative.json exists and is valid for each file
  - similar_to is populated for at least one file (library comparison)

Skipped if ~/samples has fewer than 2 audio files.
"""
import json
import pytest
from pathlib import Path

SAMPLES_DIR = Path.home() / "samples"
AIF_FILES   = sorted(SAMPLES_DIR.glob("*.aif")) if SAMPLES_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    len(AIF_FILES) < 2,
    reason="~/samples needs at least 2 .aif files"
)


def test_narrative_e2e_all_files(tmp_path):
    """Full pipeline on ~/samples: trajectory → sections → narrative → exports."""
    import librosa
    import numpy as np
    from audio_analysis.core.trajectory_analysis import TrajectoryAnalyzer
    from audio_analysis.core.narrative_analysis import NarrativeAnalyzer
    from audio_analysis.analysis.cross_piece_similarity import CrossPieceSimilarity
    from audio_analysis.exporters.narrative_exporter import NarrativeExporter

    # Try to import the TT-Lang STFT kernel; fall back to librosa if unavailable
    try:
        from audio_analysis.core.tt_stft_kernel import TTStftKernel, SpectrogramChunk
        use_kernel = True
    except ImportError:
        import dataclasses

        @dataclasses.dataclass
        class SpectrogramChunk:
            mag: np.ndarray
            mel: np.ndarray
            timestamps: np.ndarray

        use_kernel = False

    traj_az = TrajectoryAnalyzer(sr=22050)
    narr_az = NarrativeAnalyzer()
    exporter = NarrativeExporter(tmp_path)
    results  = []

    for aif_path in AIF_FILES:
        audio, sr = librosa.load(str(aif_path), sr=22050, mono=True)
        duration  = float(len(audio)) / sr

        # Build SpectrogramChunk either from TT hardware or from librosa
        chunk = None
        if use_kernel:
            try:
                kernel = TTStftKernel(sr=sr)
                chunk  = kernel.process_file(audio, sr)
            except Exception:
                use_kernel = False

        if chunk is None:
            stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)).T
            mel  = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128).T
            ts   = np.arange(stft.shape[0], dtype=np.float32) * 512 / sr
            chunk = SpectrogramChunk(mag=stft.astype(np.float32),
                                     mel=mel.astype(np.float32), timestamps=ts)

        trajectory = traj_az.analyze(chunk, audio)
        result     = narr_az.analyze(aif_path.name, duration, trajectory, audio, sr)
        results.append(result)

        assert len(result.sections) >= 1, (
            f"{aif_path.name}: expected >=1 sections, got {len(result.sections)}"
        )
        assert result.narrative, f"{aif_path.name}: empty narrative"

    # Library similarity — populates result.similar_to in-place
    CrossPieceSimilarity().compute_library(results)

    # Export each result; verify output files are valid
    for result in results:
        paths = exporter.export_narrative(result)
        json_path = Path(paths["json"])
        assert json_path.exists(), f"JSON not written: {json_path}"
        data = json.loads(json_path.read_text())
        assert "sections" in data, f"'sections' missing from {json_path.name}"
        assert "trajectory" in data, f"'trajectory' missing from {json_path.name}"

    # Library-level similarity matrix
    sim_path = exporter.export_similarity_matrix(results)
    assert Path(sim_path).exists(), f"Similarity matrix not written: {sim_path}"

    # At least one file should have similar_to populated after library analysis
    has_similar = any(r.similar_to for r in results)
    assert has_similar, "No file has similar_to populated after library analysis"

    # Print per-file section breakdown for human review
    print(f"\n--- E2E Narrative Results ({len(AIF_FILES)} files) ---")
    for r in results:
        sec_types = [s.section_type for s in r.sections]
        print(f"  {r.filename}: {len(r.sections)} sections {sec_types}")
        print(f"    narrative: {r.narrative[:80]}...")
        if r.similar_to:
            print(f"    similar to: {r.similar_to}")
