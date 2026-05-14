# tests/test_narrative_analysis.py
import numpy as np
import pytest
from audio_analysis.core.narrative_types import TrajectoryPoint, SectionMotion, Section
from audio_analysis.core.narrative_analysis import NarrativeAnalyzer


def _make_trajectory(n: int, tension: float = 0.3) -> list:
    return [
        TrajectoryPoint(
            time=float(i * 2),
            energy=0.05, brightness=1800.0, roughness=0.07,
            zcr=0.10, chroma_peak="C", chroma_spread=0.5,
            tension_score=tension,
        )
        for i in range(n)
    ]


def _make_jump_trajectory(n_before: int, n_after: int,
                           tension_before: float = 0.2,
                           tension_after: float = 0.8) -> list:
    before = [
        TrajectoryPoint(
            time=float(i * 2),
            energy=0.03, brightness=1200.0, roughness=0.04,
            zcr=0.05, chroma_peak="C", chroma_spread=0.4,
            tension_score=tension_before,
        )
        for i in range(n_before)
    ]
    after = [
        TrajectoryPoint(
            time=float((n_before + i) * 2),
            energy=0.15, brightness=3000.0, roughness=0.18,
            zcr=0.25, chroma_peak="G", chroma_spread=0.7,
            tension_score=tension_after,
        )
        for i in range(n_after)
    ]
    return before + after


def test_no_sections_for_short_uniform_audio():
    traj = _make_trajectory(20)   # 40 seconds, flat
    az = NarrativeAnalyzer()
    sections = az.detect_sections(traj, duration=40.0)
    assert len(sections) >= 1


def test_jump_at_30s_detected_within_8s():
    traj = _make_jump_trajectory(15, 15)
    az = NarrativeAnalyzer()
    sections = az.detect_sections(traj, duration=60.0)
    assert len(sections) >= 2, "Expected at least 2 sections around the jump"

    boundaries = [s.start for s in sections[1:]]
    closest = min(boundaries, key=lambda b: abs(b - 30.0))
    assert abs(closest - 30.0) <= 8.0, (
        f"No section boundary within 8s of jump at 30s; boundaries={boundaries}"
    )


def test_minimum_section_duration_enforced():
    traj = _make_jump_trajectory(5, 25)   # jump at 10s, total 60s
    az = NarrativeAnalyzer()
    sections = az.detect_sections(traj, duration=60.0)
    for sec in sections:
        assert (sec.end - sec.start) >= 14.0, (
            f"Section too short: {sec.start:.1f}–{sec.end:.1f}"
        )


def test_maximum_section_duration_enforced():
    # Build a 180-second trajectory with a slow-ramp tension variation so that
    # _split_long has genuine gradient peaks to split on.  A completely flat
    # trajectory would correctly *not* be split (no change point exists), so
    # we need real variation to exercise the max-duration ceiling.
    n = 90  # 180 seconds at 2s/point
    traj = [
        TrajectoryPoint(
            time=float(i * 2),
            energy=0.05 + 0.10 * abs(np.sin(i * np.pi / 30)),
            brightness=1800.0 + 400.0 * np.sin(i * np.pi / 30),
            roughness=0.07,
            zcr=0.10, chroma_peak="C", chroma_spread=0.5,
            tension_score=0.2 + 0.6 * abs(np.sin(i * np.pi / 30)),
        )
        for i in range(n)
    ]
    az = NarrativeAnalyzer()
    sections = az.detect_sections(traj, duration=180.0)
    for sec in sections:
        assert (sec.end - sec.start) <= 62.0, (
            f"Section too long: {sec.start:.1f}–{sec.end:.1f}"
        )


def _section_from_tension(tensions: list, start: float = 0.0, sr_points: float = 2.0) -> Section:
    traj = [
        TrajectoryPoint(
            time=start + i * sr_points,
            energy=t * 0.1, brightness=1800.0, roughness=0.07,
            zcr=0.10, chroma_peak="C", chroma_spread=0.5,
            tension_score=t,
        )
        for i, t in enumerate(tensions)
    ]
    motion = SectionMotion("stable", "gradual", "gradual")
    mean_t = float(np.mean(tensions))
    end = start + len(tensions) * sr_points
    return Section(
        start=start, end=end, section_type="plateau",
        tension_arc="plateau", motion_in=motion, motion_out=motion,
        dominant_mood="unknown", dominant_character="unknown",
        instruments=[], tension_score=mean_t, trajectory=traj,
    )


def test_section_type_intro():
    az = NarrativeAnalyzer()
    sec = _section_from_tension([0.1, 0.15, 0.2, 0.18, 0.12], start=0.0)
    sections = [sec]
    az._classify_sections(sections, duration=sec.end)
    assert sections[0].section_type == "intro", f"got {sections[0].section_type}"


def test_section_type_outro():
    az = NarrativeAnalyzer()
    sec = _section_from_tension([0.15, 0.12, 0.10, 0.08], start=100.0)
    sections = [sec]
    az._classify_sections(sections, duration=sec.end)
    assert sections[0].section_type == "outro", f"got {sections[0].section_type}"


def test_section_type_rising():
    az = NarrativeAnalyzer()
    tensions = [0.2, 0.3, 0.4, 0.5, 0.6]   # 0.4/10s = 0.04/s > 0.015 threshold
    sec = _section_from_tension(tensions, start=30.0)
    sections = [sec, _section_from_tension([0.5], start=sec.end)]
    az._classify_sections(sections, duration=sections[-1].end)
    assert sections[0].section_type == "rising", f"got {sections[0].section_type}"


def test_section_type_climax():
    az = NarrativeAnalyzer()
    tensions = [0.80, 0.82, 0.78, 0.81, 0.79]
    sec = _section_from_tension(tensions, start=60.0)
    sections = [_section_from_tension([0.2], start=0.0), sec]
    az._classify_sections(sections, duration=sec.end)
    assert sections[1].section_type == "climax", f"got {sections[1].section_type}"


def test_tension_arc_building():
    az = NarrativeAnalyzer()
    tensions = [0.3, 0.4, 0.5, 0.6]
    sec = _section_from_tension(tensions)
    az._assign_tension_arc(sec)
    assert sec.tension_arc == "building", f"got {sec.tension_arc}"


def test_motion_abrupt_or_swell():
    az = NarrativeAnalyzer()
    before = [TrajectoryPoint(t, 0.02, 1000.0, 0.03, 0.05, "C", 0.4, 0.1)
              for t in [0.0, 2.0]]
    after  = [TrajectoryPoint(t, 0.20, 4000.0, 0.20, 0.30, "G", 0.6, 0.9)
              for t in [4.0, 6.0]]
    motion = az._compute_motion(before, after)
    assert motion.type in ("abrupt", "swell"), f"got {motion.type}"


def test_split_long_flat_tension_no_micro_sections():
    """Flat trajectory split should not produce sections < MIN_SECTION_SEC."""
    from audio_analysis.core.narrative_analysis import MIN_SECTION_SEC
    traj = _make_trajectory(90, tension=0.5)  # 180s, completely flat
    az = NarrativeAnalyzer()
    sections = az.detect_sections(traj, duration=180.0)
    for sec in sections:
        assert (sec.end - sec.start) >= MIN_SECTION_SEC - 1.0, (
            f"Micro-section: {sec.start:.1f}–{sec.end:.1f}"
        )


def test_narrative_prose_non_empty():
    """analyze() must return NarrativeResult with non-empty prose."""
    import librosa
    sr = 22050
    t = np.linspace(0, 30, sr * 30, endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel, SpectrogramChunk
    from audio_analysis.core.trajectory_analysis import TrajectoryAnalyzer

    kernel = TTStftKernel(sr=sr)
    chunk = kernel.process_file(audio, sr)
    traj_az = TrajectoryAnalyzer(sr=sr)
    trajectory = traj_az.analyze(chunk, audio)

    az = NarrativeAnalyzer()
    result = az.analyze("test_sine.wav", duration=30.0,
                        trajectory=trajectory, audio=audio, sr=sr)

    assert isinstance(result.narrative, str)
    assert len(result.narrative) > 20
    assert len(result.sections) >= 1
    assert result.texture_fingerprint.shape == (10,)
    assert isinstance(result.structure_fingerprint, list)


def test_narrative_word_count():
    """Narrative prose: 30–250 words."""
    import librosa
    sr = 22050
    t = np.linspace(0, 60, sr * 60, endpoint=False)
    envelope = np.linspace(0.01, 0.15, sr * 60)
    audio = (envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    from audio_analysis.core.tt_stft_kernel import TTStftKernel
    from audio_analysis.core.trajectory_analysis import TrajectoryAnalyzer

    kernel = TTStftKernel(sr=sr)
    chunk = kernel.process_file(audio, sr)
    traj_az = TrajectoryAnalyzer(sr=sr)
    trajectory = traj_az.analyze(chunk, audio)

    az = NarrativeAnalyzer()
    result = az.analyze("ramp_test.wav", duration=60.0,
                        trajectory=trajectory, audio=audio, sr=sr)

    word_count = len(result.narrative.split())
    assert 30 <= word_count <= 250, f"word_count={word_count}: {result.narrative!r}"
