# audio_analysis/core/narrative_types.py
"""
Shared dataclasses for temporal narrative analysis.

These types flow through every stage of the narrative pipeline:

    audio file
        → TrajectoryPoint  (one 2-second feature snapshot)
        → Section          (a named structural segment containing many TrajectoryPoints)
        → NarrativeResult  (the complete per-file output)

SectionMotion sits at each Section boundary and describes how the energy
and texture move when the piece enters or exits that section.

Design notes
------------
- All dataclasses use plain Python field types plus numpy.ndarray for the
  texture_fingerprint.  No validators are applied at construction time so
  that lightweight construction in tests and demos stays friction-free.
- Tension scores are always in the 0–1 range by convention; nothing in
  these types enforces that — it is the responsibility of the compute code.
- `similar_to` in NarrativeResult is intentionally empty at creation and
  gets populated later by library-wide comparison passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class TrajectoryPoint:
    """Feature snapshot for a 2-second window centered at `time`.

    Each point is computed by slicing the audio around `time`, running a
    short STFT, and deriving a small set of perceptually meaningful
    descriptors.  The 2-second window is chosen so that it is long enough
    to capture a full period of most synthesizer LFOs (≥ 0.5 Hz) while
    still being short enough to track fast energy rises.

    Attributes
    ----------
    time : float
        Centre of the analysis window, in seconds from the start of the file.
    energy : float
        RMS amplitude derived from the magnitude spectrum (not waveform), so
        it reflects perceived loudness rather than clip level.
    brightness : float
        Spectral centroid in Hz — the "centre of mass" of the spectrum.
        Low values indicate sub-bass or dark pads; high values indicate
        bright leads or metallic percussion.
    roughness : float
        Mean absolute difference between adjacent STFT frequency bins,
        normalised to [0, 1].  High roughness correlates with gritty or
        noisy textures; low roughness with pure tones and smooth pads.
    zcr : float
        Zero-crossing rate expressed as crossings per sample.  Provides a
        lightweight noisiness estimate that complements roughness.
    chroma_peak : str
        The dominant pitch class at this moment, one of the twelve names
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B".
    chroma_spread : float
        Normalised chroma entropy, 0–1.  A value near 0 means the energy is
        concentrated on one or two pitch classes (tonal).  A value near 1
        means energy is spread evenly across all twelve classes (noise-like
        or atonal).
    tension_score : float
        Composite 0–1 score combining energy, roughness, and chroma_spread.
        Computed by the trajectory builder after the full trajectory is
        assembled so that it can be normalised to the piece's own dynamic
        range.  0 = most relaxed point, 1 = most tense point.
    """

    time: float            # seconds from file start
    energy: float          # RMS from magnitude spectrum
    brightness: float      # spectral centroid (Hz)
    roughness: float       # mean |diff| between adjacent frequency bins
    zcr: float             # zero-crossing rate (crossings / sample)
    chroma_peak: str       # dominant pitch class ("C", "C#", … "B")
    chroma_spread: float   # normalised chroma entropy, 0–1
    tension_score: float   # 0–1 composite (computed after full trajectory)


@dataclass
class SectionMotion:
    """Describes how a section boundary sounds when approached or left.

    Used as both `motion_in` and `motion_out` on every Section so that
    the narrative generator can write phrases like "builds gradually into
    the climax" or "fades instantly into silence".

    Attributes
    ----------
    direction : str
        The dominant energy direction at the boundary.
        One of: "rising" | "falling" | "stable" | "oscillating".
    rate : str
        How quickly the change happens.
        One of: "fast" | "gradual" | "instant".
    type : str
        The qualitative character of the transition.
        One of: "abrupt" | "gradual" | "fade" | "swell".
    """

    direction: str   # "rising" | "falling" | "stable" | "oscillating"
    rate: str        # "fast" | "gradual" | "instant"
    type: str        # "abrupt" | "gradual" | "fade" | "swell"


@dataclass
class Section:
    """One structural section of a piece.

    Sections are produced by the section detector, which groups nearby
    TrajectoryPoints whose characteristics are mutually similar.  Each
    Section is then annotated with higher-level labels by the narrative
    builder.

    Attributes
    ----------
    start : float
        Section start time, in seconds.
    end : float
        Section end time, in seconds.
    section_type : str
        High-level structural role.  One of:
        "intro" | "rising" | "plateau" | "climax" | "falling" | "release"
        | "outro".
    tension_arc : str
        Describes the tension contour inside the section.  One of:
        "building" | "peak" | "plateau" | "releasing" | "valley".
    motion_in : SectionMotion
        How the piece moves *into* this section from the previous one
        (or from silence for the first section).
    motion_out : SectionMotion
        How the piece moves *out of* this section into the next one
        (or into silence for the last section).
    dominant_mood : str
        The single most prominent mood descriptor for the section, as
        returned by MoodAnalyzer run on the section's audio slice.
    dominant_character : str
        The single most prominent character tag for the section, as
        returned by CharacterAnalyzer run on the section's audio slice.
    instruments : List[str]
        All character tags that appear at least once within the section's
        trajectory, de-duplicated and sorted.
    tension_score : float
        Mean tension_score over all TrajectoryPoints within the section.
        Provides a single-number summary for fingerprinting and ranking.
    trajectory : List[TrajectoryPoint]
        The ordered sequence of 2-second TrajectoryPoints that fall inside
        [start, end].  The first point's time ≥ start; the last point's
        time ≤ end.
    """

    start: float                         # seconds
    end: float                           # seconds
    section_type: str                    # intro | rising | plateau | climax | falling | release | outro
    tension_arc: str                     # building | peak | plateau | releasing | valley
    motion_in: SectionMotion
    motion_out: SectionMotion
    dominant_mood: str                   # from MoodAnalyzer on section audio
    dominant_character: str              # from CharacterAnalyzer on section audio
    instruments: List[str]               # character tags present in this section
    tension_score: float                 # mean tension_score over section trajectory
    trajectory: List[TrajectoryPoint]    # 2 s points within section


@dataclass
class NarrativeResult:
    """Complete per-file output of the narrative pipeline.

    This is the top-level return value of NarrativeAnalyzer.analyze().
    It bundles the human-readable prose narrative together with all of the
    structured data needed for library-wide comparisons and downstream
    use by the sequencer and exporter.

    Attributes
    ----------
    filename : str
        The source audio filename (basename only, no directory).
    duration : float
        Total duration of the audio file, in seconds.
    narrative : str
        A prose paragraph (typically 3–6 sentences) describing the
        piece's temporal arc, written in composer-friendly language.
    sections : List[Section]
        Ordered list of structural sections covering the full duration.
    trajectory : List[TrajectoryPoint]
        Full-piece trajectory (every 2-second window), used for
        library-wide comparison and fingerprinting.
    structure_fingerprint : List[Tuple[str, float]]
        A compact sequence of (section_type, tension_score) pairs — one
        per section — that characterises the piece's macro-level shape.
        Example: [("intro", 0.2), ("rising", 0.5), ("climax", 0.9)].
        Useful for finding pieces with similar narrative arcs.
    texture_fingerprint : numpy.ndarray
        A 10-dimensional float32 vector summarising the piece's average
        timbral texture.  Components are derived from mean and variance of
        brightness, roughness, chroma_spread, energy, and zcr across the
        full trajectory.  Used for cosine-similarity comparisons between
        tracks.
    similar_to : List[str]
        Filenames of other tracks in the library that are considered
        structurally or texturally similar.  This list is empty at
        creation time and is populated by a separate library-comparison
        pass after all files have been analysed.
    """

    filename: str
    duration: float
    narrative: str                                  # prose description
    sections: List[Section]
    trajectory: List[TrajectoryPoint]               # full-piece trajectory
    structure_fingerprint: List[Tuple[str, float]]  # [(section_type, tension_score), ...]
    texture_fingerprint: np.ndarray                 # 10-dim float32 vector
    similar_to: List[str]                           # filenames; populated after library comparison
