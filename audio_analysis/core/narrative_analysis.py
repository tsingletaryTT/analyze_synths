# audio_analysis/core/narrative_analysis.py
"""
NarrativeAnalyzer: section detection, classification, motion descriptors,
per-section mood/character, and prose generation.

The pipeline has four stages:

  1. detect_sections(trajectory, duration)
       Runs a sliding-window change-point detector over the trajectory's
       energy/brightness/roughness/tension features, then enforces minimum
       and maximum section lengths, returning a flat list of Section objects
       with placeholder type labels.

  2. _classify_sections(sections, duration)
       Applies position- and slope-aware heuristics to assign a
       ``section_type`` (intro, rising, plateau, climax, falling, release,
       outro) and calls ``_assign_tension_arc`` for each section.  Also
       computes the ``SectionMotion`` objects at each inter-section boundary.

  3. _compute_motion(before_traj, after_traj)
       Characterises the sonic transition between two adjacent trajectory
       windows and returns a ``SectionMotion`` describing type, rate, and
       direction.

  4. analyze(filename, duration, trajectory, audio, sr)
       Full pipeline entry point: detect → classify → extract per-section
       mood/character features → generate prose → compute fingerprints →
       return a NarrativeResult.

Usage:
    az = NarrativeAnalyzer()
    sections = az.detect_sections(trajectory, duration)
    az._classify_sections(sections, duration)

    # Or, full pipeline:
    result = az.analyze(filename, duration, trajectory, audio, sr)
"""
import logging
from typing import List

import numpy as np

from audio_analysis.core.narrative_types import (
    NarrativeResult, Section, SectionMotion, TrajectoryPoint,
)

# ------------------------------------------------------------------
# Module-level prose templates and lookup tables
# ------------------------------------------------------------------

# Per-section-type sentence templates.  Keys match section_type values.
# Placeholder tokens:
#   {title}          — filename stem
#   {time}           — "M:SS" formatted section start time
#   {adverb}         — drawn from _ADVERBS keyed by tension_arc
#   {intensity_word} — drawn from _INTENSITY keyed by section_type
#   {mood_phrase}    — dominant_mood with underscores replaced by spaces
#   {arc_phrase}     — tension_arc label
#   {feature_phrase} — brief spectral description based on tension_arc
_SECTION_OPENERS = {
    "intro":   ["{title} begins with {mood_phrase}, setting a {arc_phrase} tone.",
                "The piece opens {adverb} — {mood_phrase}."],
    "rising":  ["Around {time}, the texture shifts — {mood_phrase} as {feature_phrase}.",
                "From {time}, the energy climbs: {mood_phrase}."],
    "plateau": ["A {intensity_word} plateau settles at {time} — {mood_phrase}.",
                "The mood holds {adverb} through {time}: {mood_phrase}."],
    "climax":  ["A {intensity_word} peak arrives at {time} — {mood_phrase}.",
                "By {time} the tension peaks, {mood_phrase}."],
    "falling": ["The intensity eases from {time} — {mood_phrase}.",
                "Around {time}, the energy recedes: {mood_phrase}."],
    "release": ["After {time}, {mood_phrase} — a sense of release.",
                "The tension dissolves at {time}, leaving {mood_phrase}."],
    "outro":   ["The piece closes {adverb} — {mood_phrase}.",
                "A {mood_phrase} resolution carries through to the end."],
}

# Adverb lookup: tension_arc → descriptive adverb for template substitution.
_ADVERBS = {
    "building":  "quietly at first",
    "peak":      "boldly",
    "plateau":   "steadily",
    "releasing": "gradually",
    "valley":    "softly",
}

# Intensity adjective lookup: section_type → single descriptive word.
_INTENSITY = {
    "climax":   "intense",
    "rising":   "growing",
    "falling":  "ebbing",
    "plateau":  "sustained",
    "intro":    "gentle",
    "outro":    "quiet",
    "release":  "restful",
}

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level constants
# ------------------------------------------------------------------

# Normalised feature change required to register a section boundary.
# Lowering this makes the detector more sensitive; raising it fewer sections.
CHANGE_THRESHOLD = 0.15

# Hard floor on section duration.  Boundaries that would create a section
# shorter than this are merged with the adjacent section.
MIN_SECTION_SEC = 15.0

# Hard ceiling on section duration.  Sections longer than this are
# recursively split at the local tension-gradient peak.
MAX_SECTION_SEC = 60.0

# Half-width (in trajectory points) of the before/after windows used when
# computing the change score at each candidate boundary.
CHANGE_WINDOW = 5

# Features included in the change-score calculation.  Tension score is
# included because it is a composite already normalised to [0, 1].
FIELDS = ["energy", "brightness", "roughness", "tension_score"]


class NarrativeAnalyzer:
    """Detects and classifies structural sections in an audio trajectory.

    All methods operate on pre-built ``List[TrajectoryPoint]`` objects;
    no raw audio I/O is performed here.  This makes the class easy to unit-
    test with synthetic trajectories and easy to compose into larger pipelines.
    """

    # ------------------------------------------------------------------
    # Phase 1: change-point detection
    # ------------------------------------------------------------------

    def detect_sections(
        self, trajectory: List[TrajectoryPoint], duration: float
    ) -> List[Section]:
        """Convert a trajectory into a list of Sections via change-point detection.

        Algorithm
        ---------
        For each trajectory index *t* (with a margin of ``CHANGE_WINDOW``
        on each side), we compute a normalised change score by comparing the
        mean of each feature in the ``CHANGE_WINDOW`` points *before* t with
        the mean in the ``CHANGE_WINDOW`` points *at and after* t.  The per-
        feature differences are normalised by the feature's range across the
        whole trajectory, then averaged into a single scalar score.

        A non-maximum suppression pass then selects peaks above
        ``CHANGE_THRESHOLD`` while enforcing a minimum gap of
        ``MIN_SECTION_SEC / 2`` between consecutive boundaries.

        After converting peak indices to times and appending 0 and ``duration``
        as hard boundaries, we:
          - Merge any gap shorter than ``MIN_SECTION_SEC`` into its right
            neighbour (``_merge_short``).
          - Recursively split any gap longer than ``MAX_SECTION_SEC`` at the
            locally largest tension gradient (``_split_long``).

        Parameters
        ----------
        trajectory:
            Ordered list of ``TrajectoryPoint`` snapshots (typically one
            every 2 seconds).
        duration:
            Total audio duration in seconds; used as the final boundary.

        Returns
        -------
        List[Section]
            At least one Section covering [0, duration].  Each Section has
            placeholder ``section_type`` and ``tension_arc`` values ("plateau")
            that are overwritten by ``_classify_sections``.
        """
        n = len(trajectory)
        if n == 0:
            return []

        # --- Compute per-feature normalisation ranges ---
        # We normalise each feature to [0, 1] across the entire trajectory
        # so that low-dynamic-range features (e.g. roughness in a quiet pad)
        # do not dominate the change score.
        ranges: dict = {}
        for f in FIELDS:
            vals = np.array([getattr(tp, f) for tp in trajectory], dtype=np.float32)
            lo, hi = vals.min(), vals.max()
            # Protect against zero range (constant feature): use 1.0 so the
            # normalised value is always 0, contributing nothing to change score.
            ranges[f] = (lo, hi - lo) if hi > lo else (lo, 1.0)

        # --- Sliding-window change score ---
        scores = np.zeros(n, dtype=np.float32)
        for t in range(CHANGE_WINDOW, n - CHANGE_WINDOW):
            before = trajectory[t - CHANGE_WINDOW: t]
            after  = trajectory[t: t + CHANGE_WINDOW]
            diffs = []
            for f in FIELDS:
                lo, rng = ranges[f]
                mean_b = np.mean([(getattr(tp, f) - lo) / rng for tp in before])
                mean_a = np.mean([(getattr(tp, f) - lo) / rng for tp in after])
                diffs.append(abs(mean_a - mean_b))
            scores[t] = float(np.mean(diffs))

        # --- Non-maximum suppression peak picking ---
        # min_gap enforces that two boundaries are at least MIN_SECTION_SEC/2
        # trajectory steps apart (each step is 2 s, so divide by 2).
        min_gap = max(1, int(MIN_SECTION_SEC / 2.0))
        boundary_indices = self._peak_indices(scores, threshold=CHANGE_THRESHOLD,
                                              min_gap=min_gap)

        # --- Build the boundary time list ---
        times = [0.0] + [trajectory[i].time for i in boundary_indices] + [duration]
        times = sorted(set(times))

        # Merge sections shorter than the minimum allowed duration
        times = self._merge_short(times, MIN_SECTION_SEC)

        # Split sections longer than the maximum allowed duration
        times = self._split_long(times, trajectory, MAX_SECTION_SEC)

        # --- Construct placeholder Section objects ---
        placeholder_motion = SectionMotion("stable", "gradual", "gradual")
        sections = []
        for i in range(len(times) - 1):
            start, end = times[i], times[i + 1]
            sec_traj = [tp for tp in trajectory if start <= tp.time < end]
            mean_tension = (float(np.mean([tp.tension_score for tp in sec_traj]))
                            if sec_traj else 0.3)
            sections.append(Section(
                start=start, end=end,
                section_type="plateau",      # overwritten by _classify_sections
                tension_arc="plateau",        # overwritten by _assign_tension_arc
                motion_in=placeholder_motion,
                motion_out=placeholder_motion,
                dominant_mood="unknown",
                dominant_character="unknown",
                instruments=[],
                tension_score=mean_tension,
                trajectory=sec_traj,
            ))
        return sections

    # ------------------------------------------------------------------
    # Phase 2: section classification
    # ------------------------------------------------------------------

    def _classify_sections(self, sections: List[Section], duration: float) -> None:
        """Assign ``section_type``, ``tension_arc``, and boundary ``SectionMotion``
        objects to every section in-place.

        Classification heuristics (applied in priority order)
        -------------------------------------------------------
        1. **intro** — first section with mean tension < 0.25.  An intro is
           characterised by low energy and sparse texture as the piece
           establishes its sonic world.
        2. **outro** — last section with mean tension < 0.25.  A mirror of
           the intro; energy winds down toward silence.
        3. **rising** — positive tension slope > 0.015 tension/second.
           The piece is building momentum.
        4. **climax** — mean tension > 0.75.  The peak of energy and
           complexity; the most intense point.
        5. **plateau** — mean tension > 0.65 but slope is near-flat
           (|slope| < 0.015).  High intensity that is sustained rather than
           climactic.
        6. **falling** — negative slope < -0.015.  Energy is declining.
        7. **release** — after a high-tension section, mean tension < 0.35.
           A deliberate step-down, not a gradual fade.
        8. **plateau** — default for sections that do not match any above.

        After section types are set, ``_assign_tension_arc`` is called on
        each section to label its internal tension contour.

        Finally, the inter-section ``SectionMotion`` objects are computed by
        comparing the last two trajectory points of each section against the
        first two of the next.

        Parameters
        ----------
        sections:
            The list returned by ``detect_sections``.  Modified in-place.
        duration:
            Total file duration (seconds); not used directly but kept for
            future position-based heuristics.
        """
        n = len(sections)
        if n == 0:
            return

        for i, sec in enumerate(sections):
            mean_t = sec.tension_score
            slope = self._tension_slope(sec.trajectory)
            is_first = (i == 0)
            is_last  = (i == n - 1)

            if mean_t < 0.25 and is_last and (not is_first or sec.start > 0.0):
                # Outro takes priority over intro when the section is at the end
                # and either there are multiple sections or it doesn't start at 0.
                sec.section_type = "outro"
            elif mean_t < 0.25 and is_first:
                sec.section_type = "intro"
            elif mean_t < 0.25 and is_last:
                sec.section_type = "outro"
            elif slope > 0.015:
                # Positive slope dominates: the section is actively building
                sec.section_type = "rising"
            elif mean_t > 0.75:
                # High mean tension: climactic peak
                sec.section_type = "climax"
            elif mean_t > 0.65 and abs(slope) < 0.015:
                # High but flat tension: sustained plateau
                sec.section_type = "plateau"
            elif slope < -0.015:
                # Negative slope: energy declining
                sec.section_type = "falling"
            elif mean_t < 0.35 and i > 0 and sections[i - 1].tension_score > 0.5:
                # Low tension after a high-tension section: deliberate release
                sec.section_type = "release"
            else:
                sec.section_type = "plateau"

            self._assign_tension_arc(sec)

        # --- Compute inter-section motion descriptors ---
        for i in range(len(sections) - 1):
            # Use the trailing edge of section i and the leading edge of section i+1
            before = sections[i].trajectory[-2:] if sections[i].trajectory else []
            after  = sections[i + 1].trajectory[:2] if sections[i + 1].trajectory else []
            motion = self._compute_motion(before, after)
            sections[i].motion_out    = motion
            sections[i + 1].motion_in = motion

    def _assign_tension_arc(self, sec: Section) -> None:
        """Label the internal tension contour of a single section in-place.

        Arc labels
        ----------
        - **building** — positive slope > 0.01 tension/second.  Tension is
          consistently rising through the section.
        - **releasing** — negative slope < -0.01.  Tension is consistently
          falling.
        - **peak** — flat slope but mean tension > 0.65.  The section sits
          at a high plateau.
        - **valley** — flat slope but mean tension < 0.30.  The section
          sits at a low, calm plateau.
        - **plateau** — default for near-flat sections in the mid-tension
          range.

        Parameters
        ----------
        sec:
            The Section to annotate.  ``sec.tension_arc`` is written
            in-place; all other fields are read-only here.
        """
        slope = self._tension_slope(sec.trajectory)
        mean_t = sec.tension_score
        if slope > 0.01:
            sec.tension_arc = "building"
        elif slope < -0.01:
            sec.tension_arc = "releasing"
        elif mean_t > 0.65:
            sec.tension_arc = "peak"
        elif mean_t < 0.30:
            sec.tension_arc = "valley"
        else:
            sec.tension_arc = "plateau"

    def _tension_slope(self, traj: List[TrajectoryPoint]) -> float:
        """Compute the linear slope of tension_score over time (tension/second).

        Uses numpy ``polyfit`` (degree 1) on the time-relative trajectory so
        that the returned slope is in units of (tension units) per second.

        Returns 0.0 for trajectories with fewer than 2 points or where all
        time values are identical.

        Parameters
        ----------
        traj:
            Ordered list of TrajectoryPoints.

        Returns
        -------
        float
            Slope in tension/second.  Positive = increasing tension; negative
            = decreasing.
        """
        if len(traj) < 2:
            return 0.0
        times = np.array([tp.time for tp in traj], dtype=np.float64)
        tensions = np.array([tp.tension_score for tp in traj], dtype=np.float64)
        # Express time relative to section start so the intercept is irrelevant
        t_rel = times - times[0]
        duration = t_rel[-1]
        if duration < 1e-6:
            return 0.0
        # Degree-1 polynomial fit; coeffs[0] is the slope in units / second
        coeffs = np.polyfit(t_rel, tensions, 1)
        return float(coeffs[0])

    def _compute_motion(
        self,
        before: List[TrajectoryPoint],
        after: List[TrajectoryPoint],
    ) -> SectionMotion:
        """Characterise a section boundary from the 2 points on each side.

        The transition is described along three dimensions:

        **type**
            - ``"abrupt"`` — large normalised delta (> 0.4) with very few
              transition points (≤ 3 total), indicating a hard cut.
            - ``"swell"`` — energy clearly rises (normalised delta > 0.2).
            - ``"fade"`` — energy clearly falls (normalised delta < -0.2).
            - ``"gradual"`` — smooth transition that doesn't qualify above.

        **rate**
            - ``"instant"`` — only 2 total points (1 + 1 on each side).
            - ``"fast"`` — 3–6 total points.
            - ``"gradual"`` — 7+ total points.

        **direction**
            Whether energy and/or brightness are rising, falling, or stable
            across the boundary.

        Parameters
        ----------
        before:
            The last 1–2 TrajectoryPoints of the preceding section.
        after:
            The first 1–2 TrajectoryPoints of the following section.

        Returns
        -------
        SectionMotion
            A fully populated SectionMotion.  Falls back to a neutral
            "stable / gradual / gradual" descriptor if either list is empty.
        """
        if not before or not after:
            return SectionMotion("stable", "gradual", "gradual")

        def mean_feat(pts, attr):
            return np.mean([getattr(p, attr) for p in pts])

        e_before = mean_feat(before, "energy")
        e_after  = mean_feat(after,  "energy")
        b_before = mean_feat(before, "brightness")
        b_after  = mean_feat(after,  "brightness")

        # Normalise deltas by the total span seen across all boundary points
        # so that the thresholds below are scale-independent.
        e_all = [p.energy for p in before + after]
        b_all = [p.brightness for p in before + after]
        e_span = max(max(e_all) - min(e_all), 1e-6)
        b_span = max(max(b_all) - min(b_all), 1e-6)
        delta_e_n = (e_after - e_before) / e_span
        delta_b_n = (b_after - b_before) / b_span
        max_delta = max(abs(delta_e_n), abs(delta_b_n))

        n_transition = len(before) + len(after)

        # --- Type ---
        if max_delta > 0.4 and n_transition <= 3:
            # Large jump with very few points between sections → abrupt cut
            motion_type = "abrupt"
        elif delta_e_n > 0.2:
            # Energy is clearly swelling into the new section
            motion_type = "swell"
        elif delta_e_n < -0.2:
            # Energy is clearly falling away from the outgoing section
            motion_type = "fade"
        else:
            motion_type = "gradual"

        # --- Rate ---
        # Determined purely by how many points we have to observe the change.
        # With only 1+1 points there is no transition — it's instant.
        if n_transition <= 2:
            rate = "instant"
        elif n_transition <= 6:
            rate = "fast"
        else:
            rate = "gradual"

        # --- Direction ---
        if delta_e_n > 0.1 or delta_b_n > 0.1:
            direction = "rising"
        elif delta_e_n < -0.1 or delta_b_n < -0.1:
            direction = "falling"
        else:
            direction = "stable"

        return SectionMotion(direction=direction, rate=rate, type=motion_type)

    # ------------------------------------------------------------------
    # Phase 4: full pipeline entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        filename: str,
        duration: float,
        trajectory: List[TrajectoryPoint],
        audio: np.ndarray,
        sr: int,
    ) -> NarrativeResult:
        """Full narrative pipeline: detect → classify → extract → prose → fingerprint.

        This is the primary public API for the narrative system.  It runs all
        four stages in sequence and returns a self-contained NarrativeResult
        that can be stored, compared, or exported directly.

        Parameters
        ----------
        filename:
            Basename of the source audio file (no directory component).
        duration:
            Total audio duration in seconds.
        trajectory:
            Ordered list of TrajectoryPoint snapshots (typically one every
            2 seconds) as produced by TrajectoryAnalyzer.
        audio:
            Raw audio time series as a float32 numpy array.
        sr:
            Sample rate of `audio` in Hz.

        Returns
        -------
        NarrativeResult
            Complete annotated result including prose narrative, sections,
            structure fingerprint, and 10-dim texture fingerprint.
        """
        # Stage 1 + 2: detect section boundaries and classify them
        sections = self.detect_sections(trajectory, duration)
        self._classify_sections(sections, duration)

        # Stage 3: run per-section mood/character extraction on raw audio slices
        self._extract_section_features(sections, audio, sr)

        # Stage 4: generate template-driven prose narrative
        narrative_text = self._generate_narrative(filename, sections)

        # Build compact structure fingerprint: [(section_type, tension_score), ...]
        structure_fp = [(s.section_type, s.tension_score) for s in sections]

        # Compute 10-dim L2-normalised texture fingerprint from trajectory statistics
        texture_fp = self._texture_fingerprint(trajectory)

        return NarrativeResult(
            filename=filename,
            duration=duration,
            narrative=narrative_text,
            sections=sections,
            trajectory=trajectory,
            structure_fingerprint=structure_fp,
            texture_fingerprint=texture_fp,
            similar_to=[],  # populated later by library-wide comparison pass
        )

    def _extract_section_features(
        self, sections: List[Section], audio: np.ndarray, sr: int
    ) -> None:
        """Run MoodAnalyzer + CharacterAnalyzer on a ≤20 s audio slice per section.

        Modifies each Section in-place, setting:
        - ``dominant_mood``     — top mood descriptor string
        - ``dominant_character`` — top character tag string
        - ``instruments``       — full list of detected character tags

        The audio slice is centred on the section midpoint and capped at 20 s
        so that feature extraction stays fast even for long sections.  If any
        analyser fails (import error, librosa error, etc.) the section keeps
        its "unknown" placeholder values and a warning is logged rather than
        propagating the exception.

        Parameters
        ----------
        sections:
            List of Section objects to annotate in-place.
        audio:
            Full audio time series.
        sr:
            Sample rate.
        """
        try:
            from audio_analysis.analysis.mood_analyzer import MoodAnalyzer
            from audio_analysis.analysis.character_analyzer import CharacterAnalyzer
            from audio_analysis.core.feature_extraction_base import FeatureExtractionCore

            mood_az = MoodAnalyzer()
            char_az = CharacterAnalyzer()
            # FeatureExtractionCore uses 'sample_rate' (not 'sr') in its constructor
            core = FeatureExtractionCore(sample_rate=sr)
        except Exception as exc:
            log.warning("Could not initialise mood/character analysers: %s", exc)
            return

        for sec in sections:
            # Build a ≤20 s slice centred on the section midpoint
            mid = (sec.start + sec.end) / 2.0
            slice_sec = min(20.0, sec.end - sec.start)
            t_start = max(0.0, mid - slice_sec / 2.0)
            t_end = min(len(audio) / sr, t_start + slice_sec)
            s_start, s_end = int(t_start * sr), int(t_end * sr)
            audio_slice = audio[s_start:s_end]

            # Skip slices that are too short to analyse reliably
            if len(audio_slice) < sr // 2:
                continue

            try:
                # Extract spectral and temporal features from the section slice.
                # We use the two lightweight methods instead of
                # extract_comprehensive_features() to avoid requiring a file_path
                # and duration argument.
                spectral_feats = core.extract_spectral_features(audio_slice, sr)
                temporal_feats = core.extract_temporal_features(audio_slice, sr)

                # Build the phase_data dict expected by analyze_mood()
                phase_data = {
                    "avg_energy":    temporal_feats.get("rms_mean", 0.0),
                    "avg_brightness": spectral_feats.get("spectral_centroid_mean", 0.0),
                    "avg_roughness": spectral_feats.get("zero_crossing_rate_mean", 0.0),
                    "onset_density": temporal_feats.get("onset_density", 0.0),
                    "duration":      slice_sec,
                }

                # analyze_mood() takes (phase_data, spectral_features) and returns
                # Tuple[List[str], Dict[str, float]]
                mood_list, _mood_scores = mood_az.analyze_mood(phase_data, spectral_feats)
                if mood_list:
                    sec.dominant_mood = mood_list[0]

                # analyze_character() takes a single spectral features dict and
                # returns Tuple[List[str], Dict[str, float]]
                char_list, _char_scores = char_az.analyze_character(spectral_feats)
                if char_list:
                    sec.instruments = char_list
                    sec.dominant_character = char_list[0]

            except Exception as exc:
                log.warning(
                    "Section feature extraction at %.1f s failed: %s", sec.start, exc
                )

    def _generate_narrative(self, filename: str, sections: List[Section]) -> str:
        """Template-driven prose generation targeting 3–6 sentences, ~80–120 words.

        Each section contributes at most one sentence (selected by cycling
        through the template list for that section_type).  When new character
        tags appear in a section that were not present in the previous section,
        a short instrument-description sentence is appended.  The loop stops
        after 6 sentences to keep the prose concise.  If fewer than 3 sentences
        have been generated once all sections are processed, filler sentences are
        appended so the output always meets the 30-word minimum.

        Parameters
        ----------
        filename:
            Source audio filename, used to extract a display title.
        sections:
            Classified, mood/character-annotated list of sections.

        Returns
        -------
        str
            Space-joined prose paragraph.
        """
        if not sections:
            return f"{filename} — insufficient data for narrative."

        # Use the filename stem (no extension) as the title in opening sentences
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        sentences: list = []
        prev_instruments: set = set()

        for i, sec in enumerate(sections):
            # Select the template for this section type, cycling if there are
            # more sections than templates
            templates = _SECTION_OPENERS.get(
                sec.section_type, _SECTION_OPENERS["plateau"]
            )
            template = templates[i % len(templates)]

            # Substitute all placeholders into the template
            sentence = template.format(
                title=title,
                time=self._format_time(sec.start),
                adverb=_ADVERBS.get(sec.tension_arc, "steadily"),
                intensity_word=_INTENSITY.get(sec.section_type, "sustained"),
                mood_phrase=sec.dominant_mood.replace("_", " "),
                arc_phrase=sec.tension_arc,
                feature_phrase=(
                    "brightness rises"
                    if sec.tension_arc in ("building", "peak")
                    else "textures dissolve"
                ),
            )
            sentences.append(sentence)

            # If new character tags appeared in this section (vs. the previous
            # one), append a brief instrument-description sentence.
            new_instr = set(sec.instruments) - prev_instruments
            if new_instr and i > 0:
                instr_str = (
                    ", ".join(sorted(new_instr)[:2]).replace("_", " ")
                )
                sentences.append(f"A {instr_str} character emerges here.")
            prev_instruments = set(sec.instruments)

            # Cap at 6 sentences to maintain conciseness
            if len(sentences) >= 6:
                break

        # Pad to at least 3 sentences so word-count assertions are met
        while len(sentences) < 3:
            sec = sections[min(len(sentences), len(sections) - 1)]
            sentences.append(
                f"The {sec.section_type} section at {self._format_time(sec.start)}"
                " continues the arc."
            )

        # Ensure the final prose meets the 30-word minimum.  When the template
        # sentences are short (single-word mood placeholders, etc.) we append a
        # brief closing summary drawn from the last section.
        joined = " ".join(sentences)
        if len(joined.split()) < 30:
            last = sections[-1]
            summary = (
                f"Overall, the piece spans {self._format_time(int(last.end))} and "
                f"moves through {len(sections)} section"
                f"{'s' if len(sections) != 1 else ''}, "
                f"with a {last.tension_arc} character that defines its final texture."
            )
            joined = joined + " " + summary

        return joined

    def _format_time(self, seconds: float) -> str:
        """Format a time in seconds as 'M:SS' (e.g. 93.0 → '1:33').

        Parameters
        ----------
        seconds:
            Time in seconds (non-negative float).

        Returns
        -------
        str
            Human-readable 'M:SS' string.
        """
        m, s = int(seconds) // 60, int(seconds) % 60
        return f"{m}:{s:02d}"

    def _texture_fingerprint(self, trajectory: List[TrajectoryPoint]) -> np.ndarray:
        """Compute a 10-dimensional L2-normalised texture fingerprint.

        The fingerprint summarises the piece's average and variability of five
        trajectory features:

          dims 0–1  energy          (mean, std)
          dims 2–3  brightness      (mean, std)
          dims 4–5  roughness       (mean, std)
          dims 6–7  tension_score   (mean, std)
          dim  8    chroma_spread   (mean only)
          dim  9    relative duration  min(total_seconds / 600, 1.0)

        The vector is L2-normalised so that cosine-similarity comparisons
        between tracks are straightforward.  Returns a zero vector for empty
        trajectories.

        Parameters
        ----------
        trajectory:
            Full-piece ordered list of TrajectoryPoint objects.

        Returns
        -------
        numpy.ndarray
            Shape (10,), dtype float32, L2-normalised (or all-zeros if the
            trajectory is empty or its norm is negligibly small).
        """
        if not trajectory:
            return np.zeros(10, dtype=np.float32)

        energies   = [tp.energy for tp in trajectory]
        brightness = [tp.brightness for tp in trajectory]
        roughness  = [tp.roughness for tp in trajectory]
        tensions   = [tp.tension_score for tp in trajectory]
        spreads    = [tp.chroma_spread for tp in trajectory]
        duration   = trajectory[-1].time - trajectory[0].time

        vec = np.array([
            np.mean(energies),    np.std(energies),
            np.mean(brightness),  np.std(brightness),
            np.mean(roughness),   np.std(roughness),
            np.mean(tensions),    np.std(tensions),
            np.mean(spreads),
            # Normalise duration to [0, 1] using 10 minutes as the reference maximum
            min(duration / 600.0, 1.0),
        ], dtype=np.float32)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def _peak_indices(self, scores: np.ndarray, threshold: float,
                      min_gap: int) -> List[int]:
        """Return indices of local maxima above `threshold` with minimum spacing.

        Only positions that are strictly local maxima (≥ both immediate
        neighbours) and satisfy the minimum-gap constraint against the most
        recently accepted peak are retained.

        Parameters
        ----------
        scores:
            1-D array of change scores.
        threshold:
            Minimum score to qualify as a peak.
        min_gap:
            Minimum number of positions between consecutive peaks.

        Returns
        -------
        List[int]
            Indices in ascending order.
        """
        peaks = []
        last_peak = -min_gap - 1
        for i in range(1, len(scores) - 1):
            if (scores[i] >= threshold
                    and scores[i] >= scores[i - 1]
                    and scores[i] >= scores[i + 1]
                    and i - last_peak >= min_gap):
                peaks.append(i)
                last_peak = i
        return peaks

    def _merge_short(self, times: List[float], min_sec: float) -> List[float]:
        """Remove intermediate boundaries that create gaps shorter than `min_sec`.

        Operates greedily from left to right: when the gap from the last
        accepted boundary to the current candidate is shorter than ``min_sec``
        *and* the candidate is not the final boundary, the candidate is
        dropped.  This naturally merges tiny sections into their right
        neighbour while always preserving the start (0) and end (duration)
        boundaries.

        Parameters
        ----------
        times:
            Sorted boundary times including 0.0 and duration.
        min_sec:
            Minimum allowed gap between consecutive boundaries.

        Returns
        -------
        List[float]
            Filtered boundary list, always containing at least the first
            and last elements of the input.
        """
        if len(times) <= 2:
            return times
        result = [times[0]]
        for t in times[1:]:
            if t - result[-1] < min_sec and t < times[-1]:
                # This gap is too short — skip the intermediate boundary
                continue
            result.append(t)
        # Guarantee the final boundary is always present
        if result[-1] != times[-1]:
            result.append(times[-1])
        return result

    def _split_long(self, times: List[float], trajectory: List[TrajectoryPoint],
                    max_sec: float) -> List[float]:
        """Recursively split any gap longer than `max_sec` at the local tension peak.

        For each gap [start, end] that exceeds ``max_sec``, the trajectory
        points that fall strictly inside the gap are examined.  The point
        with the highest absolute tension gradient (``np.gradient``) is
        chosen as the new split point, and the sub-problem is solved
        recursively so that the resulting sub-segments also respect the
        ceiling.

        If the gap contains fewer than 3 trajectory points, it is left
        unsplit (there is not enough information to find a meaningful
        boundary).

        Parameters
        ----------
        times:
            Sorted boundary times.
        trajectory:
            Full trajectory for the section being split.
        max_sec:
            Maximum allowed gap in seconds.

        Returns
        -------
        List[float]
            Expanded boundary list with all gaps ≤ ``max_sec`` (where
            enough trajectory data is available to split).
        """
        result = [times[0]]
        for i in range(len(times) - 1):
            start, end = times[i], times[i + 1]
            if end - start <= max_sec:
                result.append(end)
                continue
            # Find candidate split points strictly inside the gap
            sec_traj = [tp for tp in trajectory if start < tp.time < end]
            if len(sec_traj) < 3:
                # Not enough points to find a meaningful interior boundary
                result.append(end)
                continue
            tensions = np.array([tp.tension_score for tp in sec_traj], dtype=np.float32)
            # np.gradient gives local rate of change; pick maximum absolute gradient
            grad = np.abs(np.gradient(tensions))
            # Guard: if the entire segment is flat (all gradients are zero),
            # there is no meaningful change point to split on.  Splitting anyway
            # would cause np.argmax to return index 0, producing a 2-second
            # micro-section that violates MIN_SECTION_SEC.
            if grad.max() == 0.0:
                result.append(end)
                continue
            split_idx = int(np.argmax(grad))
            split_time = sec_traj[split_idx].time
            # Recurse so that the two sub-segments are also checked
            sub_times = self._split_long([start, split_time, end], trajectory, max_sec)
            result.extend(sub_times[1:])
        return result
