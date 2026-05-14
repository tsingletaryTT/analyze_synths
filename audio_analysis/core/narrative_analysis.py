# audio_analysis/core/narrative_analysis.py
"""
NarrativeAnalyzer: section detection, classification, motion descriptors,
per-section mood/character, and prose generation.

The pipeline has three stages:

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

Usage:
    az = NarrativeAnalyzer()
    sections = az.detect_sections(trajectory, duration)
    az._classify_sections(sections, duration)
"""
import math
from typing import List

import numpy as np

from audio_analysis.core.narrative_types import (
    NarrativeResult, Section, SectionMotion, TrajectoryPoint,
)

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
            split_idx = int(np.argmax(grad))
            split_time = sec_traj[split_idx].time
            # Recurse so that the two sub-segments are also checked
            sub_times = self._split_long([start, split_time, end], trajectory, max_sec)
            result.extend(sub_times[1:])
        return result
