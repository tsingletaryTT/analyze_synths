# audio_analysis/exporters/narrative_exporter.py
"""
NarrativeExporter: write per-file {base}_narrative.json + {base}_narrative.md
and a library-level similarity_matrix.csv.

Per-file outputs capture the full structured narrative analysis in both
machine-readable JSON (for downstream processing and MCP queries) and
human-readable Markdown (for composers browsing their library).

The library-level similarity_matrix.csv records all pairwise similarity
relationships discovered by CrossPieceSimilarity so that users can
quickly find related tracks.
"""
import dataclasses
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from audio_analysis.core.narrative_types import NarrativeResult, Section, TrajectoryPoint

log = logging.getLogger(__name__)


class NarrativeExporter:
    """Export NarrativeResult objects to JSON, Markdown, and CSV formats.

    Each call to export_narrative() produces two files:
    - ``{stem}_narrative.json``: complete structured data, JSON-serialisable
    - ``{stem}_narrative.md``:   human-readable Markdown report

    export_similarity_matrix() produces one library-wide file:
    - ``similarity_matrix.csv``: one row per (filename_a, filename_b) pair

    Parameters
    ----------
    output_dir : Path
        Directory where all output files are written.  Created automatically
        if it does not already exist.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_narrative(self, result: NarrativeResult) -> dict:
        """Write {base}_narrative.json and {base}_narrative.md.

        The base name is derived from the result's filename stem so that
        "my piece.aif" produces "my piece_narrative.json" and
        "my piece_narrative.md" — spaces in filenames are preserved.

        Parameters
        ----------
        result : NarrativeResult
            The fully-populated narrative result to serialise.

        Returns
        -------
        dict
            ``{"json": str(json_path), "md": str(md_path)}``
        """
        base = Path(result.filename).stem
        json_path = self.output_dir / f"{base}_narrative.json"
        md_path   = self.output_dir / f"{base}_narrative.md"

        self._write_json(result, json_path)
        self._write_md(result, md_path)

        log.debug("Exported narrative for %s → %s, %s", result.filename, json_path, md_path)
        return {"json": str(json_path), "md": str(md_path)}

    def export_similarity_matrix(self, results: List[NarrativeResult]) -> str:
        """Write similarity_matrix.csv for the full library.

        The CSV has three columns: ``filename_a``, ``filename_b``, and
        ``similar_to``.  One row is written for every (source, target) pair
        where target appears in source.similar_to.

        Parameters
        ----------
        results : list of NarrativeResult
            All NarrativeResult objects in the library.  Each result's
            ``similar_to`` list contributes one row per entry.

        Returns
        -------
        str
            Absolute path to the written CSV file.
        """
        csv_path = self.output_dir / "similarity_matrix.csv"
        rows = ["filename_a,filename_b,similar_to"]
        for r in results:
            for other in r.similar_to:
                # Quote both fields to handle filenames with commas or spaces.
                rows.append(f'"{r.filename}","{other}"')
        csv_path.write_text("\n".join(rows))
        log.debug("Exported similarity matrix → %s (%d pairs)", csv_path, len(rows) - 1)
        return str(csv_path)

    # ------------------------------------------------------------------
    # Private writers
    # ------------------------------------------------------------------

    def _write_json(self, result: NarrativeResult, path: Path) -> None:
        """Serialise NarrativeResult to indented JSON at path.

        numpy.ndarray (texture_fingerprint) is converted via .tolist() so
        that standard json.dumps can handle it without a custom encoder.
        structure_fingerprint contains plain Python tuples which json.dumps
        serialises as JSON arrays — the consumer must treat them as 2-element
        arrays [section_type, tension_score].
        """
        data = {
            "filename": result.filename,
            "duration": result.duration,
            "narrative": result.narrative,
            "sections": [self._section_to_dict(s) for s in result.sections],
            "trajectory": [self._tp_to_dict(tp) for tp in result.trajectory],
            # structure_fingerprint: list of [section_type, tension_score] pairs
            "structure_fingerprint": result.structure_fingerprint,
            # texture_fingerprint: 10-dim float32 array → plain list for JSON
            "texture_fingerprint": result.texture_fingerprint.tolist(),
            "similar_to": result.similar_to,
        }
        path.write_text(json.dumps(data, indent=2))

    def _write_md(self, result: NarrativeResult, path: Path) -> None:
        """Write a human-readable Markdown report at path.

        The report has three sections:
        1. Narrative prose paragraph
        2. Sections table (start | end | type | tension | mood | instruments)
        3. Similar pieces bullet list (omitted when similar_to is empty)
        """
        lines = [
            f"# {result.filename}",
            "",
            "## Narrative",
            "",
            result.narrative,
            "",
            "## Sections",
            "",
            "| Start | End | Type | Tension | Mood | Instruments |",
            "|-------|-----|------|---------|------|-------------|",
        ]
        for sec in result.sections:
            start_str = self._fmt(sec.start)
            end_str   = self._fmt(sec.end)
            # Limit instrument list to three entries to keep the table readable.
            instr = ", ".join(sec.instruments[:3]) if sec.instruments else "—"
            lines.append(
                f"| {start_str} | {end_str} | {sec.section_type} "
                f"| {sec.tension_score:.2f} | {sec.dominant_mood} | {instr} |"
            )
        if result.similar_to:
            lines += ["", "## Similar Pieces", ""]
            for name in result.similar_to:
                lines.append(f"- {name}")
        path.write_text("\n".join(lines))

    def _section_to_dict(self, sec: Section) -> dict:
        """Convert a Section to a plain dict suitable for JSON serialisation.

        SectionMotion is converted via dataclasses.asdict() which handles
        all plain-type fields automatically.  TrajectoryPoints within the
        section are converted via _tp_to_dict().
        """
        return {
            "start": sec.start,
            "end": sec.end,
            "section_type": sec.section_type,
            "tension_arc": sec.tension_arc,
            "motion_in": dataclasses.asdict(sec.motion_in),
            "motion_out": dataclasses.asdict(sec.motion_out),
            "dominant_mood": sec.dominant_mood,
            "dominant_character": sec.dominant_character,
            "instruments": sec.instruments,
            "tension_score": sec.tension_score,
            "trajectory": [self._tp_to_dict(tp) for tp in sec.trajectory],
        }

    def _tp_to_dict(self, tp: TrajectoryPoint) -> dict:
        """Convert a TrajectoryPoint to a plain dict for JSON serialisation.

        All fields are plain Python scalars (float or str) so no special
        conversion is needed beyond key renaming to match the JSON schema.
        """
        return {
            "time": tp.time,
            "energy": tp.energy,
            "brightness": tp.brightness,
            "roughness": tp.roughness,
            "zcr": tp.zcr,
            "chroma_peak": tp.chroma_peak,
            "chroma_spread": tp.chroma_spread,
            "tension_score": tp.tension_score,
        }

    @staticmethod
    def _fmt(seconds: float) -> str:
        """Format a duration in seconds as ``M:SS`` for Markdown tables.

        Examples
        --------
        >>> NarrativeExporter._fmt(0.0)
        '0:00'
        >>> NarrativeExporter._fmt(90.5)
        '1:30'
        """
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"
