"""
Re-score character tags in an existing analysis JSON using the current
CharacterAnalyzer code (including the stereo_width fix and updated thresholds).

Usage:
    python3 rescore_character_tags.py [path/to/full_library_data.json]

Writes an updated JSON file alongside the input (with _rescored suffix).
"""
import json
import sys
from pathlib import Path

# Ensure venv site-packages are on the path if invoked directly
sys.path.insert(0, str(Path(__file__).parent))

from audio_analysis.analysis.character_analyzer import CharacterAnalyzer


def rescore(json_path: Path) -> Path:
    analyzer = CharacterAnalyzer()

    with open(json_path) as f:
        data = json.load(f)

    changed = 0
    for track in data.get("tracks", []):
        old_tags = track.get("character_tags", "")
        tags, primary, scores = analyzer.analyze_track_character(track)
        new_tags = ", ".join(tags)
        track["character_tags"] = new_tags
        track["primary_character"] = primary
        if old_tags != new_tags:
            changed += 1
            fn = track.get("filename", "?")[:55]
            print(f"  {fn}: {old_tags!r} → {new_tags!r}")

    print(f"\n{changed} tracks changed out of {len(data['tracks'])}")

    out_path = json_path.with_name(json_path.stem + "_rescored.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        # Auto-discover latest
        base = Path("generated/full_library")
        candidates = sorted(
            base.glob("audio_analysis_*/reports/full_library_data.json"), reverse=True
        )
        if not candidates:
            print("No analysis JSON found — run run_full_analysis.py first.")
            sys.exit(1)
        path = candidates[0]

    print(f"Re-scoring character tags from: {path}")
    rescore(path)
