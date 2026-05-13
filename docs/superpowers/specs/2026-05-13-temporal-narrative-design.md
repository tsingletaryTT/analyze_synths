# Temporal Narrative Analysis Design

**Date:** 2026-05-13
**Depends on:** `2026-05-13-tt-stft-kernel-design.md` (SpectrogramChunk interface)

## Goal

Add a temporal narrative analysis layer to analyze-synths that captures the ebb and flow
of a full piece — not just aggregate statistics from the first few seconds, but a running
account of how tension builds and releases, how texture evolves, what instruments appear
and when, and how the piece relates to others in the library.

## Architecture

```
SpectrogramChunk stream (from TTStftKernel)
    │
    ▼
TrajectoryAnalyzer          audio_analysis/core/trajectory_analysis.py
    │  2-second windows
    │  → List[TrajectoryPoint]
    ▼
NarrativeAnalyzer           audio_analysis/core/narrative_analysis.py
    │  change-point detection → sections
    │  per-section: mood, character, instruments (real re-extraction)
    │  motion descriptors
    │  narrative prose generation
    │  → NarrativeResult
    ▼
CrossPieceSimilarity        audio_analysis/analysis/cross_piece_similarity.py
    │  structure + texture fingerprints
    │  → similar_to list
    ▼
NarrativeExporter           audio_analysis/exporters/narrative_exporter.py
    │  narrative.json, {name}_narrative.md
    └── similarity_matrix.csv (library-level)
```

## Data Structures

```python
@dataclass
class TrajectoryPoint:
    time: float             # center time of window (seconds)
    energy: float           # RMS of window
    brightness: float       # spectral centroid (Hz)
    roughness: float        # mean abs adjacent-bin diff
    zcr: float              # zero-crossing rate
    chroma_peak: str        # dominant pitch class ("C", "F#", etc.)
    chroma_spread: float    # entropy of chroma distribution, 0-1
    tension_score: float    # 0-1 composite (see formula below)

@dataclass
class SectionMotion:
    direction: str   # "rising" | "falling" | "stable" | "oscillating"
    rate: str        # "fast" | "gradual" | "instant"
    type: str        # "abrupt" | "gradual" | "fade" | "swell"

@dataclass
class Section:
    start: float            # seconds
    end: float              # seconds
    section_type: str       # "intro"|"rising"|"plateau"|"climax"|"falling"|"release"|"outro"
    tension_arc: str        # "building"|"peak"|"plateau"|"releasing"|"valley"
    motion_in: SectionMotion
    motion_out: SectionMotion
    dominant_mood: str      # from MoodAnalyzer on section audio
    dominant_character: str # from CharacterAnalyzer on section audio
    instruments: List[str]  # character tags present in this section
    tension_score: float    # mean tension_score over section trajectory
    trajectory: List[TrajectoryPoint]  # 2s-resolution points within section

@dataclass
class NarrativeResult:
    filename: str
    duration: float
    narrative: str               # prose description
    sections: List[Section]
    trajectory: List[TrajectoryPoint]  # full-piece trajectory
    structure_fingerprint: List[Tuple[str, float]]  # [(section_type, tension_score), ...]
    texture_fingerprint: np.ndarray   # 10-dim vector
    similar_to: List[str]        # filenames, populated after library comparison
```

## TrajectoryAnalyzer

**Input:** `SpectrogramChunk` (full file via `process_file`)
**Output:** `List[TrajectoryPoint]`

Window size: 2 seconds = `ceil(2 * sr / hop_length)` frames ≈ 86 frames at sr=22050, hop=512.

Per window (using `SpectrogramChunk.mag` and `.mel`):
```python
energy     = sqrt(mean(mag_window ** 2))                       # RMS from magnitude
brightness = sum(freq_weights * mag_window) / sum(mag_window)  # spectral centroid
roughness  = mean(abs(diff(mag_window, axis=-1)))              # spectral irregularity
zcr        = (computed from raw audio window, not spectrogram) # from audio chunk
chroma     = mag_window @ chroma_filter                        # reuse existing filter
chroma_peak = musical_key_names[argmax(mean(chroma, axis=0))]
chroma_spread = -sum(p * log(p+1e-6)) / log(12)               # normalized entropy, 0-1
```

**Tension score formula:**
```
tension = 0.35 * energy_norm          # high energy = more tense
        + 0.25 * roughness_norm        # spectral roughness = more tense
        + 0.20 * (1 - chroma_spread)   # narrow chroma (one dominant note) = more tense
        + 0.20 * zcr_norm              # fast zero-crossings = more tense
```
Each component is min-max normalized over the full file trajectory before combining.
The weights are empirically set for synthesizer music; they can be tuned per-piece.

`TrajectoryAnalyzer` requires raw audio in addition to the spectrogram (for ZCR).
It takes both as inputs.

## NarrativeAnalyzer

### Step 1: Change-point detection

Detect section boundaries from the trajectory using a sliding-window change score:

```python
def change_score(t: int, trajectory: List[TrajectoryPoint], window=5) -> float:
    # Compare mean of window before t vs window after t
    # across energy, brightness, roughness, tension_score
    before = trajectory[max(0, t-window):t]
    after  = trajectory[t:min(len(trajectory), t+window)]
    return mean([abs(mean_after[f] - mean_before[f]) / range[f]
                 for f in ['energy', 'brightness', 'roughness', 'tension_score']])
```

Peaks in `change_score` where the score exceeds 0.15 (tunable threshold) and sections
are at least 15s apart become section boundaries. Minimum section duration enforced: 15s.
Maximum: 60s (longer stretches are split at the point of highest internal change).

### Step 2: Section classification

For each detected section, classify `section_type` from its trajectory shape:

| Trajectory shape | section_type |
|-----------------|--------------|
| tension_score < 0.25 and first section | intro |
| tension_score < 0.25 and last section | outro |
| tension rising slope > 0.015/s | rising |
| tension high (> 0.65) and stable | plateau |
| tension > 0.75 at peak | climax |
| tension falling slope > 0.015/s | falling |
| tension < 0.35 after high section | release |
| none of the above | plateau |

`tension_arc` is derived from the slope of tension within the section:
- slope > +0.01/s → "building"
- slope < -0.01/s → "releasing"
- tension > 0.65 → "peak"
- tension < 0.30 → "valley"
- otherwise → "plateau"

### Step 3: Motion descriptors

For each section boundary, compute `SectionMotion` for `motion_out` (of section N) and
`motion_in` (of section N+1) from the 4-second window straddling the boundary:

```python
delta_energy    = energy_after_mean - energy_before_mean
delta_brightness = brightness_after_mean - brightness_before_mean
max_delta       = max(abs(delta_energy_norm), abs(delta_brightness_norm))

type = "abrupt"  if max_delta > 0.4 and transition_frames < 4  else
       "swell"   if delta_energy > 0.2                          else
       "fade"    if delta_energy < -0.2                         else
       "gradual"

rate = "instant" if transition_frames < 2  else
       "fast"    if transition_frames < 8  else
       "gradual"
```

### Step 4: Per-section mood and character (real re-extraction)

For each section, extract a 20-second audio slice centered on the section midpoint
(or the full section if shorter than 20s). Run `MoodAnalyzer` and `CharacterAnalyzer`
on this slice. This replaces the whole-file approximation currently used in phase analysis.

The existing `FeatureExtractionCore.extract_comprehensive_features()` is called on the
audio slice to get real per-section feature values.

### Step 5: Narrative prose generation

Template-driven. The narrative assembles sentences from section data:

```python
SECTION_OPENERS = {
    "intro":   ["{title} opens with {mood_phrase} — {texture_phrase}.",
                "The piece begins {adverb}, {mood_phrase}."],
    "rising":  ["Around {time}, {transition_phrase} as {feature_phrase}.",
                "The texture {direction_verb} from {time}."],
    "climax":  ["A {intensity_word} peak arrives at {time} — {mood_phrase}.",
                "By {time}, the tension has {arc_phrase}."],
    "release": ["The intensity {release_verb} after {time}.",
                "{mood_phrase} settles in from {time} onward."],
    "outro":   ["The piece closes {adverb} — {mood_phrase}.",
                "A long {mood_phrase} resolution carries through to the end."],
    ...
}

def format_time(t: float) -> str:  # 93.0 → "1:33"
```

Instrument mentions are woven in when a new character appears that wasn't in the previous
section: "an fm character emerges briefly at the peak."

The prose is a single paragraph, 3–6 sentences, targeting 80–120 words.

## CrossPieceSimilarity

**Structure fingerprint**: `[(section_type, tension_score), ...]` for all sections.
Encoded as a fixed-length vector by binning tension_score into 5 levels and one-hot
encoding `(section_type, tension_bin)` → flatten → L2 normalize. Length: 7 types × 5
bins = 35 dimensions.

**Texture fingerprint**: `[mean_energy, std_energy, mean_brightness, std_brightness,
mean_roughness, std_roughness, mean_tension, std_tension, mean_chroma_spread, duration_norm]`
→ 10 dimensions, L2 normalized.

**Similarity score**: `0.6 * cosine(structure_a, structure_b) + 0.4 * cosine(texture_a, texture_b)`

`similar_to` is populated in `ParallelAudioAnalyzer.analyze_directory()` after all files
are processed, by computing pairwise similarity and taking top-3 for each file.

## NarrativeExporter

### `narrative.json`

```json
{
  "filename": "2022-09-25 while madeline sleeps.aif",
  "duration": 642.0,
  "narrative": "Opens with an extended ambient intro...",
  "sections": [ { ...Section fields... } ],
  "trajectory": [ { ...TrajectoryPoint fields... } ],
  "similar_to": ["2020 cricketeering.aif", "..."]
}
```

One file per analyzed track: `{base_name}_narrative.json`.

### `{base_name}_narrative.md`

Human-readable. Sections:
1. Narrative paragraph
2. Section breakdown table (start | end | type | tension | mood | instruments)
3. Similar pieces

### `similarity_matrix.csv`

Library-level export (only when analyzing a directory, not a single file):
```
filename_a, filename_b, similarity_score
"while madeline sleeps.aif", "cricketeering.aif", 0.847
...
```

## Integration Points

### `ParallelAudioAnalyzer.analyze_directory()`

After existing feature extraction, add:
```python
# 1. Run TTStftKernel.process_file() for each file (or reuse spectrogram from JAX path)
# 2. Run TrajectoryAnalyzer per file
# 3. Run NarrativeAnalyzer per file → NarrativeResult
# 4. After all files: CrossPieceSimilarity.compute_library(results) → populate similar_to
# 5. Export via NarrativeExporter
```

### MCP server: `query_narrative` tool

```python
@mcp.tool()
def query_narrative(directory: str, filename: str, query: str) -> dict:
    """
    Answer questions about a piece's temporal structure.
    query examples: "what's happening at 3:20", "find pieces similar to X",
                    "describe the emotional arc", "when does the climax occur"
    """
```

## Files Changed

| File | Change |
|------|--------|
| `audio_analysis/core/trajectory_analysis.py` | New |
| `audio_analysis/core/narrative_analysis.py` | New |
| `audio_analysis/analysis/cross_piece_similarity.py` | New |
| `audio_analysis/exporters/narrative_exporter.py` | New |
| `audio_analysis/api/analyzer.py` | Add narrative pipeline call |
| `audio_analysis/api/parallel_analyzer.py` | Add narrative pipeline + similarity |
| `audio_analysis/api/mcp_server.py` | Add query_narrative tool |
| `audio_analysis/exporters/markdown_exporter.py` | Include narrative section |
| `tests/test_trajectory_analysis.py` | New |
| `tests/test_narrative_analysis.py` | New |
| `tests/test_cross_piece_similarity.py` | New |

## Testing Plan

1. **Trajectory accuracy**: on a synthetic file with a known energy arc (sine ramp),
   assert `TrajectoryAnalyzer` produces monotonically rising energy values. Assert
   tension_score on high-roughness noise segment > 0.6.

2. **Section detection**: on a synthetic file with an abrupt energy jump at 30s,
   assert a section boundary is detected within ±4 seconds of the jump.

3. **Section type classification**: construct trajectories with known shapes (arch,
   ramp, plateau); assert correct `section_type` for each.

4. **Narrative prose**: assert narrative is non-empty, contains the filename, and
   mentions at least one section type for each real file in `~/samples`.

5. **Similarity**: for 3 synthetic files where file A and B share the same section
   sequence and file C does not, assert `similarity(A, B) > similarity(A, C)`.

6. **Export round-trip**: assert `narrative.json` is valid JSON, contains all required
   fields, and `similar_to` is populated after library analysis.

7. **Integration**: run `ParallelAudioAnalyzer` on `~/samples` (8 files), assert all
   8 produce `NarrativeResult` with at least 3 sections each.
