# analyze_synths

**A composer's experiment in teaching hardware to hear music.**

This started as a personal tool for making sense of a growing library of synthesizer pieces. It became a journey through custom DSLs, Blackhole silicon, JAX kernels, and temporal narrative analysis. Here's the story.

---

## The Problem

I write a lot of synthesizer music — long-form ambient pieces, generative drones, evolving textures. After a few years the library was big enough that I couldn't remember what was in it. Which tracks share a character? Which one would flow naturally into which other? Is that patch I made in 2021 "crystalline" or "granular"? Is there a climax around the 4-minute mark or does the tension just plateau?

Standard audio analysis tools give you BPM and key. Not very useful when the track has no tempo and the key is ambiguous.

So I built something that speaks a different language: *spacey, organic, droning, prismatic, cavernous, velvet*. Creative vocabulary for creative work.

---

## v1: Creative Feature Extraction

The first version was a monolithic script around librosa. It extracted 80+ features per track — spectral centroid, roughness, chroma spread, zero-crossing rate, spectral bandwidth — and mapped them through empirically-tuned thresholds to a vocabulary of 176 descriptors:

**9 core moods**: spacey, organic, synthetic, oozy, pensive, tense, exuberant, glitchy, chaos

**8 extended moods**: ethereal, atmospheric, crystalline, warm, melodic, driving, percussive, droning

**100 advanced moods**: velvet, gossamer, tempestuous, cavernous, sforzando, fractal, volcanic, and 93 more

**59 character tags**: analog_synth, granular_synth, mellotron, pad_synth, reverbed, tape_saturated...

It also did K-means clustering across the library, detected musical phases (intro/development/climax/conclusion), and generated optimal listening sequences using key compatibility and energy arc principles.

This worked well. The results felt right. The problem was speed.

---

## v2: The Architecture Problem

When the library grew to 100+ files — many of them 10-minute ambient pieces — the monolithic script started hitting memory walls. STFT on 11 minutes of audio at 22050 Hz allocates a dense complex matrix before any analysis begins. Librosa's `stft()` is not streaming-friendly.

The solution was to redesign around streaming chunks and express the core computation in a form that hardware could accelerate.

The key insight: **STFT is just two matrix multiplications**. A DFT of a windowed frame is:

```
cos_proj = frame @ cos_basis   # (n_fft,) × (n_fft, n_freqs) → (n_freqs,)
sin_proj = frame @ sin_basis
mag = sqrt(cos_proj² + sin_proj² + ε)
```

And then the mel filterbank is a third:

```
mel = mag @ mel_filter   # (n_freqs,) × (n_freqs, n_mels) → (n_mels,)
```

Three matmuls, precomputed static matrices, pure float32. This is exactly what Tenstorrent Tensix cores are designed to do.

---

## Plan 1: TTStftKernel and the TT-Lang DSL

The first hardware target was the TT-Lang simulator — a Python DSL for writing Tensix dataflow kernels that can run on a local simulator before touching real hardware.

### The TT-Lang kernel design

TT-Lang kernels (`@ttl.operation`) run on a (grid_x, grid_y) array of Tensix cores. Each core has L1 SRAM for staging tiles, a DRAM interface, and dataflow buffers (DFBs) for producer-consumer communication between reader, compute, and writer threads.

The STFT required two separate `@ttl.operation` kernels because a DataflowBuffer cannot be simultaneously written by compute and read by the reader in the same operation:

**Kernel 1 — `_stft_compute_mag`**: frames × DFT basis → magnitude
```
for ft in range(FT):           # frame tiles
    for nt in range(NT):       # frequency tiles
        for kt in range(KT):   # K-reduction (n_fft tiles)
            c_acc += frame[ft,kt] @ cos[kt,nt]
            s_acc += frame[ft,kt] @ sin[kt,nt]
        mag[ft,nt] = sqrt(c_acc² + s_acc²)
```

**Kernel 2 — `_stft_compute_mel`**: magnitude × mel filterbank → mel spectrogram
```
for ft in range(FT):
    for mt in range(MT):       # mel band tiles
        for nt in range(NT):   # N-reduction (n_freqs tiles)
            acc += mag[ft,nt] @ mel[nt,mt]
```

A critical constraint: reader and compute threads must iterate in **identical loop order** or tiles arrive out of sequence and produce silently wrong results.

All tensor dimensions must be multiples of TILE=32:
- n_fft=2048: already aligned (64 tiles)
- n_mels=128: already aligned (4 tiles)
- n_freqs=1025: **not aligned** → padded to 1056 (33×32)
- n_frames: varies → padded to nearest multiple of 32

### Streaming design

`TTStftKernel` processes audio in 30-second chunks with 2-second overlap. The overlap prevents edge artifacts at chunk boundaries. Overlap frames are discarded from subsequent chunks using exact global-sample-center arithmetic rather than a fixed discard count — this gives ±1 frame accuracy regardless of audio length or chunk size.

The dispatch chain:
1. JAX PJRT hardware (if TT device detected)
2. TT-Lang simulator (if sim importable, no HW)
3. NumPy fallback (always available)

If a path fails on first use, its flag flips to False and subsequent chunks skip the failed import entirely.

---

## The Hardware Journey

The simulator path worked. The NumPy fallback was our ground truth. Then the actual hardware arrived.

**Four P300C Blackhole chips** in a P150X4 mesh configuration. All four confirmed working via `tt-smi`. Getting JAX to speak to them was not straightforward.

### What didn't work

**tt-metal Python env**: The `open_device()` call was hardcoded for a 2-chip P300. The 4-chip mesh triggered a crash.

**tt-xla venv**: An older PJRT plugin binary had an undefined symbol in the mlir namespace. The shared object loaded but JAX couldn't initialize the device.

**Every path that looked obvious**: Wrong.

### What worked

The correct stack:
```
/home/ttuser/p300c-xla-test/lib/python3.12/site-packages   ← JAX + PJRT support libs
/home/ttuser/tt-xla/python_package/pjrt_plugin_tt/pjrt_plugin_tt.so  ← plugin binary
__editable___pjrt_plugin_tt_0_1_260226_dev_c67d612a_finder  ← editable install finder
```

With `jax.config.update("jax_platform_name", "tt")`, `jax.devices("tt")` returns all 4 chips. The mesh is real.

### The JAX PJRT kernel

Once the environment was found, the kernel became simple. JAX's `@jax.jit` compiles the computation graph for the TT backend on first call, then reuses the binary:

```python
@jax.jit
def _stft_jit(frames_j, cos_j, sin_j, mel_j):
    cos_proj = jnp.matmul(frames_j, cos_j)
    sin_proj = jnp.matmul(frames_j, sin_j)
    mag_j    = jnp.sqrt(cos_proj * cos_proj + sin_proj * sin_proj + 1e-8)
    mel_o    = jnp.matmul(mag_j, mel_j)
    return mag_j, mel_o
```

**JIT shape strategy**: n_frames varies per audio file, but JIT recompiles on shape change. Solution: pad all chunks to `_STANDARD_N_FRAMES_PAD = 1376` — the frame count for a full 30s+2s chunk at sr=22050, hop=512. One JIT binary serves all standard chunks. Only pathologically long chunks (edge case) would trigger recompilation.

**Basis matrix caching**: The cos/sin/mel matrices are precomputed at `TTStftKernel.__init__()` and transferred to TT device memory exactly once per kernel instance. The `_DEVICE_BASES` dict keyed by `id(kernel)` holds the device-resident tensors. Subsequent chunks reuse them — only the audio frames transfer each call.

### Precision note

The hardware path stays in float32 throughout. NumPy's reference path uses float64 intermediate arithmetic (the `1e-8` literal upcasts the sqrt operands). Over K=2048 accumulation steps, this produces ~1.2% mean relative error — fundamental float32 behavior, not a bug. The parity test allows 2%.

---

## Benchmark Results

Test library: 8 synthesizer pieces, 2.5–11.5 minutes each, 59.8 minutes total audio.

### Per-file hardware timing (JIT warmed up)

```
╔════════════════════════════════════════════════════════
║  File                                    Duration  HW time   RTF
╠════════════════════════════════════════════════════════
║  2020 cricketeering.aif                   6.3 min   0.10 s   3693x
║  2021 12 20 chiminy biscuits.aif          5.5 min   0.08 s   4013x
║  2021 melody partially resolute tetrax.   2.5 min   0.04 s   3497x
║  2022 06 05 negative 60.aif               4.2 min   0.06 s   3880x
║  2022 09 17 caving in.aif                 9.5 min   0.14 s   4074x
║  2022-09-25 while madeline sleeps.aif    10.7 min   0.16 s   3997x
║  2022-12-14 a quatrax cosmos memory.aif  11.5 min   0.18 s   3941x
║  2023 04 13 centers.aif                   9.6 min   0.14 s   3983x
╠════════════════════════════════════════════════════════
║  Total                                   59.8 min    0.9 s   3931x
╚════════════════════════════════════════════════════════
```

### Comparison

```
╔═══════════════════════════════════════════════
║  Backend    Time     Real-time factor
╠═══════════════════════════════════════════════
║  TT HW      0.9 s    3,931x
║  NumPy      2.9 s    1,256x
╠═══════════════════════════════════════════════
║  Speedup    3.2x
╚═══════════════════════════════════════════════
```

60 minutes of audio analyzed in under a second. The bottleneck has moved entirely off the STFT kernel.

---

## Plan 2: Temporal Narrative Analysis

Fast STFT opened a door. If we can compute spectrogram features for an entire library in under a second, we can afford much richer temporal analysis.

The question became: **what does this piece do over time?**

Not just "this track is spacey and droning" but "it opens with diffuse tension, crystallizes around the 3-minute mark, briefly destabilizes, then resolves into a warm drone for the final 90 seconds."

### TrajectoryAnalyzer

The trajectory is a time series of `TrajectoryPoint` structs, one per 2-second window:

```python
@dataclass
class TrajectoryPoint:
    time: float
    energy: float
    brightness: float
    roughness: float
    zcr: float
    chroma_peak: float
    chroma_spread: float
    tension_score: float
```

Tension is computed as a weighted combination of normalized features:

```
tension = 0.35 * energy_norm
        + 0.25 * roughness_norm
        + 0.20 * (1 - chroma_spread_norm)   # tonal focus = more tension
        + 0.20 * zcr_norm
```

This gives a single 0–1 value that tracks when a piece is building, releasing, or holding.

### NarrativeAnalyzer

Change-point detection finds section boundaries where the trajectory shifts meaningfully (threshold=0.15, minimum section length 15s, maximum 60s). Each section is classified into one of 7 types by its tension arc shape: whether tension rises, falls, holds high, holds low, peaks, valleys, or fluctuates.

Each section gets:
- Dominant mood (from MoodAnalyzer on the section's average features)
- Dominant character tag (from CharacterAnalyzer)
- Instruments/textures detected
- Tension arc summary (direction, rate, type)
- motion_in and motion_out descriptors for transitions

The section analysis feeds a template-driven prose generator that produces readable English:

> *"The opening movement establishes an atmospheric texture with crystalline character — tension is low and stable. Around 3:14, roughness increases sharply and chroma focus tightens: the piece enters its most tense passage. The climax sustains for 47 seconds before a measured release brings it back to the opening character."*

### CrossPieceSimilarity

Each `NarrativeResult` carries two fingerprints:

- **structure_fingerprint** (35 dims): section-count, duration, tension stats, arc shape distribution, phase timing
- **texture_fingerprint** (10 dims): average spectral features across the piece

Similarity between two pieces:

```
score = 0.6 * cosine(structure_A, structure_B)
      + 0.4 * cosine(texture_A, texture_B)
```

Structure dominates because two pieces can sound different but move the same way — and those structural siblings are the interesting ones to sequence together.

The `compute_library()` call populates `similar_to` for every piece with its top-3 neighbors.

### MCP Tool

The `query_narrative` MCP tool makes this accessible to AI assistants without re-running analysis:

```python
query_narrative(directory="/path/to/library", filename="cricketeering.aif", query="what happens at 3:30")
query_narrative(directory=..., filename=..., query="describe the emotional arc")
query_narrative(directory=..., filename=..., query="find the climax")
query_narrative(directory=..., filename=..., query="find similar pieces")
```

Pre-computed JSON files on disk — no re-analysis, instant response.

### Results on the test library

```
╔══════════════════════════════════════════════════════════
║  File                                    Sections  Duration
╠══════════════════════════════════════════════════════════
║  2020 cricketeering.aif                     11     6.3 min
║  2021 12 20 chiminy biscuits.aif            12     5.5 min
║  2021 melody partially resolute tetrax.      7     2.5 min
║  2022 06 05 negative 60.aif                 13     4.2 min
║  2022 09 17 caving in.aif                   22     9.5 min
║  2022-09-25 while madeline sleeps.aif       26    10.7 min
║  2022-12-14 a quatrax cosmos memory.aif     33    11.5 min
║  2023 04 13 centers.aif                     23     9.6 min
╠══════════════════════════════════════════════════════════
║  All 8 pieces have similarity populated with top-3 peers
╚══════════════════════════════════════════════════════════
```

---

## Current Architecture

```
audio_analysis/
├── core/
│   ├── tt_stft_kernel.py          # Streaming STFT with 3-tier dispatch
│   ├── tt_stft_hw.py              # JAX PJRT hardware kernel
│   ├── tt_stft_sim.py             # TT-Lang simulator kernel
│   ├── trajectory_analysis.py     # Per-window tension trajectory
│   ├── narrative_analysis.py      # Change-point detection + prose
│   ├── narrative_types.py         # Shared dataclass definitions
│   ├── feature_extraction.py      # 80+ librosa features
│   ├── feature_extraction_base.py # Shared extraction core
│   ├── phase_detection.py         # Musical phase classification
│   ├── clustering.py              # K-means track grouping
│   └── sequencing.py              # Listening sequence optimization
├── analysis/
│   ├── mood_analyzer.py           # 117 creative descriptors
│   ├── character_analyzer.py      # 59 character tags
│   └── cross_piece_similarity.py  # Library-wide similarity graph
├── exporters/
│   ├── narrative_exporter.py      # {stem}_narrative.json + .md
│   ├── markdown_exporter.py       # Comprehensive reports
│   ├── json_exporter.py
│   └── csv_exporter.py
└── api/
    ├── mcp_server.py              # 7 tools: analyze, narrative, query
    └── parallel_analyzer.py       # Full pipeline orchestrator
```

---

## What's Next

The 3.2x STFT speedup is real but modest. The opportunity is larger:

**Multi-chip parallelism**: The P150X4 mesh has 4 Blackhole chips. The current JAX dispatch uses whatever chip JAX picks by default. Explicit device placement could run 4 chunks in parallel — potential 4x additional speedup on top of per-chip acceleration, pushing STFT for a 60-minute library well under 0.25 seconds.

**Trajectory on hardware**: `TrajectoryAnalyzer` currently runs on CPU. The per-window feature computations (roughness, ZCR, chroma spread) are embarrassingly parallel — the same matmul structure as STFT. A JIT-compiled trajectory kernel would make the full pipeline hardware-accelerated end to end.

**Real-time streaming**: The chunk-streaming architecture was designed for this. With hardware-accelerated STFT and fast trajectory, latency per chunk could drop below 10ms — enabling live analysis of audio as it's being performed or recorded.

**Online narrative learning**: Right now the mood descriptors and section thresholds are hand-tuned. A piece could carry a "ground truth" section annotation (manually marked once) that updates the classifier's priors. Over a library of 100+ annotated pieces the thresholds would reflect the actual composer's perception, not generic calibration.

**Similarity-driven sequencing**: The CrossPieceSimilarity graph currently provides neighbors as a reference. Feeding it into the `SequenceRecommender` — so that sequence transitions favor structurally complementary pieces rather than just spectrally similar ones — would produce more interesting playlists.

---

## Setup

```bash
# Clone and activate environment
git clone <repo>
cd analyze_synths
source bin/activate   # or .venv/bin/activate

# Run full analysis on a directory
python analyze_library.py /path/to/audio/

# MCP server (for AI assistant integration)
python mcp_server.py

# Run the hardware STFT tests (requires JAX TT backend)
PYTHONPATH=/home/ttuser/p300c-xla-test/lib/python3.12/site-packages:. \
  /home/ttuser/p300c-xla-test/bin/python3 -m pytest tests/test_tt_stft_hw.py -v
```

**Dependencies**: librosa, scikit-learn, numpy, fastmcp, jax (with TT PJRT plugin for hardware path)

---

## Repository Layout

```
analyze_synths/
├── audio_analysis/     # Python package (above)
├── tests/              # pytest suite
├── generated/          # Analysis output (gitignored)
├── analyze_library.py  # CLI entry point
├── mcp_server.py       # MCP server entry point
└── requirements.txt
```

---

*Built on Tenstorrent Blackhole hardware. STFT kernel via JAX PJRT. Narrative analysis via custom change-point detection and template prose. All descriptor thresholds calibrated against a personal synthesizer library.*
