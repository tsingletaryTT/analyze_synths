# audio_analysis/analysis/cross_piece_similarity.py
"""
CrossPieceSimilarity: compute pairwise similarity across a library of NarrativeResults.

Similarity = 0.6 * cosine(structure_a, structure_b)
           + 0.4 * cosine(texture_a,   texture_b)

Structure fingerprint: 35-dim vector (7 section types × 5 tension bins),
                       one-hot per (type, bin) accumulation, L2 normalized.
Texture fingerprint:   10-dim vector from NarrativeResult.texture_fingerprint.

Design rationale
----------------
Structure (0.6 weight) is given more importance than texture (0.4 weight)
because two pieces with the same narrative arc (intro→climax→outro) are more
musically similar in the sense a composer cares about than two pieces that
happen to share a similar spectral centroid.

The 7 × 5 grid encodes both *what kind of section* appeared and *how tense
it was*.  A climax at 0.9 tension is encoded differently from a plateau at
0.5 tension, which lets the similarity measure distinguish pieces whose
overall narrative shape looks the same at a coarse level but has a very
different emotional weight at each stage.

Cosine similarity is used throughout (rather than Euclidean distance) so
that a short piece with one intro and a long piece with three intros score
the same — both express the "intro" concept, just with different frequency.
The L2 normalisation in _encode_structure makes this explicit.
"""
from __future__ import annotations

from typing import List

import numpy as np

from audio_analysis.core.narrative_types import NarrativeResult

# Seven canonical section types recognised by the narrative pipeline.
# "plateau" acts as the catch-all for any unknown label.
SECTION_TYPES = ["intro", "rising", "plateau", "climax", "falling", "release", "outro"]

# Number of tension bins used to subdivide each section type.
# Five bins gives fine enough resolution to distinguish low/mid/high tension
# while keeping the vector compact and avoiding data-sparsity issues on
# short pieces.
N_TYPES = len(SECTION_TYPES)    # 7
N_BINS  = 5                     # tension bins: [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]
STRUCT_DIM = N_TYPES * N_BINS   # 35

# Pre-built lookup table: section type string → row index in the 7×5 grid.
_TYPE_IDX = {t: i for i, t in enumerate(SECTION_TYPES)}


class CrossPieceSimilarity:
    """Compute pairwise similarity scores and populate NarrativeResult.similar_to.

    This class is intentionally stateless — all data lives on the
    NarrativeResult objects, making instances cheap to create and safe to
    share across threads or calls.

    Parameters
    ----------
    None — all configuration is via method arguments.

    Methods
    -------
    score(a, b) -> float
        Return a single 0–1 similarity value between two NarrativeResults.
    compute_library(results, top_k=3) -> None
        Run pairwise comparison across the whole library and populate each
        result's similar_to list with the top_k nearest neighbours.
    _encode_structure(fingerprint) -> np.ndarray
        Convert a structure_fingerprint list into the 35-dim descriptor.
    """

    def score(self, a: NarrativeResult, b: NarrativeResult) -> float:
        """Return 0–1 similarity between two NarrativeResults.

        Combines structure similarity (60 %) and texture similarity (40 %).
        Both components use cosine distance so the result is independent of
        how many sections or trajectory points each piece has.

        Parameters
        ----------
        a, b : NarrativeResult
            The two pieces to compare.  Can be the same object, in which
            case the score is exactly 1.0.

        Returns
        -------
        float
            Value in [0, 1].  1.0 means identical structure and texture;
            0.0 means maximally dissimilar (or one/both vectors are zero).
        """
        sa = self._encode_structure(a.structure_fingerprint)
        sb = self._encode_structure(b.structure_fingerprint)
        ta = a.texture_fingerprint.astype(np.float32)
        tb = b.texture_fingerprint.astype(np.float32)
        return 0.6 * self._cosine(sa, sb) + 0.4 * self._cosine(ta, tb)

    def compute_library(self, results: List[NarrativeResult], top_k: int = 3) -> None:
        """Compute pairwise similarity for all results and write nearest neighbours.

        Encodes all structure and texture fingerprints once up-front, then
        runs an O(n²) comparison loop.  For typical library sizes (< 1000
        tracks) this is fast enough; a batched matrix implementation can be
        substituted here if needed at larger scale.

        After this call, every result's similar_to list is replaced with the
        filenames of the top_k most similar *other* tracks (self is always
        excluded).  If n < top_k + 1, similar_to will be shorter than top_k.

        Parameters
        ----------
        results : List[NarrativeResult]
            All tracks in the library.  Modified in-place.
        top_k : int, optional
            Maximum number of similar tracks to record per result.
            Default is 3.
        """
        n = len(results)
        if n == 0:
            return

        # Pre-compute all fingerprint encodings so each pair lookup is O(1).
        structs  = [self._encode_structure(r.structure_fingerprint) for r in results]
        textures = [r.texture_fingerprint.astype(np.float32) for r in results]

        for i, r in enumerate(results):
            # Build (score, filename) pairs for every other track.
            scores = []
            for j, other in enumerate(results):
                if i == j:
                    # A piece is always its own best match; exclude from ranking.
                    continue
                s = (0.6 * self._cosine(structs[i], structs[j])
                     + 0.4 * self._cosine(textures[i], textures[j]))
                scores.append((s, other.filename))

            # Sort descending by similarity score, keep top_k filenames.
            scores.sort(reverse=True)
            r.similar_to = [fname for _, fname in scores[:top_k]]

    def _encode_structure(self, fingerprint: list) -> np.ndarray:
        """Convert a structure_fingerprint into a 35-dim L2-normalised vector.

        The fingerprint is a list of (section_type, tension_score) pairs.
        Each pair maps to a cell in a 7 (section types) × 5 (tension bins)
        grid, and the corresponding cell is incremented by 1.  The result is
        L2-normalised so that cosine similarity is insensitive to piece length
        (i.e. number of sections).

        Unknown section type labels fall back to "plateau" (index 2).

        Parameters
        ----------
        fingerprint : list of (str, float)
            As stored in NarrativeResult.structure_fingerprint.

        Returns
        -------
        np.ndarray
            Shape (35,), dtype float32.  All-zeros if fingerprint is empty
            (norm guard prevents division by zero).
        """
        vec = np.zeros(STRUCT_DIM, dtype=np.float32)
        for section_type, tension_score in fingerprint:
            # Unknown labels map to "plateau" so they still contribute signal
            # rather than being silently dropped.
            type_idx = _TYPE_IDX.get(section_type, _TYPE_IDX["plateau"])
            # Clamp tension to [0, 1] then quantise into N_BINS buckets.
            # min(..., N_BINS - 1) ensures that tension_score == 1.0 maps to
            # bin 4 rather than overflowing to index 5.
            bin_idx  = min(int(float(tension_score) * N_BINS), N_BINS - 1)
            vec[type_idx * N_BINS + bin_idx] += 1.0

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity clamped to [0, 1].

        Clamping to 0 rather than allowing negative values keeps the
        weighted sum in score() well-behaved and avoids artefacts from
        floating-point rounding that would push a dot product fractionally
        below zero for near-orthogonal vectors.

        Parameters
        ----------
        a, b : np.ndarray
            Vectors of any dimension.  Zero-vectors yield 0.0.

        Returns
        -------
        float
            Cosine similarity in [0, 1].
        """
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
