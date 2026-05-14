"""
Core analysis modules for audio feature extraction and processing.

This package contains the fundamental audio analysis algorithms that power
the synthesizer music analysis toolkit. Each module focuses on a specific
aspect of audio understanding relevant to creative music production.

Modules:
- feature_extraction: Extract comprehensive audio features using librosa
- phase_detection: Detect musical phases/sections in long-form compositions
- clustering: Group similar tracks using K-means on extracted features
- sequencing: Generate optimal listening sequences using musical flow principles
- tt_stft_kernel: TT-Metal-backed STFT kernel with NumPy/JAX fallback
- narrative_types: Dataclasses for trajectory and narrative data structures
- trajectory_analysis: Time-series trajectory extraction from audio features
- narrative_analysis: Change-point detection, section classification, prose generation
"""

from .feature_extraction import FeatureExtractor
from .phase_detection import PhaseDetector
from .clustering import AudioClusterer
from .sequencing import SequenceRecommender
from .tt_stft_kernel import SpectrogramChunk, TTStftKernel
from .narrative_types import (
    TrajectoryPoint,
    SectionMotion,
    Section,
    NarrativeResult,
)
from .trajectory_analysis import TrajectoryAnalyzer
from .narrative_analysis import NarrativeAnalyzer

__all__ = [
    'FeatureExtractor',
    'PhaseDetector',
    'AudioClusterer',
    'SequenceRecommender',
    'SpectrogramChunk',
    'TTStftKernel',
    # Narrative / trajectory types
    'TrajectoryPoint',
    'SectionMotion',
    'Section',
    'NarrativeResult',
    'TrajectoryAnalyzer',
    'NarrativeAnalyzer',
]