"""
Synthesizer Music Analysis Toolkit

A comprehensive Python package for analyzing synthesizer music with creative descriptors.
Designed specifically for composers and electronic music creators to understand their work
through mood analysis, phase detection, clustering, and intelligent sequencing.

Key Features:
- Creative mood classification using 17 descriptors (spacey, organic, synthetic, etc.)
- Musical phase detection for song structure analysis
- K-means clustering for track grouping and playlist creation
- Intelligent song sequencing based on musical flow principles
- Export to multiple formats (CSV, JSON, Markdown) optimized for LLM consumption
- FastMCP server integration for remote analysis capabilities
- Parallel processing for scalable analysis across multiple CPU cores
- Tensor-optimized operations for hardware acceleration (Tenstorrent, GPU)
- Streaming processing for large audio files and datasets

Usage:
    # Standard analysis
    from audio_analysis import AudioAnalyzer
    
    analyzer = AudioAnalyzer('/path/to/audio/files')
    results = analyzer.analyze_directory()
    analyzer.export_comprehensive_analysis(export_format="all")
    
    # Parallel processing for better performance
    from audio_analysis import ParallelAudioAnalyzer
    
    analyzer = ParallelAudioAnalyzer('/path/to/audio/files')
    results = analyzer.analyze_directory()
    analyzer.export_comprehensive_analysis(export_format="markdown", base_name="my_analysis")
"""

import logging as _logging
import os as _os

from .api.analyzer import AudioAnalyzer
from .api.parallel_analyzer import ParallelAudioAnalyzer
from .api.mcp_server import MCPAudioAnalyzer
from .analysis.descriptors import MoodDescriptors, CharacterTags

# Parallel processing components
from .core.parallel_feature_extraction import ParallelFeatureExtractor, ProcessingConfig
from .core.parallel_clustering import ParallelKMeansClusterer, ClusteringConfig
from .core.tensor_operations import TensorFeatureExtractor, TensorBatch

_logger = _logging.getLogger(__name__)


def _activate_tt_pjrt() -> None:
    """Set PJRT plugin env var if not already set by the caller's environment."""
    plugin_path = _os.path.expanduser('~/tt-xla/build/lib/libpjrt_tt.so')
    if not _os.environ.get('PJRT_PLUGIN_LIBRARY_PATH') and _os.path.exists(plugin_path):
        _os.environ['PJRT_PLUGIN_LIBRARY_PATH'] = plugin_path


def _detect_tt_device() -> str:
    """
    Probe for Tenstorrent hardware via JAX PJRT plugin.

    Three-tier fallback chain:
      1. TT hardware detected + PJRT plugin loads → returns 'tenstorrent'
         logs: "Tenstorrent hardware detected: N device(s)"
      2. JAX importable but TT devices unavailable / PJRT init fails →
         returns 'cpu', logs warning: "TT device unavailable, using CPU JAX"
      3. JAX not importable → returns 'cpu',
         logs debug: "JAX not available, using librosa path"
    """
    try:
        import jax  # noqa: PLC0415
    except ImportError:
        _logger.debug("JAX not available, using librosa path")
        return 'cpu'
    try:
        _activate_tt_pjrt()
        # Probe once at import so DEFAULT_DEVICE is a consistent module-level constant
        devices = jax.devices()
        if any(d.platform == 'tt' for d in devices):
            tt_count = sum(1 for d in devices if d.platform == 'tt')
            _logger.info("Tenstorrent hardware detected: %d device(s)", tt_count)
            return 'tenstorrent'
        _logger.warning("TT device unavailable, using CPU JAX")
        return 'cpu'
    except Exception as exc:
        _logger.warning("TT device unavailable, using CPU JAX: %s", exc)
        return 'cpu'


DEFAULT_DEVICE: str = _detect_tt_device()

__version__ = "2.0.0"
__author__ = "Audio Analysis Toolkit Team"
__email__ = "support@audioanalysis.com"

__all__ = [
    'AudioAnalyzer',
    'ParallelAudioAnalyzer',
    'MCPAudioAnalyzer',
    'MoodDescriptors',
    'CharacterTags',
    'ParallelFeatureExtractor',
    'ProcessingConfig',
    'ParallelKMeansClusterer',
    'ClusteringConfig',
    'TensorFeatureExtractor',
    'TensorBatch',
    'DEFAULT_DEVICE',
    '_detect_tt_device',
]