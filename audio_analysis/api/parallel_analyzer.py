"""
Parallel Audio Analyzer

This module provides a parallel-processing version of the AudioAnalyzer class
optimized for scalable processing and future hardware acceleration including
Tenstorrent processors.

Key Features:
1. Parallel file processing using multiple CPU cores
2. Batch processing for efficient memory utilization
3. Tensor-friendly data structures for hardware acceleration
4. Memory-efficient streaming for large datasets
5. Configurable processing parameters for different hardware

The design maintains backward compatibility with the original AudioAnalyzer
while providing significant performance improvements for large collections.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from dataclasses import dataclass
import time

# Import core parallel processing modules
from ..core.parallel_feature_extraction import (
    ParallelFeatureExtractor, 
    StreamingFeatureExtractor,
    ProcessingConfig,
    AudioBatch
)

# Import existing analysis modules
from ..core.phase_detection import PhaseDetector
from ..core.clustering import AudioClusterer
from ..core.sequencing import SequenceRecommender
from ..analysis.mood_analyzer import MoodAnalyzer
from ..analysis.character_analyzer import CharacterAnalyzer

# Import utility modules
from ..utils.audio_io import AudioLoader
from ..utils.data_processing import DataProcessor
from ..utils.visualization import Visualizer

# Import exporters
from ..exporters.csv_exporter import CSVExporter
from ..exporters.json_exporter import JSONExporter
from ..exporters.markdown_exporter import MarkdownExporter

# Import export utilities
from ..utils.export_utils import create_export_directory, create_export_subdirectories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParallelProcessingStats:
    """Statistics for parallel processing performance."""
    total_files: int = 0
    files_processed: int = 0
    processing_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parallel_speedup: float = 1.0
    memory_peak_mb: float = 0.0
    cpu_utilization: float = 0.0
    
    @property
    def processing_time(self) -> float:
        """Get total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Get processing success rate as percentage."""
        if self.total_files > 0:
            return (self.files_processed / self.total_files) * 100
        return 0.0


class ParallelAudioAnalyzer:
    """
    Parallel audio analyzer optimized for scalable processing and hardware acceleration.
    
    This class provides all the functionality of the original AudioAnalyzer
    with significant performance improvements through parallel processing.
    It's designed to scale efficiently across multiple CPU cores and
    prepare for future hardware acceleration.
    """
    
    def __init__(self, directory_path: Path, 
                 config: Optional[ProcessingConfig] = None,
                 enable_streaming: bool = False):
        """
        Initialize the parallel audio analyzer.
        
        Args:
            directory_path: Path to directory containing audio files
            config: Processing configuration parameters
            enable_streaming: Enable streaming processing for large files
        """
        # Validate directory path
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Initialize processing configuration
        self.config = config or ProcessingConfig()
        self.enable_streaming = enable_streaming
        
        # Initialize parallel processing components
        self.parallel_extractor = ParallelFeatureExtractor(self.config)
        if enable_streaming:
            self.streaming_extractor = StreamingFeatureExtractor(
                chunk_size_seconds=self.config.chunk_size_seconds
            )
        
        # Initialize existing analysis components
        self.phase_detector = PhaseDetector()
        self.clusterer = AudioClusterer()
        self.sequencer = SequenceRecommender()
        self.mood_analyzer = MoodAnalyzer()
        self.character_analyzer = CharacterAnalyzer()
        
        # Initialize utility components
        self.audio_loader = AudioLoader(self.config.sample_rate)
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        
        # Initialize exporters
        self.csv_exporter = CSVExporter()
        self.json_exporter = JSONExporter()
        self.markdown_exporter = MarkdownExporter()
        
        # Initialize data storage
        self.audio_features = []
        self.phase_data = []
        self.df = None
        self.cluster_labels = None
        self.cluster_analysis = None
        self.sequence_recommendations = None
        # Narrative analysis results: maps filename -> NarrativeResult
        self.narrative_results: dict = {}
        
        # Initialize processing statistics
        self.processing_stats = ParallelProcessingStats()
        
        logger.info(f"Initialized ParallelAudioAnalyzer with {self.config.max_workers} workers")
    
    def analyze_directory(self, show_progress: bool = True) -> Optional[pd.DataFrame]:
        """
        Analyze all audio files in the directory using parallel processing.
        
        This method implements the complete parallel analysis pipeline:
        1. Discover audio files
        2. Extract features in parallel batches
        3. Perform phase detection in parallel
        4. Analyze mood and character in parallel
        5. Combine results into structured DataFrame
        
        Args:
            show_progress: Whether to display progress information
            
        Returns:
            DataFrame with comprehensive analysis results
        """
        self.processing_stats.start_time = datetime.now()
        
        # Stage 1: Audio File Discovery
        if show_progress:
            print(f"Discovering audio files in: {self.directory_path}")
        
        audio_files = self._discover_audio_files()
        if not audio_files:
            print("No supported audio files found")
            return None
        
        self.processing_stats.total_files = len(audio_files)
        
        if show_progress:
            print(f"Found {len(audio_files)} audio files")
            
            # Show processing time estimate
            time_estimate = self.parallel_extractor.estimate_processing_time(audio_files)
            print(f"Estimated processing time: {time_estimate['estimated_time']:.1f}s "
                  f"({time_estimate['per_file_time']:.2f}s per file with {time_estimate['parallel_speedup']}x speedup)")
        
        # Stage 2: Parallel Feature Extraction
        if show_progress:
            print("Extracting features in parallel...")
        
        start_extraction = time.time()
        self.audio_features = self.parallel_extractor.extract_features_batch(audio_files)
        extraction_time = time.time() - start_extraction
        
        self.processing_stats.files_processed = len(self.audio_features)
        self.processing_stats.processing_errors = len(audio_files) - len(self.audio_features)
        
        if show_progress:
            print(f"Feature extraction completed in {extraction_time:.2f}s")
            print(f"Successfully processed {self.processing_stats.files_processed}/{self.processing_stats.total_files} files")
        
        if not self.audio_features:
            print("No files were successfully processed")
            return None
        
        # Stage 3: Parallel Phase Detection and Analysis
        if show_progress:
            print("Performing phase detection and creative analysis...")
        
        self._perform_parallel_analysis()
        
        # Stage 4: Create DataFrame
        self.df = pd.DataFrame(self.audio_features)
        self.df = self.data_processor.clean_dataframe(self.df)

        # Stage 5: Narrative Pipeline
        # Run temporal narrative analysis on each file, producing a NarrativeResult
        # per track that includes section segmentation, mood arcs, and prose summaries.
        # Each step is wrapped in a per-file try/except so a single failure cannot
        # derail the rest of the analysis pipeline.
        self._run_narrative_pipeline(audio_files, show_progress)

        # Record completion time
        self.processing_stats.end_time = datetime.now()
        
        # Calculate parallel speedup
        sequential_estimate = len(audio_files) * 0.5  # Estimated sequential time
        actual_time = self.processing_stats.processing_time
        self.processing_stats.parallel_speedup = sequential_estimate / actual_time if actual_time > 0 else 1.0
        
        if show_progress:
            self._print_analysis_summary()
        
        return self.df
    
    def _discover_audio_files(self) -> List[Path]:
        """
        Discover all supported audio files in the directory.
        
        Returns:
            List of paths to valid audio files
        """
        supported_extensions = ['*.wav', '*.WAV', '*.aiff', '*.AIFF', '*.aif', '*.AIF', '*.mp3', '*.MP3']
        
        audio_files = []
        for extension in supported_extensions:
            audio_files.extend(list(self.directory_path.glob(extension)))
        
        # Validate and filter files
        valid_files = []
        for file_path in audio_files:
            if file_path.is_file() and file_path.stat().st_size > 0:
                valid_files.append(file_path)
        
        return sorted(valid_files)

    def _run_narrative_pipeline(self, audio_files: List[Path],
                                 show_progress: bool = True) -> None:
        """
        Run the temporal narrative analysis pipeline on every audio file.

        For each file this method:
          1. Loads raw audio via librosa (sr=22050, mono).
          2. Builds a SpectrogramChunk using TTStftKernel when available, falling
             back to a pure-librosa STFT/mel-spectrogram if the kernel is absent or
             raises an exception.
          3. Passes the chunk to TrajectoryAnalyzer to obtain per-frame trajectory
             points (energy, brightness, roughness, etc.).
          4. Passes the trajectory to NarrativeAnalyzer to segment sections, assign
             mood arcs, and compose the prose narrative.
          5. Stores the resulting NarrativeResult in ``self.narrative_results``.

        After all files are processed, CrossPieceSimilarity.compute_library() is
        called once to populate the ``similar_to`` field on every result.

        Any exception during a single file's processing is caught and logged as a
        WARNING so that one bad file cannot abort the whole pipeline.

        Args:
            audio_files: Ordered list of audio file paths (same as discovered list).
            show_progress: Whether to print a per-stage progress message.
        """
        # Lazy imports — keep top-level module load fast
        from audio_analysis.core.trajectory_analysis import TrajectoryAnalyzer
        from audio_analysis.core.narrative_analysis import NarrativeAnalyzer
        from audio_analysis.analysis.cross_piece_similarity import CrossPieceSimilarity

        if show_progress:
            print("Running narrative analysis pipeline...")

        traj_az = TrajectoryAnalyzer(sr=22050)
        narr_az = NarrativeAnalyzer()
        # Reset narrative results in case analyze_directory is called more than once
        self.narrative_results = {}

        for file_path in audio_files:
            filename = file_path.name
            try:
                import librosa
                # Load at a consistent sample rate for all downstream analysis
                audio, sr = librosa.load(str(file_path), sr=22050, mono=True)
                duration = float(len(audio)) / sr

                # Build SpectrogramChunk using librosa.
                #
                # We intentionally skip TTStftKernel here because the TT-Lang
                # simulator (tt_stft_sim.py) can produce C-level segfaults that
                # Python's try/except cannot intercept.  The narrative pipeline
                # runs serially and a crash in one file would abort the entire
                # batch.  TTStftKernel / the TT simulator are exercised separately
                # via the dedicated hardware smoke tests; for this production path
                # librosa is the reliable choice.
                import numpy as np
                from audio_analysis.core.tt_stft_kernel import SpectrogramChunk
                stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)).T
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
                ).T
                # Timestamps: centre of each STFT frame in seconds
                ts = np.arange(stft.shape[0], dtype=np.float32) * 512 / sr
                chunk = SpectrogramChunk(
                    mag=stft.astype(np.float32),
                    mel=mel.astype(np.float32),
                    timestamps=ts,
                )

                # Derive per-frame trajectory from spectrogram + waveform
                trajectory = traj_az.analyze(chunk, audio)

                # Segment into sections, assign moods, and compose narrative prose
                result = narr_az.analyze(filename, duration, trajectory, audio, sr)
                self.narrative_results[filename] = result

            except Exception as exc:
                logger.warning("Narrative analysis failed for %s: %s", filename, exc)

        # Compute cross-piece similarity once we have all results — this populates
        # the ``similar_to`` list on every NarrativeResult so MCP queries can answer
        # "find pieces similar to this one" without re-running analysis.
        if len(self.narrative_results) > 1:
            CrossPieceSimilarity().compute_library(
                list(self.narrative_results.values())
            )

        if show_progress:
            print(f"Narrative analysis complete: {len(self.narrative_results)}/{len(audio_files)} files")

    def _perform_parallel_analysis(self):
        """
        Perform parallel phase detection and creative analysis.
        
        This method runs additional analysis steps in parallel to maximize
        processing efficiency.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Prepare tasks for parallel processing
        analysis_tasks = []
        
        # Create tasks for each file
        for features in self.audio_features:
            file_path = Path(features['filename'])
            # Find the actual file path (features only contains filename)
            actual_file_path = None
            for audio_file in self.directory_path.glob(f"**/{file_path.name}"):
                if audio_file.is_file():
                    actual_file_path = audio_file
                    break
            
            if actual_file_path:
                analysis_tasks.append((features, actual_file_path))
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self._analyze_single_file, features, file_path): (features, file_path)
                for features, file_path in analysis_tasks
            }
            
            for future in as_completed(future_to_task):
                features, file_path = future_to_task[future]
                try:
                    enhanced_features = future.result()
                    if enhanced_features:
                        # Update the features in place
                        features.update(enhanced_features)
                except Exception as e:
                    logger.error(f"Error in parallel analysis for {file_path}: {e}")
    
    def _analyze_single_file(self, features: Dict[str, Any], file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Perform complete analysis for a single file.
        
        Args:
            features: Basic features already extracted
            file_path: Path to the audio file
            
        Returns:
            Enhanced features with phase, mood, and character analysis
        """
        try:
            # Load audio for phase detection
            audio_data, sample_rate = self.audio_loader.load_audio(file_path)
            if audio_data is None:
                return None
            
            # Phase detection
            phases, times, rms_smooth, spectral_smooth, change_signal = self.phase_detector.detect_phases(
                audio_data, sample_rate
            )
            
            # Store phase data
            phase_info = {
                'filename': file_path.name,
                'total_duration': features['duration'],
                'num_phases': len(phases),
                'phases': phases
            }
            self.phase_data.append(phase_info)
            
            # Mood analysis
            mood_descriptors, primary_mood, mood_confidence = self.mood_analyzer.analyze_track_mood(features)
            
            # Character analysis
            character_tags, primary_character, character_confidence = self.character_analyzer.analyze_track_character(features)
            
            # Phase-level mood analysis
            for phase in phases:
                phase_mood, phase_confidence = self.mood_analyzer.analyze_mood(
                    phase['phase_data'], phase['basic_spectral']
                )
                phase['mood_descriptors'] = phase_mood
                phase['mood_confidence'] = phase_confidence
            
            # Calculate structural features
            structural_features = self._calculate_structural_features(phases)
            
            # Return enhanced features
            enhanced_features = {
                **structural_features,
                'mood_descriptors': ', '.join(mood_descriptors),
                'primary_mood': primary_mood,
                'mood_confidence': mood_confidence.get(primary_mood, 0),
                'character_tags': ', '.join(character_tags),
                'primary_character': primary_character,
                'character_confidence': character_confidence.get(primary_character, 0),
                'num_phases': len(phases),
                'phase_analysis_available': True
            }
            
            return enhanced_features
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_structural_features(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate structural features from phase analysis.
        
        Args:
            phases: List of detected phases
            
        Returns:
            Dictionary with structural features
        """
        if not phases:
            return {}
        
        from ..utils.type_conversion import safe_float_convert
        from ..utils.statistics import calculate_progression_trend
        
        energies = [safe_float_convert(phase['avg_energy']) for phase in phases]
        brightnesses = [safe_float_convert(phase['avg_brightness']) for phase in phases]
        
        structural_features = {
            'energy_range': max(energies) - min(energies),
            'brightness_range': max(brightnesses) - min(brightnesses),
            'avg_phase_duration': np.mean([safe_float_convert(phase['duration']) for phase in phases]),
            'phase_duration_std': np.std([safe_float_convert(phase['duration']) for phase in phases]),
            'has_climax': any('Climax' in phase['phase_type'] for phase in phases),
            'has_breakdown': any('Breakdown' in phase['phase_type'] or 'Quiet' in phase['phase_type'] for phase in phases),
            'has_build_up': any('Build-up' in phase['phase_type'] for phase in phases),
            'structural_complexity': len(set(phase['phase_type'] for phase in phases)),
            'energy_progression': self._get_progression_description(calculate_progression_trend(energies), 'energy'),
            'brightness_progression': self._get_progression_description(calculate_progression_trend(brightnesses), 'brightness')
        }
        
        return structural_features
    
    def _get_progression_description(self, trend: str, feature_type: str) -> str:
        """
        Get progression description based on trend and feature type.
        
        Args:
            trend: Trend direction ('increasing', 'decreasing', 'stable')
            feature_type: Type of feature ('energy', 'brightness')
            
        Returns:
            Human-readable progression description
        """
        if feature_type == 'energy':
            if trend == 'increasing':
                return 'building'
            elif trend == 'decreasing':
                return 'declining'
            else:
                return 'stable'
        elif feature_type == 'brightness':
            if trend == 'increasing':
                return 'brightening'
            elif trend == 'decreasing':
                return 'darkening'
            else:
                return 'stable'
        else:
            return trend
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Perform clustering analysis using parallel processing.
        
        Args:
            n_clusters: Number of clusters (None = auto-determine)
            
        Returns:
            Tuple containing cluster labels, cluster centers, and feature names
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for clustering. Run analyze_directory() first.")
        
        # Use existing clustering implementation
        cluster_labels, cluster_centers, feature_names = self.clusterer.perform_clustering(
            self.df, n_clusters
        )
        
        self.cluster_labels = cluster_labels
        self.cluster_analysis = self.clusterer.analyze_clusters(self.df, cluster_labels)
        
        return cluster_labels, cluster_centers, feature_names
    
    def recommend_sequence(self) -> List[Dict[str, Any]]:
        """
        Generate optimal sequence recommendations.
        
        Returns:
            List of sequence recommendations
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for sequencing. Run analyze_directory() first.")
        
        self.sequence_recommendations = self.sequencer.recommend_sequence(self.df)
        return self.sequence_recommendations
    
    def export_comprehensive_analysis(self, export_dir: Optional[Path] = None,
                                    show_plots: bool = False,
                                    export_format: str = "all",
                                    base_name: str = "analysis") -> Dict[str, Any]:
        """
        Export comprehensive analysis results in specified formats.
        
        Args:
            export_dir: Directory for export
            show_plots: Whether to display plots
            export_format: Format for exports ("all", "csv", "json", "markdown")
            base_name: Base name for generated files
            
        Returns:
            Dictionary with export information
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for export. Run analyze_directory() first.")
        
        # Create export directory using shared utility
        export_dir = create_export_directory(export_dir, prefix="parallel_audio_analysis")
        
        # Create subdirectories using shared utility
        create_export_subdirectories(export_dir)
        
        print(f"Exporting comprehensive analysis to: {export_dir}")
        
        # Export based on requested format
        data_exports = {}
        visualization_exports = {}
        report_exports = {}
        
        if export_format in ["all", "csv"]:
            # Export data files
            data_exports = self._export_data_files(export_dir, base_name)
        
        if export_format == "all":
            # Generate visualizations
            visualization_exports = self._generate_visualizations(export_dir, show_plots)
        
        # Generate reports based on format
        report_exports = self._generate_reports(export_dir, export_format, base_name)
        
        # Create export summary with parallel processing stats
        export_summary = {
            'export_directory': str(export_dir),
            'export_timestamp': datetime.now().isoformat(),
            'files_analyzed': len(self.df),
            'data_exports': data_exports,
            'visualization_exports': visualization_exports,
            'report_exports': report_exports,
            'processing_stats': {
                'total_files': self.processing_stats.total_files,
                'files_processed': self.processing_stats.files_processed,
                'processing_errors': self.processing_stats.processing_errors,
                'processing_time': self.processing_stats.processing_time,
                'success_rate': self.processing_stats.success_rate,
                'parallel_speedup': self.processing_stats.parallel_speedup,
                'parallel_config': {
                    'max_workers': self.config.max_workers,
                    'batch_size': self.config.batch_size,
                    'use_multiprocessing': self.config.use_multiprocessing
                }
            }
        }
        
        # Save export summary
        with open(export_dir / "export_summary.json", 'w') as f:
            json.dump(export_summary, f, indent=2, default=str)
        
        return export_summary
    
    def _export_data_files(self, export_dir: Path) -> Dict[str, str]:
        """Export data files in various formats."""
        data_dir = export_dir / "data"
        
        # Export main features
        features_path = data_dir / "audio_features.csv"
        self.csv_exporter.export_features(self.df, features_path)
        
        # Export phase data
        phase_path = data_dir / "phase_analysis.csv"
        self.csv_exporter.export_phases(self.phase_data, phase_path)
        
        # Export cluster analysis if available
        cluster_path = None
        if self.cluster_analysis:
            cluster_path = data_dir / "cluster_analysis.csv"
            self.csv_exporter.export_clusters(self.cluster_analysis, cluster_path)
        
        # Export sequence recommendations if available
        sequence_path = None
        if self.sequence_recommendations:
            sequence_path = data_dir / "sequence_recommendations.csv"
            self.csv_exporter.export_sequence(self.sequence_recommendations, sequence_path)
        
        # Export summary statistics
        summary_path = data_dir / "summary_statistics.csv"
        self.csv_exporter.export_summary_stats(self.df, self.phase_data, summary_path)
        
        return {
            'features': str(features_path),
            'phases': str(phase_path),
            'clusters': str(cluster_path) if cluster_path else None,
            'sequence': str(sequence_path) if sequence_path else None,
            'summary': str(summary_path)
        }
    
    def _generate_visualizations(self, export_dir: Path, show_plots: bool) -> Dict[str, str]:
        """Generate visualization files."""
        images_dir = export_dir / "images"
        visualization_paths = {}
        
        # Phase timeline visualization
        if self.phase_data:
            phase_timeline_path = images_dir / "phase_timeline.png"
            self.visualizer.create_phase_timeline(self.phase_data, phase_timeline_path, show_plots)
            visualization_paths['phase_timeline'] = str(phase_timeline_path)
        
        # Cluster visualization
        if self.cluster_labels is not None:
            cluster_path = images_dir / "cluster_analysis.png"
            features_scaled = self.data_processor.prepare_clustering_features(self.df)
            standardized_features, _ = self.data_processor.standardize_features(features_scaled)
            
            self.visualizer.create_cluster_visualization(
                self.df, self.cluster_labels, 
                standardized_features.values, list(standardized_features.columns),
                cluster_path, show_plots
            )
            visualization_paths['cluster_analysis'] = str(cluster_path)
        
        # Mood distribution visualization
        if 'primary_mood' in self.df.columns:
            mood_path = images_dir / "mood_distribution.png"
            mood_data = self.mood_analyzer.analyze_mood_distribution(self.df['primary_mood'].tolist())
            self.visualizer.create_mood_distribution_plot(mood_data, mood_path, show_plots)
            visualization_paths['mood_distribution'] = str(mood_path)
        
        # Sequence visualization
        if self.sequence_recommendations:
            sequence_path = images_dir / "sequence_recommendations.png"
            self.visualizer.create_sequence_visualization(
                self.sequence_recommendations, sequence_path, show_plots
            )
            visualization_paths['sequence_recommendations'] = str(sequence_path)
        
        return visualization_paths
    
    def _generate_reports(self, export_dir: Path) -> Dict[str, str]:
        """Generate analysis reports."""
        reports_dir = export_dir / "reports"
        report_paths = {}
        
        # Generate comprehensive markdown report
        markdown_path = reports_dir / "comprehensive_analysis_report.md"
        self.markdown_exporter.generate_comprehensive_report(
            self.df, self.phase_data, self.cluster_analysis, 
            self.sequence_recommendations, markdown_path
        )
        report_paths['comprehensive_report'] = str(markdown_path)
        
        # Generate JSON export
        json_path = reports_dir / "analysis_data.json"
        self.json_exporter.export_comprehensive_data(
            self.df, self.phase_data, self.cluster_analysis,
            self.sequence_recommendations, json_path
        )
        report_paths['json_data'] = str(json_path)
        
        return report_paths
    
    def _print_analysis_summary(self):
        """Print a summary of the parallel analysis results."""
        print("\n" + "="*60)
        print("PARALLEL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Files found: {self.processing_stats.total_files}")
        print(f"Files processed: {self.processing_stats.files_processed}")
        print(f"Processing errors: {self.processing_stats.processing_errors}")
        print(f"Processing time: {self.processing_stats.processing_time:.1f} seconds")
        print(f"Success rate: {self.processing_stats.success_rate:.1f}%")
        print(f"Parallel speedup: {self.processing_stats.parallel_speedup:.1f}x")
        print(f"Workers used: {self.config.max_workers}")
        print(f"Batch size: {self.config.batch_size}")
        
        if self.df is not None:
            print(f"Features extracted: {len(self.df.columns)}")
            print(f"Total phases detected: {sum(len(f['phases']) for f in self.phase_data)}")
            print(f"Average phases per track: {np.mean([len(f['phases']) for f in self.phase_data]):.1f}")
        
        # Performance metrics
        if self.processing_stats.processing_time > 0:
            throughput = self.processing_stats.files_processed / self.processing_stats.processing_time
            print(f"Throughput: {throughput:.1f} files/second")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'parallel_processing_stats': {
                'total_files': self.processing_stats.total_files,
                'files_processed': self.processing_stats.files_processed,
                'processing_errors': self.processing_stats.processing_errors,
                'processing_time': self.processing_stats.processing_time,
                'success_rate': self.processing_stats.success_rate,
                'parallel_speedup': self.processing_stats.parallel_speedup,
                'throughput': self.processing_stats.files_processed / self.processing_stats.processing_time if self.processing_stats.processing_time > 0 else 0
            },
            'configuration': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'use_multiprocessing': self.config.use_multiprocessing,
                'enable_tensor_optimization': self.config.enable_tensor_optimization,
                'memory_limit_mb': self.config.memory_limit_mb
            }
        }
        
        if self.df is not None:
            stats['data_stats'] = {
                'total_tracks': len(self.df),
                'total_features': len(self.df.columns),
                'total_phases': sum(len(f['phases']) for f in self.phase_data),
                'avg_track_duration': self.df['duration'].mean(),
                'total_collection_duration': self.df['duration'].sum()
            }
        
        return stats