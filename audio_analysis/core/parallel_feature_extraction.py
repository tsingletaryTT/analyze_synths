"""
Parallel Feature Extraction Module

This module implements parallel audio feature extraction designed for scalable
processing across multiple cores and optimized for future hardware acceleration
including Tenstorrent processors.

Key Design Principles:
1. Batch processing for multiple audio files simultaneously
2. Vectorized operations optimized for tensor processing units
3. Memory-efficient streaming for large datasets
4. Configurable parallelism levels for different hardware
5. Tensor-friendly data structures for hardware acceleration

The architecture separates computation into distinct parallel stages:
- Audio loading and preprocessing (I/O bound)
- Feature extraction (compute bound, vectorizable)
- Post-processing and aggregation (memory bound)

This separation allows for optimal hardware utilization on systems with
specialized processing units like Tenstorrent's tensor cores.
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
from functools import partial
import time

# Import shared feature extraction core
from .feature_extraction_base import extract_features_from_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioBatch:
    """
    Container for batched audio processing data.
    
    Designed for efficient memory access patterns and tensor operations.
    All arrays are structured for vectorized processing.
    """
    audio_data: List[np.ndarray] = field(default_factory=list)
    sample_rates: List[int] = field(default_factory=list)
    file_paths: List[Path] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    batch_size: int = 0
    
    def __post_init__(self):
        self.batch_size = len(self.audio_data)
    
    def to_tensor_format(self) -> Dict[str, np.ndarray]:
        """
        Convert batch to tensor-friendly format for hardware acceleration.
        
        Returns:
            Dictionary with arrays structured for tensor processing
        """
        # Find maximum length for padding
        max_length = max(len(audio) for audio in self.audio_data) if self.audio_data else 0
        
        # Create padded tensor with shape (batch_size, max_length)
        # This format is optimal for tensor processing units
        padded_audio = np.zeros((self.batch_size, max_length), dtype=np.float32)
        lengths = np.zeros(self.batch_size, dtype=np.int32)
        
        for i, audio in enumerate(self.audio_data):
            audio_len = len(audio)
            padded_audio[i, :audio_len] = audio.astype(np.float32)
            lengths[i] = audio_len
        
        return {
            'audio_tensor': padded_audio,
            'lengths': lengths,
            'sample_rates': np.array(self.sample_rates, dtype=np.int32),
            'durations': np.array(self.durations, dtype=np.float32)
        }


def _get_default_device() -> str:
    """Lazy import avoids circular imports at module load time."""
    try:
        from audio_analysis import DEFAULT_DEVICE  # noqa: PLC0415
        return DEFAULT_DEVICE
    except Exception:
        return 'cpu'


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing parameters."""
    max_workers: int = field(default_factory=lambda: mp.cpu_count())
    batch_size: int = 8
    use_multiprocessing: bool = True
    enable_tensor_optimization: bool = True
    memory_limit_mb: int = 2048
    chunk_size_seconds: float = 30.0
    sample_rate: Optional[int] = None
    device: str = field(default_factory=_get_default_device)

    def __post_init__(self):
        # Enforce 22050 Hz when targeting TT so filter matrices stay consistent.
        if self.device == 'tenstorrent' and self.sample_rate is None:
            self.sample_rate = 22050
        # Adjust batch size based on available memory
        if self.memory_limit_mb < 1024:
            self.batch_size = min(self.batch_size, 4)
        elif self.memory_limit_mb > 4096:
            self.batch_size = min(self.batch_size, 16)


class ParallelFeatureExtractor:
    """
    Parallel audio feature extractor optimized for scalable processing.
    
    This class implements a parallel processing pipeline designed for:
    1. Multi-core CPU processing (current systems)
    2. Future hardware acceleration (Tenstorrent, GPUs)
    3. Efficient memory usage for large datasets
    4. Vectorized operations for tensor processing
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the parallel feature extractor.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessingConfig()
        self.processing_stats = {
            'files_processed': 0,
            'total_processing_time': 0,
            'average_file_time': 0,
            'memory_usage_mb': 0
        }
        
        # Initialize feature extraction parameters
        self.feature_params = {
            'n_mfcc': 13,
            'n_chroma': 12,
            'n_tonnetz': 6,
            'hop_length': 512,
            'n_fft': 2048
        }
        
        logger.info(f"Initialized ParallelFeatureExtractor with {self.config.max_workers} workers")
    
    def extract_features_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract features from multiple audio files in parallel.
        
        This method implements the main parallel processing pipeline:
        1. Load audio files in parallel (I/O bound)
        2. Process features in batches (compute bound)
        3. Aggregate results (memory bound)
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            List of feature dictionaries for each file
        """
        start_time = time.time()
        
        # Stage 1: Parallel audio loading
        logger.info(f"Loading {len(file_paths)} audio files in parallel...")
        audio_batches = self._load_audio_parallel(file_paths)
        
        # Stage 2: Parallel feature extraction
        logger.info(f"Extracting features from {len(audio_batches)} batches...")
        feature_results = self._extract_features_parallel(audio_batches)
        
        # Stage 3: Results aggregation
        all_features = []
        for batch_features in feature_results:
            all_features.extend(batch_features)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats.update({
            'files_processed': len(file_paths),
            'total_processing_time': processing_time,
            'average_file_time': processing_time / len(file_paths) if file_paths else 0
        })
        
        logger.info(f"Processed {len(file_paths)} files in {processing_time:.2f}s "
                   f"({self.processing_stats['average_file_time']:.3f}s per file)")
        
        return all_features
    
    def _load_audio_parallel(self, file_paths: List[Path]) -> List[AudioBatch]:
        """
        Load audio files in parallel and organize into batches.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of AudioBatch objects ready for processing
        """
        # Create batches of file paths
        batches = []
        for i in range(0, len(file_paths), self.config.batch_size):
            batch_paths = file_paths[i:i + self.config.batch_size]
            batches.append(batch_paths)
        
        # Load each batch in parallel.
        # IMPORTANT: Use 'spawn' start method, not the default 'fork'.  JAX is
        # multithreaded so os.fork() after JAX initialisation causes a deadlock.
        # spawn creates a clean child process that imports libraries fresh.
        audio_batches = []
        if self.config.use_multiprocessing:
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=mp.get_context('spawn'),
            ) as executor:
                future_to_batch = {
                    executor.submit(self._load_audio_batch, batch_paths): batch_paths
                    for batch_paths in batches
                }
                
                for future in as_completed(future_to_batch):
                    batch_paths = future_to_batch[future]
                    try:
                        audio_batch = future.result()
                        if audio_batch.batch_size > 0:
                            audio_batches.append(audio_batch)
                    except Exception as e:
                        logger.error(f"Error loading batch {batch_paths}: {e}")
        else:
            # Fallback to sequential loading
            for batch_paths in batches:
                audio_batch = self._load_audio_batch(batch_paths)
                if audio_batch.batch_size > 0:
                    audio_batches.append(audio_batch)
        
        return audio_batches
    
    def _load_audio_batch(self, file_paths: List[Path]) -> AudioBatch:
        """
        Load a batch of audio files.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            AudioBatch with loaded audio data
        """
        batch = AudioBatch()
        
        for file_path in file_paths:
            try:
                # Load audio using librosa
                y, sr = librosa.load(file_path, sr=self.config.sample_rate)
                
                # Basic validation
                if len(y) == 0:
                    logger.warning(f"Empty audio file: {file_path}")
                    continue
                
                # Calculate duration
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Add to batch
                batch.audio_data.append(y)
                batch.sample_rates.append(sr)
                batch.file_paths.append(file_path)
                batch.durations.append(duration)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        batch.batch_size = len(batch.audio_data)
        return batch
    
    def _extract_features_parallel(self, audio_batches: List[AudioBatch]) -> List[List[Dict[str, Any]]]:
        """
        Extract features from audio batches in parallel.
        
        Args:
            audio_batches: List of AudioBatch objects
            
        Returns:
            List of feature dictionaries for each batch
        """
        batch_results = []
        
        if self.config.use_multiprocessing:
            # Use 'spawn' to avoid JAX fork-deadlock (see _load_audio_files_parallel).
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=mp.get_context('spawn'),
            ) as executor:
                future_to_batch = {
                    executor.submit(self._extract_batch_features, batch): batch
                    for batch in audio_batches
                }
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        features = future.result()
                        batch_results.append(features)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        batch_results.append([])
        else:
            # Fallback to sequential processing
            for batch in audio_batches:
                features = self._extract_batch_features(batch)
                batch_results.append(features)
        
        return batch_results
    
    def _extract_batch_features(self, batch: AudioBatch) -> List[Dict[str, Any]]:
        """
        Extract features from a single batch of audio files.
        
        This method implements vectorized feature extraction optimized
        for tensor processing units.
        
        Args:
            batch: AudioBatch with audio data
            
        Returns:
            List of feature dictionaries
        """
        if batch.batch_size == 0:
            return []
        
        batch_features = []
        
        # Convert to tensor format if enabled
        if self.config.enable_tensor_optimization:
            tensor_data = batch.to_tensor_format()
            # Process using vectorized operations
            batch_features = self._extract_features_vectorized(batch, tensor_data)
        else:
            # Process each file individually
            for i in range(batch.batch_size):
                features = self._extract_single_file_features(
                    batch.audio_data[i],
                    batch.sample_rates[i],
                    batch.file_paths[i],
                    batch.durations[i]
                )
                if features:
                    batch_features.append(features)
        
        return batch_features
    
    def _extract_features_vectorized(self, batch: AudioBatch, tensor_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract features using TT hardware when device='tenstorrent',
        otherwise fall back to per-file librosa extraction.
        """
        if self.config.device == 'tenstorrent':
            from .tensor_operations import TenstorrentTensorProcessor  # noqa: PLC0415

            processor = TenstorrentTensorProcessor()
            return processor.compute_features(
                tensor_data['audio_tensor'],
                tensor_data['lengths'],
                tensor_data['sample_rates'],
                batch.file_paths,
            )

        # CPU fallback — original per-file path
        features_list = []
        for i in range(batch.batch_size):
            features = self._extract_single_file_features(
                batch.audio_data[i],
                batch.sample_rates[i],
                batch.file_paths[i],
                batch.durations[i],
            )
            if features:
                features_list.append(features)
        return features_list
    
    def _extract_single_file_features(self, audio_data: np.ndarray, sample_rate: int, 
                                    file_path: Path, duration: float) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive features from a single audio file.
        
        This method maintains compatibility with the original feature extraction
        while optimizing for parallel processing.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate
            file_path: Path to the audio file
            duration: Duration in seconds
            
        Returns:
            Feature dictionary or None if extraction fails
        """
        # Use shared feature extraction core for consistency
        return extract_features_from_audio(audio_data, sample_rate, file_path, duration)
    
    # Note: Feature extraction methods moved to shared core in feature_extraction_base.py
    # This eliminates code duplication and ensures consistency across all processing approaches
    
    def estimate_processing_time(self, file_paths: List[Path]) -> Dict[str, float]:
        """
        Estimate processing time for a list of files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            Dictionary with time estimates
        """
        if not file_paths:
            return {'estimated_time': 0, 'per_file_time': 0}
        
        # Sample processing time estimation
        if self.processing_stats['average_file_time'] > 0:
            per_file_time = self.processing_stats['average_file_time']
        else:
            # Default estimate based on typical processing speed
            per_file_time = 0.5  # seconds per file
        
        # Account for parallelization speedup
        parallel_speedup = min(self.config.max_workers, len(file_paths))
        estimated_time = (len(file_paths) * per_file_time) / parallel_speedup
        
        return {
            'estimated_time': estimated_time,
            'per_file_time': per_file_time,
            'parallel_speedup': parallel_speedup,
            'total_files': len(file_paths)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'processing_stats': self.processing_stats,
            'config': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'use_multiprocessing': self.config.use_multiprocessing,
                'enable_tensor_optimization': self.config.enable_tensor_optimization
            }
        }


class StreamingFeatureExtractor:
    """
    Streaming feature extractor for processing large audio files in chunks.
    
    This class is designed for future hardware acceleration where large files
    can be processed in streaming fashion without loading entire files into memory.
    """
    
    def __init__(self, chunk_size_seconds: float = 30.0, overlap_seconds: float = 1.0):
        """
        Initialize streaming feature extractor.
        
        Args:
            chunk_size_seconds: Size of each processing chunk in seconds
            overlap_seconds: Overlap between chunks in seconds
        """
        self.chunk_size_seconds = chunk_size_seconds
        self.overlap_seconds = overlap_seconds
        self.extractor = ParallelFeatureExtractor()
    
    def extract_features_streaming(self, file_path: Path, 
                                 sample_rate: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract features from a large audio file using streaming approach.
        
        Args:
            file_path: Path to the audio file
            sample_rate: Target sample rate
            
        Returns:
            List of feature dictionaries for each chunk
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate chunk parameters
            chunk_samples = int(self.chunk_size_seconds * sr)
            overlap_samples = int(self.overlap_seconds * sr)
            hop_samples = chunk_samples - overlap_samples
            
            # Extract features from each chunk
            chunk_features = []
            for start in range(0, len(y), hop_samples):
                end = min(start + chunk_samples, len(y))
                chunk_y = y[start:end]
                
                # Skip very short chunks
                if len(chunk_y) < sr:  # Less than 1 second
                    continue
                
                # Extract features for this chunk using shared core
                chunk_duration = len(chunk_y) / sr
                features = extract_features_from_audio(
                    chunk_y, sr, file_path, chunk_duration
                )
                
                if features:
                    features['chunk_start'] = start / sr
                    features['chunk_end'] = end / sr
                    features['chunk_index'] = len(chunk_features)
                    chunk_features.append(features)
            
            return chunk_features
            
        except Exception as e:
            logger.error(f"Error in streaming extraction for {file_path}: {e}")
            return []
    
    def aggregate_chunk_features(self, chunk_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate features from multiple chunks into a single representation.
        
        Args:
            chunk_features: List of feature dictionaries from chunks
            
        Returns:
            Aggregated feature dictionary
        """
        if not chunk_features:
            return {}
        
        # Get numeric features from first chunk as template
        first_chunk = chunk_features[0]
        numeric_features = {k: v for k, v in first_chunk.items() 
                          if isinstance(v, (int, float)) and k not in ['chunk_start', 'chunk_end', 'chunk_index']}
        
        # Aggregate numeric features
        aggregated = {}
        for feature_name in numeric_features.keys():
            values = [chunk.get(feature_name, 0) for chunk in chunk_features 
                     if feature_name in chunk and isinstance(chunk[feature_name], (int, float))]
            
            if values:
                aggregated[f'{feature_name}_mean'] = float(np.mean(values))
                aggregated[f'{feature_name}_std'] = float(np.std(values))
                aggregated[f'{feature_name}_min'] = float(np.min(values))
                aggregated[f'{feature_name}_max'] = float(np.max(values))
        
        # Add metadata
        aggregated['filename'] = first_chunk.get('filename', 'unknown')
        aggregated['total_chunks'] = len(chunk_features)
        aggregated['total_duration'] = sum(chunk.get('duration', 0) for chunk in chunk_features)
        
        return aggregated