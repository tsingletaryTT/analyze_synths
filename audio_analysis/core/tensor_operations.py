"""
Tensor Operations Module

This module provides tensor-optimized data structures and operations designed
for efficient processing on specialized hardware including Tenstorrent processors.

Key Features:
1. Tensor-friendly data layouts for optimal memory access
2. Vectorized operations for parallel processing
3. Batch processing primitives for hardware acceleration
4. Memory-efficient data structures for large datasets
5. Hardware-agnostic interface for portability

The design separates logical operations from hardware-specific implementations,
allowing for easy adaptation to different acceleration platforms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TensorBatch:
    """
    Optimized tensor batch for hardware acceleration.
    
    This structure organizes data in memory-efficient layouts optimized
    for tensor processing units. All arrays use consistent dtypes and
    shapes for vectorized operations.
    """
    # Audio data tensor: (batch_size, max_length)
    audio_data: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata tensors: (batch_size,)
    lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    durations: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Feature tensors: (batch_size, feature_dim)
    spectral_features: np.ndarray = field(default_factory=lambda: np.array([]))
    temporal_features: np.ndarray = field(default_factory=lambda: np.array([]))
    harmonic_features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Identifiers
    filenames: List[str] = field(default_factory=list)
    batch_size: int = 0
    
    def __post_init__(self):
        """Validate tensor consistency after initialization."""
        if len(self.filenames) > 0:
            self.batch_size = len(self.filenames)
            self._validate_tensor_shapes()
    
    def _validate_tensor_shapes(self):
        """Validate that all tensors have consistent batch dimensions."""
        expected_shape = (self.batch_size,)
        
        if self.lengths.size > 0 and self.lengths.shape != expected_shape:
            raise ValueError(f"Length tensor shape {self.lengths.shape} != expected {expected_shape}")
        
        if self.sample_rates.size > 0 and self.sample_rates.shape != expected_shape:
            raise ValueError(f"Sample rate tensor shape {self.sample_rates.shape} != expected {expected_shape}")
        
        if self.durations.size > 0 and self.durations.shape != expected_shape:
            raise ValueError(f"Duration tensor shape {self.durations.shape} != expected {expected_shape}")
        
        # Validate feature tensors have correct batch dimension
        if self.spectral_features.size > 0 and self.spectral_features.shape[0] != self.batch_size:
            raise ValueError(f"Spectral features batch dim {self.spectral_features.shape[0]} != {self.batch_size}")
    
    def to_device_format(self, device: str = "cpu") -> Dict[str, np.ndarray]:
        """
        Convert batch to device-specific format.
        
        This method prepares data for hardware-specific processing.
        Currently supports CPU format, but can be extended for other devices.
        
        Args:
            device: Target device ("cpu", "tenstorrent", "gpu")
            
        Returns:
            Dictionary with device-optimized tensors
        """
        if device == "cpu":
            return self._to_cpu_format()
        elif device == "tenstorrent":
            return self._to_tenstorrent_format()
        else:
            raise ValueError(f"Unsupported device: {device}")
    
    def _to_cpu_format(self) -> Dict[str, np.ndarray]:
        """Convert to CPU-optimized format."""
        return {
            'audio_data': self.audio_data.astype(np.float32),
            'lengths': self.lengths.astype(np.int32),
            'sample_rates': self.sample_rates.astype(np.int32),
            'durations': self.durations.astype(np.float32),
            'spectral_features': self.spectral_features.astype(np.float32),
            'temporal_features': self.temporal_features.astype(np.float32),
            'harmonic_features': self.harmonic_features.astype(np.float32)
        }
    
    def _to_tenstorrent_format(self) -> Dict[str, np.ndarray]:
        """
        Convert to Tenstorrent-optimized format.
        
        Tenstorrent processors work optimally with specific tensor layouts
        and data types. This method prepares data accordingly.
        """
        # Ensure optimal tensor layouts for Tenstorrent
        # Tenstorrent processors prefer certain alignment and data types
        
        # Pad tensors to optimal sizes (powers of 2 when possible)
        def pad_to_optimal_size(tensor: np.ndarray, axis: int) -> np.ndarray:
            """Pad tensor along specified axis to optimal size."""
            current_size = tensor.shape[axis]
            optimal_size = 2 ** int(np.ceil(np.log2(current_size)))
            
            if current_size == optimal_size:
                return tensor
            
            pad_width = [(0, 0)] * tensor.ndim
            pad_width[axis] = (0, optimal_size - current_size)
            
            return np.pad(tensor, pad_width, mode='constant', constant_values=0)
        
        # Optimize tensor layouts
        optimized_data = {}
        
        if self.audio_data.size > 0:
            # Pad audio data to optimal length
            audio_padded = pad_to_optimal_size(self.audio_data, axis=1)
            optimized_data['audio_data'] = audio_padded.astype(np.float32)
        
        if self.spectral_features.size > 0:
            # Pad feature tensors to optimal dimensions
            spectral_padded = pad_to_optimal_size(self.spectral_features, axis=1)
            optimized_data['spectral_features'] = spectral_padded.astype(np.float32)
        
        if self.temporal_features.size > 0:
            temporal_padded = pad_to_optimal_size(self.temporal_features, axis=1)
            optimized_data['temporal_features'] = temporal_padded.astype(np.float32)
        
        if self.harmonic_features.size > 0:
            harmonic_padded = pad_to_optimal_size(self.harmonic_features, axis=1)
            optimized_data['harmonic_features'] = harmonic_padded.astype(np.float32)
        
        # Metadata tensors
        optimized_data.update({
            'lengths': self.lengths.astype(np.int32),
            'sample_rates': self.sample_rates.astype(np.int32),
            'durations': self.durations.astype(np.float32)
        })
        
        return optimized_data


class TensorProcessor(ABC):
    """
    Abstract base class for tensor processors.
    
    This class defines the interface for hardware-specific tensor operations,
    allowing for easy adaptation to different acceleration platforms.
    """
    
    @abstractmethod
    def process_batch(self, batch: TensorBatch) -> TensorBatch:
        """Process a tensor batch."""
        pass
    
    @abstractmethod
    def compute_features(self, audio_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute features from audio tensor."""
        pass
    
    @abstractmethod
    def cluster_features(self, features: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform clustering on feature tensors."""
        pass


class CPUTensorProcessor(TensorProcessor):
    """
    CPU-optimized tensor processor.
    
    This implementation uses optimized NumPy operations for CPU processing.
    """
    
    def __init__(self, num_threads: int = -1):
        """
        Initialize CPU tensor processor.
        
        Args:
            num_threads: Number of threads to use (-1 for all available)
        """
        self.num_threads = num_threads
        if num_threads > 0:
            # Set NumPy thread count for optimal performance
            import os
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    def process_batch(self, batch: TensorBatch) -> TensorBatch:
        """
        Process a tensor batch using CPU-optimized operations.
        
        Args:
            batch: Input tensor batch
            
        Returns:
            Processed tensor batch
        """
        device_data = batch.to_device_format("cpu")
        
        # Process audio data if available
        if 'audio_data' in device_data and device_data['audio_data'].size > 0:
            processed_audio = self._process_audio_tensor(device_data['audio_data'])
            
            # Update batch with processed data
            batch.audio_data = processed_audio
        
        return batch
    
    def compute_features(self, audio_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute features from audio tensor using vectorized operations.
        
        Args:
            audio_tensor: Audio data tensor (batch_size, max_length)
            
        Returns:
            Dictionary of feature tensors
        """
        batch_size = audio_tensor.shape[0]
        
        # Initialize feature tensors
        spectral_features = np.zeros((batch_size, 4), dtype=np.float32)  # 4 spectral features
        temporal_features = np.zeros((batch_size, 3), dtype=np.float32)  # 3 temporal features
        
        # Process each audio signal in the batch
        for i in range(batch_size):
            audio_signal = audio_tensor[i]
            
            # Remove padding (assuming non-zero values are actual audio)
            valid_length = np.count_nonzero(audio_signal)
            if valid_length > 0:
                audio_signal = audio_signal[:valid_length]
                
                # Compute spectral features
                spectral_features[i] = self._compute_spectral_features(audio_signal)
                
                # Compute temporal features
                temporal_features[i] = self._compute_temporal_features(audio_signal)
        
        return {
            'spectral_features': spectral_features,
            'temporal_features': temporal_features
        }
    
    def cluster_features(self, features: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering using CPU-optimized operations.
        
        Args:
            features: Feature tensor (n_samples, n_features)
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        from sklearn.cluster import KMeans
        
        # Use sklearn's optimized KMeans implementation
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return labels, kmeans.cluster_centers_
    
    def _process_audio_tensor(self, audio_tensor: np.ndarray) -> np.ndarray:
        """Apply audio processing to tensor."""
        # Normalize audio tensor
        normalized = audio_tensor / (np.max(np.abs(audio_tensor), axis=1, keepdims=True) + 1e-8)
        
        # Apply basic filtering (high-pass filter to remove DC component)
        # This is a simple implementation - could be enhanced with more sophisticated filtering
        filtered = normalized - np.mean(normalized, axis=1, keepdims=True)
        
        return filtered.astype(np.float32)
    
    def _compute_spectral_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Compute spectral features for a single audio signal."""
        # Simple spectral feature computation
        # In practice, this would use more sophisticated methods
        
        # Compute FFT
        fft = np.fft.fft(audio_signal)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral centroid (simplified)
        freqs = np.arange(len(magnitude))
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        
        # Spectral rolloff (90% of energy)
        cumulative_energy = np.cumsum(magnitude)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.9 * total_energy)[0]
        spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(magnitude) - 1
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_signal)))) / len(audio_signal)
        
        return np.array([spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossings], dtype=np.float32)
    
    def _compute_temporal_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Compute temporal features for a single audio signal."""
        # RMS energy
        rms_energy = np.sqrt(np.mean(audio_signal ** 2))
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio_signal))
        
        # Dynamic range
        dynamic_range = np.max(audio_signal) - np.min(audio_signal)
        
        return np.array([rms_energy, peak_amplitude, dynamic_range], dtype=np.float32)


class TenstorrentTensorProcessor(TensorProcessor):
    """
    Tenstorrent-optimized tensor processor.
    
    This implementation is designed for future Tenstorrent hardware.
    Currently provides a framework for Tenstorrent-specific optimizations.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize Tenstorrent tensor processor.
        
        Args:
            device_id: Tenstorrent device identifier
        """
        self.device_id = device_id
        logger.info(f"Initialized Tenstorrent processor for device {device_id}")
    
    def process_batch(self, batch: TensorBatch) -> TensorBatch:
        """
        Process tensor batch using Tenstorrent-optimized operations.
        
        This is a placeholder implementation that demonstrates the interface.
        In practice, this would use Tenstorrent's SDK and optimized operations.
        """
        # Convert to Tenstorrent format
        device_data = batch.to_device_format("tenstorrent")
        
        # Placeholder for Tenstorrent-specific processing
        # In practice, this would use Tenstorrent's neural network compiler
        logger.info(f"Processing batch of size {batch.batch_size} on Tenstorrent device {self.device_id}")
        
        return batch
    
    def compute_features(
        self,
        audio_tensor: np.ndarray,
        lengths: np.ndarray,
        sample_rates: np.ndarray,
        file_paths,
    ):
        """Compute features using JaxAudioFeatureExtractor on TT hardware."""
        from .jax_feature_extraction import JaxAudioFeatureExtractor  # noqa: PLC0415

        sr = int(np.bincount(sample_rates).argmax())
        extractor = JaxAudioFeatureExtractor(sr=sr)
        return extractor.extract_batch(audio_tensor, lengths, sr, file_paths)

    def cluster_features(
        self,
        features: np.ndarray,
        n_clusters: int,
    ):
        """Cluster features via JAX k-means on TT hardware."""
        from .jax_feature_extraction import jax_kmeans  # noqa: PLC0415

        return jax_kmeans(features, n_clusters)


class TensorProcessorFactory:
    """Factory class for creating tensor processors."""
    
    @staticmethod
    def create_processor(device: str = "cpu", **kwargs) -> TensorProcessor:
        """
        Create a tensor processor for the specified device.
        
        Args:
            device: Target device ("cpu", "tenstorrent")
            **kwargs: Device-specific configuration
            
        Returns:
            TensorProcessor instance
        """
        if device == "cpu":
            return CPUTensorProcessor(**kwargs)
        elif device == "tenstorrent":
            return TenstorrentTensorProcessor(**kwargs)
        else:
            raise ValueError(f"Unsupported device: {device}")


class TensorFeatureExtractor:
    """
    High-level tensor-based feature extraction interface.
    
    This class provides a unified interface for tensor-based feature extraction
    across different hardware platforms.
    """
    
    def __init__(self, device: str = "cpu", **device_kwargs):
        """
        Initialize tensor feature extractor.
        
        Args:
            device: Target device for processing
            **device_kwargs: Device-specific configuration
        """
        self.device = device
        self.processor = TensorProcessorFactory.create_processor(device, **device_kwargs)
        logger.info(f"Initialized TensorFeatureExtractor for device: {device}")
    
    def extract_features_from_paths(self, file_paths: List[Path], 
                                   batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Extract features from audio files using tensor operations.
        
        Args:
            file_paths: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_features = self._process_batch_paths(batch_paths)
            features_list.extend(batch_features)
        
        return features_list
    
    def _process_batch_paths(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of file paths."""
        try:
            # Load audio files into tensor batch
            batch = self._load_audio_batch(file_paths)
            
            # Process batch using tensor operations
            processed_batch = self.processor.process_batch(batch)
            
            # Extract features from processed batch
            feature_tensors = self.processor.compute_features(processed_batch.audio_data)
            
            # Convert tensor results to feature dictionaries
            return self._tensors_to_features(feature_tensors, processed_batch)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    def _load_audio_batch(self, file_paths: List[Path]) -> TensorBatch:
        """Load audio files into a tensor batch."""
        import librosa
        
        audio_data = []
        sample_rates = []
        durations = []
        filenames = []
        
        for file_path in file_paths:
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                audio_data.append(y)
                sample_rates.append(sr)
                durations.append(duration)
                filenames.append(file_path.name)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not audio_data:
            return TensorBatch()
        
        # Pad audio data to same length
        max_length = max(len(audio) for audio in audio_data)
        padded_audio = np.zeros((len(audio_data), max_length), dtype=np.float32)
        lengths = np.zeros(len(audio_data), dtype=np.int32)
        
        for i, audio in enumerate(audio_data):
            audio_len = len(audio)
            padded_audio[i, :audio_len] = audio.astype(np.float32)
            lengths[i] = audio_len
        
        return TensorBatch(
            audio_data=padded_audio,
            lengths=lengths,
            sample_rates=np.array(sample_rates, dtype=np.int32),
            durations=np.array(durations, dtype=np.float32),
            filenames=filenames
        )
    
    def _tensors_to_features(self, feature_tensors: Dict[str, np.ndarray], 
                           batch: TensorBatch) -> List[Dict[str, Any]]:
        """Convert tensor results to feature dictionaries."""
        features_list = []
        
        for i in range(batch.batch_size):
            features = {
                'filename': batch.filenames[i],
                'duration': float(batch.durations[i]),
                'sample_rate': int(batch.sample_rates[i])
            }
            
            # Add spectral features
            if 'spectral_features' in feature_tensors:
                spectral = feature_tensors['spectral_features'][i]
                features.update({
                    'spectral_centroid_mean': float(spectral[0]),
                    'spectral_rolloff_mean': float(spectral[1]),
                    'spectral_bandwidth_mean': float(spectral[2]),
                    'zero_crossing_rate_mean': float(spectral[3])
                })
            
            # Add temporal features
            if 'temporal_features' in feature_tensors:
                temporal = feature_tensors['temporal_features'][i]
                features.update({
                    'rms_mean': float(temporal[0]),
                    'peak_amplitude': float(temporal[1]),
                    'dynamic_range': float(temporal[2])
                })
            
            features_list.append(features)
        
        return features_list
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the processing device."""
        return {
            'device': self.device,
            'processor_type': type(self.processor).__name__
        }