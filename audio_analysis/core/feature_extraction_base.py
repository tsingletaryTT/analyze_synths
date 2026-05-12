"""
Base Feature Extraction Module

This module provides the core feature extraction logic that is shared between
traditional and parallel processing approaches. It separates the analytical
logic from the processing methodology to eliminate code duplication.

Key Design Principles:
1. Single source of truth for all feature extraction algorithms
2. Shared analytical logic between traditional and parallel approaches
3. Easy extension for new mood descriptors and character analysis
4. Hardware-agnostic feature computation functions
5. Comprehensive documentation for creative relevance

When adding new moods, textures, or features, modifications should be made
only in this module to ensure consistency across all processing approaches.
"""

import librosa
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureExtractionCore:
    """
    Core feature extraction algorithms shared between traditional and parallel approaches.
    
    This class contains the fundamental audio analysis logic that is independent
    of processing methodology (sequential vs parallel vs hardware-accelerated).
    All feature extraction approaches should use these core methods to ensure
    consistency and eliminate code duplication.
    """
    
    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize the feature extraction core.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
        # Core feature extraction parameters
        self.feature_params = {
            'n_mfcc': 13,
            'n_chroma': 12,
            'n_tonnetz': 6,
            'hop_length': 512,
            'n_fft': 2048
        }
        
        # Musical keys for harmonic analysis
        self.musical_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features that capture the "brightness" and "color" of sound.
        
        These features are crucial for understanding synthesizer textures and are
        used consistently across all processing approaches.
        
        Technical Analysis:
        - Spectral centroid: "brightness" of the sound (higher = brighter, more treble)
        - Spectral rolloff: frequency below which 85% of energy is concentrated
        - Spectral bandwidth: "width" of frequency distribution (narrow = pure tones)
        - Zero-crossing rate: measure of signal "roughness" or "noisiness"
        
        Creative Relevance:
        - Spectral centroid distinguishes between different synthesizer patches
        - Spectral rolloff indicates synthesizer filter settings and harmonic fullness
        - Spectral bandwidth characterizes synthesizer sound design complexity
        - Zero-crossing rate indicates synthesizer texture and filter resonance
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        # Compute spectral features in batch for efficiency
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
        }
    
    def extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract temporal features that capture energy and rhythm characteristics.
        
        These features define the musical "feel" and emotional impact of the sound.
        
        Technical Analysis:
        - RMS Energy: overall "loudness" and dynamic range distribution
        - Tempo: rhythmic characteristics using onset detection
        - Beat tracking: rhythmic patterns and density
        
        Creative Relevance:
        - RMS energy captures musical dynamics and emotional intensity
        - Tempo and beat tracking are essential for musical flow and sequencing
        - Onset density distinguishes between rhythmic and ambient textures
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of temporal features
        """
        # RMS Energy: overall "loudness" and dynamic range
        # Not just volume, but energy distribution over time
        # Critical for understanding musical dynamics and emotional intensity
        rms = librosa.feature.rms(y=y)[0]
        
        # Tempo and beat tracking: rhythmic characteristics
        # Essential for understanding musical flow and sequencing
        # Uses onset detection to find rhythmic patterns
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Calculate onset density for rhythm characterization
        # Higher density = more rhythmic activity
        # Lower density = more sustained, ambient textures
        onset_density = len(beats) / (len(y) / sr) if len(y) > 0 else 0
        
        # Extract scalar tempo value (librosa returns array, need first element)
        tempo_scalar = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'tempo': tempo_scalar,
            'onset_density': float(onset_density),
        }
    
    def extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract harmonic features that capture musical content and key information.
        
        These features are essential for understanding compositional structure
        and harmonic relationships between tracks.
        
        Technical Analysis:
        - MFCC: Mel-frequency cepstral coefficients for timbre fingerprinting
        - Chroma: Harmonic and melodic content by pitch class
        - Tonnetz: Tonal centroid features for harmonic relationships
        - Key detection: Predominant key estimation from chroma features
        
        Creative Relevance:
        - MFCCs capture synthesizer "character" independent of pitch
        - Chroma features enable key detection and harmonic analysis
        - Tonnetz captures harmonic relationships and chord progressions
        - Key detection is crucial for sequencing and harmonic compatibility
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of harmonic features
        """
        # MFCC (Mel-frequency cepstral coefficients): timbre fingerprint
        # Captures the "character" of the sound independent of pitch
        # First 13 coefficients are most musically relevant
        # Critical for distinguishing between different synthesizer types
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.feature_params['n_mfcc'])
        
        # Chroma features: harmonic and melodic content
        # Captures the "pitch class" distribution (C, C#, D, etc.)
        # Essential for key detection and harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tonnetz: tonal centroid features
        # Captures harmonic relationships and chord progressions
        # Based on music theory concepts of tonal space
        # Requires harmonic component extraction for accuracy
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Key detection using chroma features
        # Uses chroma features to estimate the predominant key
        # Important for sequencing and harmonic compatibility analysis
        chroma_mean = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_mean)
        detected_key = self.musical_keys[key_index]
        key_confidence = float(np.max(chroma_mean))

        # Compile harmonic features
        # Field name is 'key' (spec-compliant); 'detected_key' is an old alias
        # preserved for backward compatibility via the reference update below.
        features = {
            'key': detected_key,
            'key_confidence': key_confidence,
        }
        
        # MFCC statistics - capture timbre characteristics
        for i in range(self.feature_params['n_mfcc']):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
        
        # Chroma statistics - capture harmonic content by pitch class
        for i, key in enumerate(self.musical_keys):
            features[f'chroma_{key}_mean'] = float(np.mean(chroma[i]))
        
        # Tonnetz statistics - capture harmonic relationships
        for i in range(self.feature_params['n_tonnetz']):
            features[f'tonnetz_{i+1}_mean'] = float(np.mean(tonnetz[i]))
        
        return features
    
    def extract_comprehensive_features(self, y: np.ndarray, sr: int, 
                                     file_path: Path, duration: float) -> Dict[str, Any]:
        """
        Extract comprehensive features from audio data using all extraction methods.
        
        This is the main feature extraction method that combines all analysis
        approaches into a complete feature set. It serves as the single source
        of truth for feature extraction logic.
        
        Args:
            y: Audio time series
            sr: Sample rate
            file_path: Path to the audio file
            duration: Duration in seconds
            
        Returns:
            Dictionary with comprehensive features
        """
        # Basic file information
        features = {
            'filename': file_path.name,
            'duration': duration,
        }
        
        # Extract all feature types
        spectral_features = self.extract_spectral_features(y, sr)
        temporal_features = self.extract_temporal_features(y, sr)
        harmonic_features = self.extract_harmonic_features(y, sr)
        
        # Combine all features
        features.update(spectral_features)
        features.update(temporal_features)
        features.update(harmonic_features)
        
        return features
    
    def extract_basic_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract basic spectral features for use in phase analysis.
        
        This is a lightweight version used when we need features for individual
        phases/sections rather than entire tracks. Focuses on essential characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with basic spectral features
        """
        # Calculate core spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
        }
    
    def get_numeric_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract only numeric features suitable for machine learning.
        
        This method filters out string/categorical features and returns
        only the numeric values that can be used for clustering and
        statistical analysis.
        
        Args:
            features: Full feature dictionary
            
        Returns:
            Dictionary containing only numeric features
        """
        numeric_features = {}
        
        for key, value in features.items():
            # Skip non-numeric features
            if key in ['filename', 'key']:
                continue
                
            # Include numeric values
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_features[key] = float(value)
                
        return numeric_features
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that extracted features are complete and valid.
        
        Args:
            features: Feature dictionary to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        # Check for required features
        required_features = [
            'spectral_centroid_mean', 'spectral_rolloff_mean', 'spectral_bandwidth_mean',
            'zero_crossing_rate_mean', 'rms_mean', 'tempo', 'key'
        ]
        
        for feature in required_features:
            if feature not in features:
                logger.warning(f"Missing required feature: {feature}")
                return False
        
        # Check for NaN or infinite values
        for key, value in features.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid value for feature {key}: {value}")
                    return False
        
        return True
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get human-readable descriptions of all extracted features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            # Spectral features
            'spectral_centroid_mean': 'Average brightness/treble content of the sound',
            'spectral_centroid_std': 'Variation in brightness over time',
            'spectral_rolloff_mean': 'Average frequency rolloff (harmonic fullness)',
            'spectral_bandwidth_mean': 'Average spectral width (tonal complexity)',
            'zero_crossing_rate_mean': 'Average signal roughness/noisiness',
            
            # Temporal features
            'rms_mean': 'Average energy/loudness level',
            'rms_std': 'Variation in energy over time (dynamics)',
            'tempo': 'Detected tempo in beats per minute',
            'onset_density': 'Rhythmic activity density',
            
            # Harmonic features
            'key': 'Predominant musical key',
            'key_confidence': 'Confidence in key detection',
            
            # MFCC features (timbre)
            **{f'mfcc_{i+1}_mean': f'MFCC coefficient {i+1} mean (timbre characteristic)' 
               for i in range(self.feature_params['n_mfcc'])},
            **{f'mfcc_{i+1}_std': f'MFCC coefficient {i+1} std (timbre variation)' 
               for i in range(self.feature_params['n_mfcc'])},
            
            # Chroma features (harmonic content)
            **{f'chroma_{key}_mean': f'Average {key} pitch class strength' 
               for key in self.musical_keys},
            
            # Tonnetz features (harmonic relationships)
            **{f'tonnetz_{i+1}_mean': f'Tonnetz component {i+1} (harmonic relationship)' 
               for i in range(self.feature_params['n_tonnetz'])},
        }


# Global instance for shared use
feature_extraction_core = FeatureExtractionCore()


def extract_features_from_audio(audio_data: np.ndarray, sample_rate: int, 
                               file_path: Path, duration: float) -> Optional[Dict[str, Any]]:
    """
    Global function to extract features from audio data.
    
    This function provides a consistent interface for feature extraction
    that can be used by both traditional and parallel processing approaches.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sample rate
        file_path: Path to the audio file
        duration: Duration in seconds
        
    Returns:
        Dictionary with comprehensive features or None if extraction fails
    """
    try:
        features = feature_extraction_core.extract_comprehensive_features(
            audio_data, sample_rate, file_path, duration
        )
        
        # Validate features
        if not feature_extraction_core.validate_features(features):
            logger.warning(f"Invalid features extracted from {file_path}")
            return None
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {str(e)}")
        return None


def extract_basic_spectral_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Global function to extract basic spectral features for phase analysis.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sample rate
        
    Returns:
        Dictionary with basic spectral features
    """
    return feature_extraction_core.extract_basic_spectral_features(audio_data, sample_rate)


def get_numeric_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Global function to extract numeric features for machine learning.
    
    Args:
        features: Full feature dictionary
        
    Returns:
        Dictionary containing only numeric features
    """
    return feature_extraction_core.get_numeric_features(features)