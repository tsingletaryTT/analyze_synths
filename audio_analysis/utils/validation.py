"""
Validation Utilities for Audio Analysis

This module provides centralized validation functions used throughout
the audio analysis toolkit. It eliminates code duplication for range
checking, data validation, and other common validation patterns.
"""

from typing import Dict, Any, Tuple, List, Optional
from .type_conversion import safe_float_convert


def validate_range(value: Any, range_tuple: Tuple[float, float]) -> bool:
    """
    Check if a value falls within the specified range.
    
    This function handles both finite and infinite ranges, allowing
    for flexible descriptor definitions. It also ensures proper type
    conversion to prevent comparison errors.
    
    Args:
        value: Value to check
        range_tuple: (min, max) tuple defining the range
        
    Returns:
        True if value is in range, False otherwise
    """
    min_val, max_val = range_tuple
    
    # Ensure all values are numeric for comparison
    value = safe_float_convert(value, 0.0)
    min_val = safe_float_convert(min_val, float('-inf'))
    max_val = safe_float_convert(max_val, float('inf'))
    
    return min_val <= value <= max_val


def validate_feature_ranges(features_dict: Dict[str, Any], 
                          range_definitions: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    """
    Validate multiple features against their defined ranges.
    
    Args:
        features_dict: Dictionary of feature values
        range_definitions: Dictionary mapping feature names to (min, max) ranges
        
    Returns:
        Dictionary mapping feature names to validation results
    """
    validation_results = {}
    
    for feature_name, range_tuple in range_definitions.items():
        if feature_name in features_dict:
            validation_results[feature_name] = validate_range(
                features_dict[feature_name], range_tuple
            )
        else:
            validation_results[feature_name] = False
    
    return validation_results


def validate_phase_data(phase_data: Dict[str, Any]) -> List[str]:
    """
    Validate phase data dictionary for required fields and reasonable values.
    
    Args:
        phase_data: Phase data dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    required_fields = ['avg_energy', 'avg_brightness', 'avg_roughness', 'onset_density', 'duration']
    
    # Check for required fields
    for field in required_fields:
        if field not in phase_data:
            errors.append(f"Missing required field: {field}")
            continue
        
        # Check for reasonable values
        value = safe_float_convert(phase_data[field], None)
        if value is None:
            errors.append(f"Invalid numeric value for {field}: {phase_data[field]}")
            continue
        
        # Field-specific validation
        if field == 'duration' and value <= 0:
            errors.append(f"Duration must be positive, got: {value}")
        elif field in ['avg_energy', 'avg_brightness', 'onset_density'] and value < 0:
            errors.append(f"{field} cannot be negative, got: {value}")
    
    return errors


def validate_spectral_features(spectral_features: Dict[str, Any]) -> List[str]:
    """
    Validate spectral features dictionary for required fields and reasonable values.
    
    Args:
        spectral_features: Spectral features dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    required_fields = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 
                      'spectral_rolloff_mean', 'zero_crossing_rate_mean']
    
    # Check for required fields
    for field in required_fields:
        if field not in spectral_features:
            errors.append(f"Missing required spectral feature: {field}")
            continue
        
        # Check for reasonable values
        value = safe_float_convert(spectral_features[field], None)
        if value is None:
            errors.append(f"Invalid numeric value for {field}: {spectral_features[field]}")
            continue
        
        # Field-specific validation
        if field == 'zero_crossing_rate_mean' and not (0 <= value <= 1):
            errors.append(f"Zero crossing rate should be between 0 and 1, got: {value}")
        elif field in ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean'] and value < 0:
            errors.append(f"{field} cannot be negative, got: {value}")
    
    return errors


def validate_audio_files_list(audio_files: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of audio file dictionaries for MCP server use.
    
    Args:
        audio_files: List of audio file dictionaries
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    supported_formats = ['.wav', '.aiff', '.aif', '.mp3']
    
    if not audio_files:
        errors.append("No audio files provided")
        return errors
    
    for i, file_data in enumerate(audio_files):
        file_prefix = f"File {i+1}"
        
        # Check required fields
        if 'filename' not in file_data:
            errors.append(f"{file_prefix}: Missing 'filename' field")
            continue
            
        if 'content' not in file_data:
            errors.append(f"{file_prefix}: Missing 'content' field")
            continue
        
        filename = file_data['filename']
        
        # Check file extension
        if not any(filename.lower().endswith(ext) for ext in supported_formats):
            errors.append(f"{file_prefix} ({filename}): Unsupported format. Supported: {', '.join(supported_formats)}")
        
        # Basic content validation
        try:
            import base64
            content = file_data['content']
            decoded_content = base64.b64decode(content)
            
            # Check for reasonable file size
            if len(decoded_content) < 1000:  # Less than 1KB
                errors.append(f"{file_prefix} ({filename}): File appears too small to be valid audio")
            elif len(decoded_content) > 100 * 1024 * 1024:  # More than 100MB
                errors.append(f"{file_prefix} ({filename}): File too large (>100MB)")
                
        except Exception as e:
            errors.append(f"{file_prefix} ({filename}): Invalid base64 content - {str(e)}")
    
    return errors


def validate_clustering_parameters(n_clusters: Optional[int], n_samples: int) -> Tuple[int, List[str]]:
    """
    Validate and adjust clustering parameters based on data constraints.
    
    Args:
        n_clusters: Requested number of clusters (None for auto-determination)
        n_samples: Number of data samples available
        
    Returns:
        Tuple of (adjusted_n_clusters, validation_warnings)
    """
    warnings = []
    
    # Auto-determine clusters if not specified
    if n_clusters is None:
        if n_samples <= 2:
            n_clusters = 1
            warnings.append("Too few samples for meaningful clustering, using 1 cluster")
        elif n_samples <= 5:
            n_clusters = min(2, n_samples)
            warnings.append(f"Small dataset, using {n_clusters} clusters")
        else:
            n_clusters = min(5, max(2, n_samples // 3))
            warnings.append(f"Auto-determined {n_clusters} clusters for {n_samples} samples")
    else:
        # Validate provided cluster count
        if n_clusters > n_samples:
            n_clusters = n_samples
            warnings.append(f"Reduced cluster count to {n_clusters} (cannot exceed number of samples)")
        elif n_clusters < 1:
            n_clusters = 1
            warnings.append("Cluster count must be at least 1")
    
    return n_clusters, warnings


def validate_sequence_data(tracks_data: List[Dict[str, Any]]) -> List[str]:
    """
    Validate track data for sequence recommendation.
    
    Args:
        tracks_data: List of track dictionaries
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    # Accept both 'key' (spec-compliant) and legacy 'detected_key' field name
    required_fields = ['filename', 'duration', 'tempo', 'key',
                      'primary_mood', 'primary_character', 'rms_mean']
    
    if not tracks_data:
        errors.append("No track data provided for sequencing")
        return errors
    
    for i, track in enumerate(tracks_data):
        track_prefix = f"Track {i+1}"
        
        # Check required fields
        for field in required_fields:
            if field not in track:
                errors.append(f"{track_prefix}: Missing required field '{field}'")
        
        # Validate specific field types and ranges
        if 'duration' in track:
            duration = safe_float_convert(track['duration'], None)
            if duration is None or duration <= 0:
                errors.append(f"{track_prefix}: Invalid duration value")
        
        if 'tempo' in track:
            tempo = safe_float_convert(track['tempo'], None)
            if tempo is None or not (30 <= tempo <= 300):
                errors.append(f"{track_prefix}: Tempo should be between 30-300 BPM")
    
    return errors