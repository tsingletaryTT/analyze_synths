"""
Statistical Utilities for Audio Analysis

This module provides centralized statistical calculation functions that are
used across multiple components of the audio analysis system. It eliminates
code duplication by providing shared implementations of common statistical
operations.

Key Functions:
- Summary statistics calculation
- Data aggregation functions
- Statistical analysis helpers
- Data preparation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def calculate_phase_statistics(phase_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for phase analysis data.
    
    This function computes key metrics about phases across all analyzed files,
    providing insights into structural characteristics and phase distributions.
    
    Args:
        phase_data: List of phase analysis results from multiple files
        
    Returns:
        Dictionary containing phase statistics
    """
    if not phase_data:
        return {
            'total_phases': 0,
            'avg_phases_per_track': 0,
            'total_files': 0
        }
    
    # Extract phase counts from all files
    phase_counts = [len(file_info['phases']) for file_info in phase_data]
    
    # Calculate basic statistics
    total_phases = sum(phase_counts)
    total_files = len(phase_data)
    avg_phases_per_track = np.mean(phase_counts) if phase_counts else 0
    
    # Calculate phase duration statistics
    all_phase_durations = []
    for file_info in phase_data:
        for phase in file_info['phases']:
            all_phase_durations.append(phase['duration'])
    
    duration_stats = {
        'avg_phase_duration': np.mean(all_phase_durations) if all_phase_durations else 0,
        'median_phase_duration': np.median(all_phase_durations) if all_phase_durations else 0,
        'std_phase_duration': np.std(all_phase_durations) if all_phase_durations else 0
    }
    
    # Calculate phase type distribution
    phase_types = []
    for file_info in phase_data:
        for phase in file_info['phases']:
            phase_types.append(phase.get('phase_type', 'unknown'))
    
    type_distribution = {}
    for phase_type in set(phase_types):
        type_distribution[phase_type] = phase_types.count(phase_type)
    
    return {
        'total_phases': total_phases,
        'avg_phases_per_track': avg_phases_per_track,
        'total_files': total_files,
        'min_phases_per_track': min(phase_counts) if phase_counts else 0,
        'max_phases_per_track': max(phase_counts) if phase_counts else 0,
        'phase_type_distribution': type_distribution,
        **duration_stats
    }


def calculate_collection_summary(df: pd.DataFrame, phase_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for an audio collection.
    
    This function provides high-level metrics about the entire analyzed collection,
    combining track-level and phase-level data into a cohesive summary.
    
    Args:
        df: DataFrame containing track features
        phase_data: List of phase analysis results
        
    Returns:
        Dictionary containing collection summary statistics
    """
    if df.empty:
        return {'error': 'No data available for summary'}
    
    # Basic collection metrics
    summary = {
        'total_files': len(df),
        'total_duration_seconds': float(df['duration'].sum()) if 'duration' in df.columns else 0,
        'total_duration_minutes': float(df['duration'].sum()) / 60 if 'duration' in df.columns else 0,
        'avg_duration_seconds': float(df['duration'].mean()) if 'duration' in df.columns else 0,
        'avg_duration_minutes': float(df['duration'].mean()) / 60 if 'duration' in df.columns else 0
    }
    
    # Musical characteristics
    # Support both old 'detected_key' and new spec-compliant 'key' field name
    _key_col = 'key' if 'key' in df.columns else ('detected_key' if 'detected_key' in df.columns else None)
    if _key_col:
        summary['unique_keys'] = df[_key_col].nunique()
        summary['most_common_key'] = df[_key_col].mode().iloc[0] if not df[_key_col].mode().empty else 'Unknown'
    
    if 'tempo' in df.columns:
        summary['avg_tempo'] = float(df['tempo'].mean())
        summary['tempo_range'] = float(df['tempo'].max() - df['tempo'].min())
        summary['tempo_std'] = float(df['tempo'].std())
    
    # Energy characteristics
    if 'rms_mean' in df.columns:
        summary['avg_energy'] = float(df['rms_mean'].mean())
        summary['energy_range'] = float(df['rms_mean'].max() - df['rms_mean'].min())
        summary['energy_std'] = float(df['rms_mean'].std())
    
    # Spectral characteristics
    if 'spectral_centroid_mean' in df.columns:
        summary['avg_brightness'] = float(df['spectral_centroid_mean'].mean())
        summary['brightness_range'] = float(df['spectral_centroid_mean'].max() - df['spectral_centroid_mean'].min())
        summary['brightness_std'] = float(df['spectral_centroid_mean'].std())
    
    # Phase statistics
    phase_stats = calculate_phase_statistics(phase_data)
    summary.update(phase_stats)
    
    # Mood statistics
    if 'primary_mood' in df.columns:
        mood_counts = df['primary_mood'].value_counts()
        summary['unique_moods'] = len(mood_counts)
        summary['dominant_mood'] = mood_counts.index[0] if len(mood_counts) > 0 else 'Unknown'
        summary['mood_distribution'] = mood_counts.to_dict()
    
    # Character statistics
    if 'primary_character' in df.columns:
        character_counts = df['primary_character'].value_counts()
        summary['unique_characters'] = len(character_counts)
        summary['dominant_character'] = character_counts.index[0] if len(character_counts) > 0 else 'Unknown'
        summary['character_distribution'] = character_counts.to_dict()
    
    return summary


def calculate_cluster_statistics(df: pd.DataFrame, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for cluster analysis results.
    
    Args:
        df: DataFrame containing track features with cluster assignments
        cluster_analysis: Cluster analysis results dictionary
        
    Returns:
        Dictionary containing cluster statistics
    """
    if not cluster_analysis or df.empty:
        return {'error': 'No cluster data available'}
    
    stats = {
        'total_clusters': len(cluster_analysis),
        'cluster_sizes': {},
        'cluster_characteristics': {}
    }
    
    # Calculate cluster size distribution
    for cluster_name, analysis in cluster_analysis.items():
        cluster_id = cluster_name.split('_')[1] if '_' in cluster_name else cluster_name
        stats['cluster_sizes'][cluster_id] = analysis.get('count', 0)
        
        # Extract key characteristics for each cluster
        stats['cluster_characteristics'][cluster_id] = {
            'avg_duration': analysis.get('avg_duration', 0),
            'avg_tempo': analysis.get('avg_tempo', 0),
            'avg_energy': analysis.get('avg_energy', 0),
            'dominant_mood': analysis.get('dominant_mood', 'Unknown'),
            'dominant_character': analysis.get('dominant_character', 'Unknown')
        }
    
    # Calculate cluster balance (how evenly distributed tracks are)
    cluster_sizes = list(stats['cluster_sizes'].values())
    if cluster_sizes:
        stats['cluster_balance'] = min(cluster_sizes) / max(cluster_sizes) if max(cluster_sizes) > 0 else 0
        stats['avg_cluster_size'] = np.mean(cluster_sizes)
        stats['cluster_size_std'] = np.std(cluster_sizes)
    
    return stats


def calculate_feature_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for extracted features.
    
    Args:
        df: DataFrame containing audio features
        
    Returns:
        Dictionary containing feature statistics
    """
    if df.empty:
        return {'error': 'No feature data available'}
    
    # Select numeric columns for analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    stats = {
        'total_features': len(df.columns),
        'numeric_features': len(numeric_columns),
        'categorical_features': len(df.columns) - len(numeric_columns),
        'feature_statistics': {}
    }
    
    # Calculate statistics for each numeric feature
    for column in numeric_columns:
        feature_stats = {
            'mean': float(df[column].mean()),
            'std': float(df[column].std()),
            'min': float(df[column].min()),
            'max': float(df[column].max()),
            'median': float(df[column].median()),
            'range': float(df[column].max() - df[column].min()),
            'coefficient_of_variation': float(df[column].std() / df[column].mean()) if df[column].mean() != 0 else 0
        }
        stats['feature_statistics'][column] = feature_stats
    
    # Identify features with high variance (most informative)
    feature_variances = df[numeric_columns].var().sort_values(ascending=False)
    stats['high_variance_features'] = feature_variances.head(10).to_dict()
    
    # Identify features with low variance (potentially redundant)
    stats['low_variance_features'] = feature_variances.tail(5).to_dict()
    
    return stats


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a human-readable way.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "3:45", "1:23:45")
    """
    if seconds >= 3600:  # More than an hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def format_time_position(seconds: float) -> str:
    """
    Format time position in MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "03:45")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_progression_trend(values: List[float]) -> str:
    """
    Calculate the overall trend of a sequence of values.
    
    Args:
        values: List of numeric values in order
        
    Returns:
        String describing the trend ('increasing', 'decreasing', 'stable')
    """
    if len(values) < 2:
        return 'stable'
    
    # Calculate overall change
    start_avg = np.mean(values[:max(1, len(values)//4)])  # First quarter
    end_avg = np.mean(values[-max(1, len(values)//4):])   # Last quarter
    
    change_ratio = end_avg / start_avg if start_avg != 0 else 1
    
    if change_ratio > 1.2:
        return 'increasing'
    elif change_ratio < 0.8:
        return 'decreasing'
    else:
        return 'stable'


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def calculate_normalized_score(value: float, min_val: float, max_val: float) -> float:
    """
    Calculate a normalized score between 0 and 1.
    
    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
        
    Returns:
        Normalized score between 0 and 1
    """
    if max_val == min_val:
        return 0.5  # Middle value when no range
    return (value - min_val) / (max_val - min_val)