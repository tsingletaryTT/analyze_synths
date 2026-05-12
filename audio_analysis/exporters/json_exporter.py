"""
JSON Export Module for Audio Analysis Results

This module provides comprehensive JSON export functionality for audio analysis
results. JSON format is ideal for programmatic access, API integration, and
data interchange between different systems.

The JSON exporter creates structured, hierarchical data that preserves the
relationships between different analysis components while ensuring compatibility
with various JSON processing tools and libraries.

Key features:
- Hierarchical data structure preservation
- Proper data type handling (avoiding numpy/pandas types)
- Comprehensive metadata inclusion
- Backwards compatibility with existing formats
- Optimized for both human readability and machine processing
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class JSONExporter:
    """
    Comprehensive JSON export system for audio analysis results.
    
    This class handles the export of all analysis results to JSON format,
    ensuring proper data serialization, type conversion, and structure
    preservation for programmatic access.
    """
    
    def __init__(self):
        """Initialize the JSON exporter with default settings."""
        pass
    
    def export_comprehensive_data(self, df: pd.DataFrame, phase_data: List[Dict[str, Any]],
                                cluster_analysis: Optional[Dict[str, Any]] = None,
                                sequence_recommendations: Optional[List[Dict[str, Any]]] = None,
                                output_path: Path = None) -> Dict[str, Any]:
        """
        Export comprehensive analysis data to JSON format.
        
        This method creates a complete JSON export containing all analysis
        results in a structured, hierarchical format. The export includes
        metadata, versioning, and timestamps for full traceability.
        
        Args:
            df: DataFrame containing audio features
            phase_data: List of phase analysis results
            cluster_analysis: Cluster analysis results (optional)
            sequence_recommendations: Sequence recommendations (optional)
            output_path: Path to save JSON file (optional)
            
        Returns:
            Dictionary containing all analysis data
        """
        try:
            # Create comprehensive data structure
            comprehensive_data = {
                'metadata': self._create_metadata(),
                'collection_summary': self._create_collection_summary(df, phase_data),
                'tracks': self._convert_tracks_data(df),
                'phase_analysis': self._convert_phase_data(phase_data),
                'cluster_analysis': self._convert_cluster_data(cluster_analysis) if cluster_analysis else None,
                'sequence_recommendations': self._convert_sequence_data(sequence_recommendations) if sequence_recommendations else None,
                'analysis_statistics': self._calculate_analysis_statistics(df, phase_data, cluster_analysis, sequence_recommendations)
            }
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(comprehensive_data, f, indent=2, default=self._json_serializer)
                print(f"Comprehensive JSON data exported to: {output_path}")
            
            return comprehensive_data
            
        except Exception as e:
            print(f"Error exporting comprehensive data to JSON: {str(e)}")
            return {}
    
    def export_tracks_only(self, df: pd.DataFrame, output_path: Path) -> bool:
        """
        Export only track-level data to JSON format.
        
        This method creates a focused JSON export containing just the
        track-level analysis results, useful for lightweight integrations.
        
        Args:
            df: DataFrame containing audio features
            output_path: Path to save JSON file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            tracks_data = {
                'metadata': self._create_metadata(),
                'tracks': self._convert_tracks_data(df)
            }
            
            with open(output_path, 'w') as f:
                json.dump(tracks_data, f, indent=2, default=self._json_serializer)
            
            print(f"Track data exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting track data to JSON: {str(e)}")
            return False
    
    def export_phases_only(self, phase_data: List[Dict[str, Any]], output_path: Path) -> bool:
        """
        Export only phase analysis data to JSON format.
        
        Args:
            phase_data: List of phase analysis results
            output_path: Path to save JSON file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            phases_data = {
                'metadata': self._create_metadata(),
                'phase_analysis': self._convert_phase_data(phase_data)
            }
            
            with open(output_path, 'w') as f:
                json.dump(phases_data, f, indent=2, default=self._json_serializer)
            
            print(f"Phase analysis data exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting phase data to JSON: {str(e)}")
            return False
    
    def _create_metadata(self) -> Dict[str, Any]:
        """
        Create metadata for the JSON export.
        
        Returns:
            Dictionary containing export metadata
        """
        return {
            'export_timestamp': datetime.now().isoformat(),
            'export_format': 'json',
            'schema_version': '1.0.0',
            'generator': 'AudioAnalyzer',
            'generator_version': '1.0.0',
            'description': 'Comprehensive audio analysis results for synthesizer music',
            'data_types': {
                'features': 'Audio feature vectors with 80+ dimensions',
                'phases': 'Musical structure analysis with mood descriptors',
                'clusters': 'Track grouping based on similarity',
                'sequences': 'Optimal track ordering recommendations'
            }
        }
    
    def _create_collection_summary(self, df: pd.DataFrame, 
                                 phase_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create high-level summary of the collection.
        
        Args:
            df: DataFrame with track features
            phase_data: Phase analysis data
            
        Returns:
            Dictionary with collection summary
        """
        summary = {
            'total_tracks': len(df),
            'total_duration_seconds': float(df['duration'].sum()),
            'total_duration_minutes': float(df['duration'].sum()) / 60,
            'average_track_duration': float(df['duration'].mean()),
            'total_phases': sum(len(f['phases']) for f in phase_data),
            'average_phases_per_track': np.mean([len(f['phases']) for f in phase_data]) if phase_data else 0
        }
        
        # Add musical characteristics if available
        if 'tempo' in df.columns:
            summary['tempo_range'] = {
                'min': float(df['tempo'].min()),
                'max': float(df['tempo'].max()),
                'mean': float(df['tempo'].mean()),
                'std': float(df['tempo'].std())
            }
        
        # Support both 'key' (spec-compliant) and legacy 'detected_key' field name
        _key_col = 'key' if 'key' in df.columns else ('detected_key' if 'detected_key' in df.columns else None)
        if _key_col:
            key_counts = df[_key_col].value_counts()
            summary['key_distribution'] = {
                'unique_keys': len(key_counts),
                'most_common_key': key_counts.index[0],
                'key_diversity': len(key_counts) / len(df)
            }
        
        if 'primary_mood' in df.columns:
            mood_counts = df['primary_mood'].value_counts()
            summary['mood_distribution'] = {
                'unique_moods': len(mood_counts),
                'dominant_mood': mood_counts.index[0] if len(mood_counts) > 0 else 'Unknown',
                'mood_diversity': len(mood_counts) / len(df)
            }
        
        return summary
    
    def _convert_tracks_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to JSON-serializable track data.
        
        Args:
            df: DataFrame containing track features
            
        Returns:
            List of track dictionaries
        """
        tracks = []
        
        for _, row in df.iterrows():
            track = {}
            
            # Convert each column value to JSON-serializable format
            for column, value in row.items():
                track[column] = self._convert_value(value)
            
            tracks.append(track)
        
        return tracks
    
    def _convert_phase_data(self, phase_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert phase data to JSON-serializable format.
        
        Args:
            phase_data: List of phase analysis results
            
        Returns:
            List of phase analysis dictionaries
        """
        converted_phases = []
        
        for file_info in phase_data:
            file_phases = {
                'filename': file_info['filename'],
                'total_duration': float(file_info['total_duration']),
                'num_phases': int(file_info['num_phases']),
                'phases': []
            }
            
            for phase in file_info['phases']:
                converted_phase = {
                    'phase_number': int(phase['phase_number']),
                    'start_time': float(phase['start_time']),
                    'end_time': float(phase['end_time']),
                    'duration': float(phase['duration']),
                    'phase_type': str(phase['phase_type']),
                    'characteristics': {
                        'avg_energy': float(phase['avg_energy']),
                        'avg_brightness': float(phase['avg_brightness']),
                        'avg_roughness': float(phase['avg_roughness']),
                        'onset_density': float(phase['onset_density'])
                    },
                    'mood_descriptors': phase.get('mood_descriptors', []),
                    'mood_confidence': {
                        mood: float(confidence) 
                        for mood, confidence in phase.get('mood_confidence', {}).items()
                    } if phase.get('mood_confidence') else {}
                }
                
                file_phases['phases'].append(converted_phase)
            
            converted_phases.append(file_phases)
        
        return converted_phases
    
    def _convert_cluster_data(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert cluster analysis to JSON-serializable format.
        
        Args:
            cluster_analysis: Cluster analysis results
            
        Returns:
            Dictionary with converted cluster data
        """
        converted_clusters = {}
        
        for cluster_name, analysis in cluster_analysis.items():
            converted_clusters[cluster_name] = {
                'track_count': int(analysis.get('count', 0)),
                'track_files': analysis.get('files', []),
                'characteristics': {
                    'avg_tempo': float(analysis.get('avg_tempo', 0)),
                    'tempo_std': float(analysis.get('tempo_std', 0)),
                    'avg_duration': float(analysis.get('avg_duration', 0)),
                    'duration_std': float(analysis.get('duration_std', 0)),
                    'avg_energy': float(analysis.get('avg_energy', 0)),
                    'energy_std': float(analysis.get('energy_std', 0)),
                    'avg_brightness': float(analysis.get('avg_brightness', 0)),
                    'brightness_std': float(analysis.get('brightness_std', 0))
                },
                'musical_properties': {
                    'common_key': str(analysis.get('common_key', 'Unknown')),
                    'key_diversity': int(analysis.get('key_diversity', 0)),
                    'avg_phases': float(analysis.get('avg_phases', 0)),
                    'climax_percentage': float(analysis.get('has_climax_percent', 0)),
                    'breakdown_percentage': float(analysis.get('has_breakdown_percent', 0))
                },
                'creative_properties': {
                    'dominant_mood': str(analysis.get('dominant_mood', 'Unknown')),
                    'mood_diversity': int(analysis.get('mood_diversity', 0)),
                    'dominant_character': str(analysis.get('dominant_character', 'Unknown')),
                    'character_diversity': int(analysis.get('character_diversity', 0))
                },
                'quality_metrics': {
                    'homogeneity': float(analysis.get('homogeneity', 0)),
                    'musical_coherence': str(analysis.get('musical_coherence', 'Unknown'))
                }
            }
        
        return converted_clusters
    
    def _convert_sequence_data(self, sequence_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert sequence recommendations to JSON-serializable format.
        
        Args:
            sequence_recommendations: List of sequence recommendations
            
        Returns:
            Dictionary with converted sequence data
        """
        return {
            'total_tracks': len(sequence_recommendations),
            'total_duration': sum(track['duration'] for track in sequence_recommendations),
            'sequence': [
                {
                    'position': int(track['position']),
                    'filename': str(track['filename']),
                    'duration': float(track['duration']),
                    'musical_properties': {
                        'tempo': float(track['tempo']),
                        'key': str(track['key']),
                        'mood': str(track['mood']),
                        'character': str(track['character']),
                        'energy': float(track['energy'])
                    },
                    'placement_reasoning': str(track['reasoning'])
                }
                for track in sequence_recommendations
            ]
        }
    
    def _calculate_analysis_statistics(self, df: pd.DataFrame, phase_data: List[Dict[str, Any]],
                                     cluster_analysis: Optional[Dict[str, Any]],
                                     sequence_recommendations: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate comprehensive analysis statistics.
        
        Args:
            df: DataFrame with track features
            phase_data: Phase analysis data
            cluster_analysis: Cluster analysis results
            sequence_recommendations: Sequence recommendations
            
        Returns:
            Dictionary with analysis statistics
        """
        stats = {
            'feature_extraction': {
                'total_features_per_track': len(df.columns),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns)
            },
            'phase_analysis': {
                'total_phases': sum(len(f['phases']) for f in phase_data),
                'avg_phases_per_track': float(np.mean([len(f['phases']) for f in phase_data])) if phase_data else 0,
                'phase_types_detected': len(set(
                    phase['phase_type'] 
                    for file_info in phase_data 
                    for phase in file_info['phases']
                ))
            }
        }
        
        if cluster_analysis:
            stats['clustering'] = {
                'num_clusters': len(cluster_analysis),
                'largest_cluster_size': max(analysis.get('count', 0) for analysis in cluster_analysis.values()),
                'smallest_cluster_size': min(analysis.get('count', 0) for analysis in cluster_analysis.values())
            }
        
        if sequence_recommendations:
            stats['sequencing'] = {
                'total_sequence_duration': sum(track['duration'] for track in sequence_recommendations),
                'tempo_range_in_sequence': {
                    'min': min(track['tempo'] for track in sequence_recommendations),
                    'max': max(track['tempo'] for track in sequence_recommendations)
                },
                'energy_progression': {
                    'start': sequence_recommendations[0]['energy'],
                    'end': sequence_recommendations[-1]['energy']
                }
            }
        
        return stats
    
    def _convert_value(self, value: Any) -> Any:
        """
        Convert a value to JSON-serializable format.
        
        This method handles the conversion of numpy and pandas types
        to standard Python types for JSON serialization.
        
        Args:
            value: Value to convert
            
        Returns:
            JSON-serializable value
        """
        # Handle numpy types
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.bool_, bool)):
            return bool(value)
        
        # Handle pandas types
        elif hasattr(value, 'item'):  # pandas scalar
            return value.item()
        
        # Handle None and NaN
        elif pd.isna(value):
            return None
        
        # Return as-is for standard Python types
        else:
            return value
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for handling special types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        
        # Handle pandas types
        elif hasattr(obj, 'item'):  # pandas scalar
            return obj.item()
        
        # Handle Path objects
        elif isinstance(obj, Path):
            return str(obj)
        
        # Fallback for other types
        else:
            return str(obj)