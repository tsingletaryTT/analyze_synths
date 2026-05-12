"""
CSV Export Module for Audio Analysis Results

This module provides comprehensive CSV export functionality for audio analysis
results. CSV format is ideal for spreadsheet analysis, data visualization tools,
and integration with other data analysis workflows.

The CSV exporter creates multiple specialized files:
- Audio features: Complete feature matrix for each track
- Phase analysis: Detailed phase-by-phase breakdown
- Cluster analysis: Cluster characteristics and membership
- Sequence recommendations: Optimal track ordering
- Summary statistics: High-level collection metrics

All exports are optimized for LLM consumption and human readability,
with proper data types, meaningful column names, and consistent formatting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class CSVExporter:
    """
    Comprehensive CSV export system for audio analysis results.
    
    This class handles the export of all analysis results to CSV format,
    ensuring data is properly formatted, cleaned, and organized for
    downstream analysis and visualization.
    """
    
    def __init__(self):
        """Initialize the CSV exporter with default settings."""
        pass
    
    def export_features(self, df: pd.DataFrame, output_path: Path) -> bool:
        """
        Export the main audio features DataFrame to CSV.
        
        This method exports the comprehensive feature matrix containing
        all extracted audio characteristics. The CSV is optimized for
        both human readability and machine processing.
        
        Args:
            df: DataFrame containing audio features
            output_path: Path where CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Prepare DataFrame for export
            export_df = self._prepare_dataframe_for_export(df)
            
            # Export to CSV with proper formatting
            export_df.to_csv(output_path, index=False, float_format='%.4f')
            
            print(f"Features exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting features to CSV: {str(e)}")
            return False
    
    def export_phases(self, phase_data: List[Dict[str, Any]], output_path: Path) -> bool:
        """
        Export phase analysis results to CSV format.
        
        This method flattens the hierarchical phase data into a tabular
        format suitable for spreadsheet analysis. Each row represents
        a single phase with all its characteristics.
        
        Args:
            phase_data: List of phase analysis results
            output_path: Path where CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Flatten phase data into rows
            phase_rows = []
            
            for file_info in phase_data:
                filename = file_info['filename']
                total_duration = file_info['total_duration']
                num_phases = file_info['num_phases']
                
                for phase in file_info['phases']:
                    # Create row for this phase
                    phase_row = {
                        'filename': filename,
                        'total_duration_seconds': total_duration,
                        'total_duration_minutes': total_duration / 60,
                        'total_phases': num_phases,
                        'phase_number': phase['phase_number'],
                        'phase_type': phase['phase_type'],
                        'start_time_seconds': phase['start_time'],
                        'end_time_seconds': phase['end_time'],
                        'duration_seconds': phase['duration'],
                        'duration_minutes': phase['duration'] / 60,
                        'avg_energy': phase['avg_energy'],
                        'avg_brightness_hz': phase['avg_brightness'],
                        'avg_roughness': phase['avg_roughness'],
                        'onset_density_per_second': phase['onset_density'],
                        'mood_descriptors': ', '.join(phase.get('mood_descriptors', [])),
                        'mood_count': len(phase.get('mood_descriptors', [])),
                        'start_time_formatted': format_time_position(phase['start_time']),
                        'end_time_formatted': format_time_position(phase['end_time']),
                        'duration_formatted': format_time_duration(phase['duration'])
                    }
                    
                    phase_rows.append(phase_row)
            
            if not phase_rows:
                print("No phase data to export")
                return False
            
            # Create DataFrame and export
            phase_df = pd.DataFrame(phase_rows)
            phase_df = self._prepare_dataframe_for_export(phase_df)
            phase_df.to_csv(output_path, index=False, float_format='%.4f')
            
            print(f"Phase analysis exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting phase analysis to CSV: {str(e)}")
            return False
    
    def export_clusters(self, cluster_analysis: Dict[str, Any], output_path: Path) -> bool:
        """
        Export cluster analysis results to CSV format.
        
        This method converts the cluster analysis dictionary into a
        tabular format with one row per cluster and columns for all
        cluster characteristics.
        
        Args:
            cluster_analysis: Cluster analysis results
            output_path: Path where CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if not cluster_analysis:
                print("No cluster analysis data to export")
                return False
            
            # Convert cluster analysis to rows
            cluster_rows = []
            
            for cluster_name, analysis in cluster_analysis.items():
                cluster_row = {
                    'cluster_name': cluster_name,
                    'cluster_id': cluster_name.split('_')[1] if '_' in cluster_name else cluster_name,
                    'track_count': analysis.get('count', 0),
                    'track_files': '; '.join(analysis.get('files', [])),
                    'avg_tempo_bpm': analysis.get('avg_tempo', 0),
                    'tempo_std': analysis.get('tempo_std', 0),
                    'avg_duration_seconds': analysis.get('avg_duration', 0),
                    'avg_duration_minutes': analysis.get('avg_duration', 0) / 60,
                    'duration_std_seconds': analysis.get('duration_std', 0),
                    'avg_energy': analysis.get('avg_energy', 0),
                    'energy_std': analysis.get('energy_std', 0),
                    'avg_brightness_hz': analysis.get('avg_brightness', 0),
                    'brightness_std': analysis.get('brightness_std', 0),
                    'common_key': analysis.get('common_key', 'Unknown'),
                    'key_diversity': analysis.get('key_diversity', 0),
                    'avg_phases': analysis.get('avg_phases', 0),
                    'climax_percentage': analysis.get('has_climax_percent', 0),
                    'breakdown_percentage': analysis.get('has_breakdown_percent', 0),
                    'dominant_mood': analysis.get('dominant_mood', 'Unknown'),
                    'mood_diversity': analysis.get('mood_diversity', 0),
                    'dominant_character': analysis.get('dominant_character', 'Unknown'),
                    'character_diversity': analysis.get('character_diversity', 0),
                    'homogeneity_score': analysis.get('homogeneity', 0),
                    'musical_coherence': analysis.get('musical_coherence', 'Unknown')
                }
                
                cluster_rows.append(cluster_row)
            
            # Create DataFrame and export
            cluster_df = pd.DataFrame(cluster_rows)
            cluster_df = self._prepare_dataframe_for_export(cluster_df)
            cluster_df.to_csv(output_path, index=False, float_format='%.4f')
            
            print(f"Cluster analysis exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting cluster analysis to CSV: {str(e)}")
            return False
    
    def export_sequence(self, sequence_recommendations: List[Dict[str, Any]], 
                       output_path: Path) -> bool:
        """
        Export sequence recommendations to CSV format.
        
        This method exports the recommended track ordering with all
        the reasoning and characteristics for each position.
        
        Args:
            sequence_recommendations: List of sequence recommendations
            output_path: Path where CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if not sequence_recommendations:
                print("No sequence recommendations to export")
                return False
            
            # Convert sequence recommendations to DataFrame-friendly format
            sequence_rows = []
            
            for rec in sequence_recommendations:
                sequence_row = {
                    'position': rec['position'],
                    'filename': rec['filename'],
                    'duration_seconds': rec['duration'],
                    'duration_minutes': rec['duration'] / 60,
                    'duration_formatted': format_time_duration(rec['duration']),
                    'tempo_bpm': rec['tempo'],
                    'key': rec['key'],
                    'primary_mood': rec['mood'],
                    'primary_character': rec['character'],
                    'energy_level': rec['energy'],
                    'reasoning': rec['reasoning'],
                    'position_percentage': (rec['position'] - 1) / (len(sequence_recommendations) - 1) * 100,
                    'cumulative_duration_seconds': sum(r['duration'] for r in sequence_recommendations[:rec['position']]),
                    'cumulative_duration_minutes': sum(r['duration'] for r in sequence_recommendations[:rec['position']]) / 60
                }
                
                sequence_rows.append(sequence_row)
            
            # Create DataFrame and export
            sequence_df = pd.DataFrame(sequence_rows)
            sequence_df = self._prepare_dataframe_for_export(sequence_df)
            sequence_df.to_csv(output_path, index=False, float_format='%.4f')
            
            print(f"Sequence recommendations exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting sequence recommendations to CSV: {str(e)}")
            return False
    
    def export_summary_stats(self, df: pd.DataFrame, phase_data: List[Dict[str, Any]], 
                           output_path: Path) -> bool:
        """
        Export summary statistics to CSV format.
        
        This method creates a high-level summary of the entire collection,
        providing key metrics and insights in a single CSV file.
        
        Args:
            df: DataFrame with track features
            phase_data: Phase analysis data
            output_path: Path where CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Calculate comprehensive summary statistics using shared utility
            summary_stats = calculate_collection_summary(df, phase_data)
            
            # Convert to DataFrame for CSV export
            # Create rows for different categories of statistics
            summary_rows = []
            
            # Collection overview
            summary_rows.append({
                'category': 'Collection Overview',
                'metric': 'Total Tracks',
                'value': summary_stats['total_files'],
                'unit': 'tracks',
                'description': 'Number of audio files analyzed'
            })
            
            summary_rows.append({
                'category': 'Collection Overview',
                'metric': 'Total Duration',
                'value': summary_stats['total_duration_minutes'],
                'unit': 'minutes',
                'description': 'Total playback time of all tracks'
            })
            
            summary_rows.append({
                'category': 'Collection Overview',
                'metric': 'Average Track Duration',
                'value': summary_stats['avg_duration_minutes'],
                'unit': 'minutes',
                'description': 'Average length per track'
            })
            
            # Musical characteristics
            summary_rows.append({
                'category': 'Musical Characteristics',
                'metric': 'Unique Keys',
                'value': summary_stats['unique_keys'],
                'unit': 'keys',
                'description': 'Number of different musical keys detected'
            })
            
            summary_rows.append({
                'category': 'Musical Characteristics',
                'metric': 'Average Tempo',
                'value': summary_stats['avg_tempo'],
                'unit': 'BPM',
                'description': 'Average tempo across all tracks'
            })
            
            summary_rows.append({
                'category': 'Musical Characteristics',
                'metric': 'Average Energy',
                'value': summary_stats['avg_energy'],
                'unit': 'level',
                'description': 'Average energy level (0-1 scale)'
            })
            
            summary_rows.append({
                'category': 'Musical Characteristics',
                'metric': 'Average Brightness',
                'value': summary_stats['avg_brightness'],
                'unit': 'Hz',
                'description': 'Average spectral brightness'
            })
            
            # Structural analysis
            summary_rows.append({
                'category': 'Structural Analysis',
                'metric': 'Total Phases',
                'value': summary_stats['total_phases'],
                'unit': 'phases',
                'description': 'Total number of musical phases detected'
            })
            
            summary_rows.append({
                'category': 'Structural Analysis',
                'metric': 'Average Phases per Track',
                'value': summary_stats['avg_phases_per_track'],
                'unit': 'phases',
                'description': 'Average structural complexity'
            })
            
            # Add mood and character statistics if available
            if 'primary_mood' in df.columns:
                mood_counts = df['primary_mood'].value_counts()
                summary_rows.append({
                    'category': 'Creative Analysis',
                    'metric': 'Unique Moods',
                    'value': len(mood_counts),
                    'unit': 'moods',
                    'description': 'Number of different moods detected'
                })
                
                summary_rows.append({
                    'category': 'Creative Analysis',
                    'metric': 'Dominant Mood',
                    'value': mood_counts.index[0] if len(mood_counts) > 0 else 'Unknown',
                    'unit': 'mood',
                    'description': 'Most common mood in the collection'
                })
            
            if 'primary_character' in df.columns:
                character_counts = df['primary_character'].value_counts()
                summary_rows.append({
                    'category': 'Creative Analysis',
                    'metric': 'Dominant Character',
                    'value': character_counts.index[0] if len(character_counts) > 0 else 'Unknown',
                    'unit': 'character',
                    'description': 'Most common character type'
                })
            
            # Create DataFrame and export
            summary_df = pd.DataFrame(summary_rows)
            summary_df = self._prepare_dataframe_for_export(summary_df)
            summary_df.to_csv(output_path, index=False)
            
            print(f"Summary statistics exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting summary statistics to CSV: {str(e)}")
            return False
    
    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for CSV export with proper formatting.
        
        This method ensures the DataFrame is optimally formatted for
        CSV export, with proper data types and formatting.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Prepared DataFrame
        """
        export_df = df.copy()
        
        # Convert numpy types to Python types for better CSV compatibility
        for column in export_df.columns:
            if export_df[column].dtype in ['int64', 'int32']:
                export_df[column] = export_df[column].astype(int)
            elif export_df[column].dtype in ['float64', 'float32']:
                # Round to appropriate precision based on column type
                if 'mfcc' in column.lower():
                    export_df[column] = export_df[column].round(4)
                elif 'spectral' in column.lower():
                    export_df[column] = export_df[column].round(2)
                elif 'tempo' in column.lower():
                    export_df[column] = export_df[column].round(1)
                elif 'duration' in column.lower():
                    export_df[column] = export_df[column].round(2)
                elif 'energy' in column.lower():
                    export_df[column] = export_df[column].round(4)
                else:
                    export_df[column] = export_df[column].round(3)
        
        # Ensure string columns are properly formatted
        string_columns = export_df.select_dtypes(include=['object']).columns
        for column in string_columns:
            export_df[column] = export_df[column].astype(str)
            # Clean up common string issues
            export_df[column] = export_df[column].str.replace('nan', 'Unknown')
            export_df[column] = export_df[column].str.replace('None', 'Unknown')
        
        return export_df
