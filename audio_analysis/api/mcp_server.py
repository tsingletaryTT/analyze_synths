"""
FastMCP Server Integration for Audio Analysis

This module provides MCP (Model Context Protocol) server integration for the
audio analysis toolkit, allowing remote access to analysis capabilities
through a standardized protocol interface.

The MCP server exposes six main tools:
1. analyze_audio_mood - Creative mood and character analysis
2. analyze_audio_phases - Musical phase detection and analysis
3. recommend_song_sequence - Intelligent track sequencing
4. analyze_audio_clusters - Similarity-based track grouping
5. comprehensive_audio_analysis - Complete analysis with export options
6. get_supported_formats - Format and capability information

All tools support base64-encoded audio file transmission and provide
structured JSON responses optimized for programmatic access.
"""

import asyncio
import json
import tempfile
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: FastMCP not available. Install with: pip install fastmcp")

from .analyzer import AudioAnalyzer


class MCPAudioAnalyzer:
    """
    MCP server wrapper for the AudioAnalyzer toolkit.
    
    This class provides a bridge between the FastMCP server protocol
    and the audio analysis toolkit, handling file management, validation,
    and response formatting for remote analysis requests.
    
    Key Features:
    - Base64 audio file encoding/decoding
    - Temporary file session management
    - Input validation and error handling
    - Structured JSON response formatting
    - Session isolation for concurrent requests
    """
    
    def __init__(self):
        """Initialize the MCP audio analyzer."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analyzer = None
        self.supported_formats = ['.wav', '.aiff', '.aif', '.mp3']
        
    def setup_analyzer(self, audio_files: List[Dict[str, Any]]) -> str:
        """
        Set up analyzer with audio files from MCP request.
        
        This method creates a temporary analysis session by:
        1. Creating an isolated temporary directory
        2. Decoding base64 audio file content
        3. Writing audio files to the temporary directory
        4. Initializing the AudioAnalyzer for the session
        
        Args:
            audio_files: List of audio file dictionaries with 'filename' and 'content'
            
        Returns:
            Path to the session directory
        """
        # Create unique session directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_dir = self.temp_dir / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)
        
        # Decode and write audio files
        for file_data in audio_files:
            filename = file_data['filename']
            content = base64.b64decode(file_data['content'])
            
            file_path = session_dir / filename
            with open(file_path, 'wb') as f:
                f.write(content)
        
        # Initialize analyzer for this session
        self.analyzer = AudioAnalyzer(session_dir)
        
        return str(session_dir)
    
    def validate_audio_files(self, audio_files: List[Dict[str, Any]]) -> List[str]:
        """
        Validate audio files for format support and completeness.
        
        This method performs comprehensive validation:
        1. Checks file format against supported extensions
        2. Validates presence of required fields
        3. Estimates file size and content validity
        4. Returns detailed error messages for issues
        
        Args:
            audio_files: List of audio file dictionaries
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
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
            if not any(filename.lower().endswith(ext) for ext in self.supported_formats):
                errors.append(f"{file_prefix} ({filename}): Unsupported format. Supported: {', '.join(self.supported_formats)}")
            
            # Check base64 content validity
            try:
                content = file_data['content']
                decoded_content = base64.b64decode(content)
                
                # Check for reasonable file size (basic validation)
                if len(decoded_content) < 1000:  # Less than 1KB
                    errors.append(f"{file_prefix} ({filename}): File appears too small to be valid audio")
                elif len(decoded_content) > 100 * 1024 * 1024:  # More than 100MB
                    errors.append(f"{file_prefix} ({filename}): File too large (>100MB)")
                    
            except Exception as e:
                errors.append(f"{file_prefix} ({filename}): Invalid base64 content - {str(e)}")
        
        return errors


# Initialize FastMCP server if available
if MCP_AVAILABLE:
    mcp = FastMCP("Synthesizer Music Analysis")
    analyzer_instance = MCPAudioAnalyzer()


    @mcp.tool()
    def analyze_audio_mood(audio_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the mood and character of audio files using creative descriptors.
        
        This tool provides comprehensive mood analysis using 17 creative descriptors
        specifically designed for synthesizer music. It identifies emotional and
        atmospheric characteristics along with synthesis type classification.
        
        Args:
            audio_files: List of audio files with 'filename' and 'content' (base64 encoded)
        
        Returns:
            Dict with mood analysis including creative descriptors and confidence scores
        """
        # Validate input files
        errors = analyzer_instance.validate_audio_files(audio_files)
        if errors:
            return {
                "success": False,
                "error": "Validation failed",
                "details": errors
            }
        
        try:
            # Set up analysis session
            session_dir = analyzer_instance.setup_analyzer(audio_files)
            
            # Run comprehensive analysis
            df = analyzer_instance.analyzer.analyze_directory()
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Failed to analyze audio files",
                    "details": ["No features could be extracted from the provided files"]
                }
            
            # Extract mood and character data
            results = []
            for _, row in df.iterrows():
                track_result = {
                    "filename": str(row['filename']),
                    "analysis": {
                        "primary_mood": str(row.get('primary_mood', 'unknown')),
                        "mood_descriptors": str(row.get('mood_descriptors', '')),
                        "mood_confidence": float(row.get('mood_confidence', 0)),
                        "primary_character": str(row.get('primary_character', 'unknown')),
                        "character_tags": str(row.get('character_tags', '')),
                        "character_confidence": float(row.get('character_confidence', 0))
                    },
                    "technical_properties": {
                        "duration_seconds": float(row.get('duration', 0)),
                        "duration_minutes": float(row.get('duration', 0)) / 60,
                        "tempo_bpm": float(row.get('tempo', 0)),
                        "key": str(row.get('key', row.get('detected_key', 'unknown'))),
                        "energy_level": float(row.get('rms_mean', 0)),
                        "spectral_brightness_hz": float(row.get('spectral_centroid_mean', 0))
                    }
                }
                results.append(track_result)
            
            return {
                "success": True,
                "analysis_type": "mood_and_character",
                "total_tracks": len(results),
                "session_directory": session_dir,
                "tracks": results,
                "summary": {
                    "unique_moods": len(set(track['analysis']['primary_mood'] for track in results)),
                    "unique_characters": len(set(track['analysis']['primary_character'] for track in results)),
                    "avg_confidence": np.mean([track['analysis']['mood_confidence'] for track in results]),
                    "total_duration_minutes": sum(track['technical_properties']['duration_minutes'] for track in results)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "details": ["Unexpected error during mood analysis"]
            }


    @mcp.tool()
    def analyze_audio_phases(audio_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze musical phases/sections in audio files with detailed characteristics.
        
        This tool detects musical phases (intro, build-up, climax, breakdown, etc.)
        in synthesizer music and provides detailed analysis of each section including
        mood descriptors, energy characteristics, and structural information.
        
        Args:
            audio_files: List of audio files with 'filename' and 'content' (base64 encoded)
        
        Returns:
            Dict with phase analysis including timing, types, and characteristics
        """
        # Validate input files
        errors = analyzer_instance.validate_audio_files(audio_files)
        if errors:
            return {
                "success": False,
                "error": "Validation failed",
                "details": errors
            }
        
        try:
            # Set up analysis session
            session_dir = analyzer_instance.setup_analyzer(audio_files)
            
            # Run analysis to get phase data
            df = analyzer_instance.analyzer.analyze_directory()
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Failed to analyze audio files",
                    "details": ["No features could be extracted from the provided files"]
                }
            
            # Extract phase data
            results = []
            for file_info in analyzer_instance.analyzer.phase_data:
                phases = []
                for phase in file_info['phases']:
                    phase_result = {
                        "phase_number": int(phase['phase_number']),
                        "timing": {
                            "start_time_seconds": float(phase['start_time']),
                            "end_time_seconds": float(phase['end_time']),
                            "duration_seconds": float(phase['duration']),
                            "start_time_formatted": f"{int(phase['start_time']//60):02d}:{int(phase['start_time']%60):02d}",
                            "end_time_formatted": f"{int(phase['end_time']//60):02d}:{int(phase['end_time']%60):02d}",
                            "duration_formatted": f"{int(phase['duration']//60):02d}:{int(phase['duration']%60):02d}"
                        },
                        "classification": {
                            "phase_type": str(phase['phase_type']),
                            "mood_descriptors": phase.get('mood_descriptors', []),
                            "structural_role": _classify_structural_role(phase['phase_type'])
                        },
                        "characteristics": {
                            "avg_energy": float(phase['avg_energy']),
                            "avg_brightness_hz": float(phase['avg_brightness']),
                            "avg_roughness": float(phase['avg_roughness']),
                            "onset_density_per_second": float(phase['onset_density']),
                            "energy_level": _categorize_energy(phase['avg_energy']),
                            "brightness_level": _categorize_brightness(phase['avg_brightness'])
                        }
                    }
                    phases.append(phase_result)
                
                file_result = {
                    "filename": str(file_info['filename']),
                    "overall_structure": {
                        "total_duration_seconds": float(file_info['total_duration']),
                        "total_duration_formatted": f"{int(file_info['total_duration']//60):02d}:{int(file_info['total_duration']%60):02d}",
                        "num_phases": int(file_info['num_phases']),
                        "structural_complexity": _assess_complexity(file_info['num_phases'], file_info['total_duration'])
                    },
                    "phases": phases
                }
                results.append(file_result)
            
            return {
                "success": True,
                "analysis_type": "musical_phases",
                "total_files": len(results),
                "session_directory": session_dir,
                "files": results,
                "summary": {
                    "total_phases": sum(file['overall_structure']['num_phases'] for file in results),
                    "avg_phases_per_track": np.mean([file['overall_structure']['num_phases'] for file in results]),
                    "unique_phase_types": len(set(
                        phase['classification']['phase_type'] 
                        for file in results 
                        for phase in file['phases']
                    )),
                    "total_duration_minutes": sum(file['overall_structure']['total_duration_seconds'] for file in results) / 60
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Phase analysis failed: {str(e)}",
                "details": ["Unexpected error during phase analysis"]
            }


    @mcp.tool()
    def recommend_song_sequence(audio_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an optimal listening sequence for a collection of audio files.
        
        This tool uses advanced algorithms to create the best possible listening
        order based on musical flow principles including key relationships,
        tempo progression, energy arcs, and mood compatibility.
        
        Args:
            audio_files: List of audio files with 'filename' and 'content' (base64 encoded)
        
        Returns:
            Dict with recommended sequence and detailed reasoning for each position
        """
        # Validate input files
        errors = analyzer_instance.validate_audio_files(audio_files)
        if errors:
            return {
                "success": False,
                "error": "Validation failed", 
                "details": errors
            }
        
        try:
            # Set up analysis session
            session_dir = analyzer_instance.setup_analyzer(audio_files)
            
            # Run analysis
            df = analyzer_instance.analyzer.analyze_directory()
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Failed to analyze audio files",
                    "details": ["No features could be extracted from the provided files"]
                }
            
            # Generate sequence recommendations
            sequence = analyzer_instance.analyzer.recommend_sequence()
            
            # Format results with enhanced information
            formatted_sequence = []
            cumulative_time = 0
            
            for rec in sequence:
                cumulative_time += rec['duration']
                
                sequence_item = {
                    "position": int(rec['position']),
                    "filename": str(rec['filename']),
                    "timing": {
                        "duration_seconds": float(rec['duration']),
                        "duration_formatted": f"{int(rec['duration']//60):02d}:{int(rec['duration']%60):02d}",
                        "cumulative_time_seconds": float(cumulative_time),
                        "cumulative_time_formatted": f"{int(cumulative_time//60):02d}:{int(cumulative_time%60):02d}"
                    },
                    "musical_properties": {
                        "tempo_bpm": float(rec['tempo']),
                        "detected_key": str(rec['key']),
                        "primary_mood": str(rec['mood']),
                        "primary_character": str(rec['character']),
                        "energy_level": float(rec['energy']),
                        "tempo_category": _categorize_tempo(rec['tempo']),
                        "energy_category": _categorize_energy(rec['energy'])
                    },
                    "sequencing": {
                        "placement_reasoning": str(rec['reasoning']),
                        "position_in_arc": _describe_position_in_arc(rec['position'], len(sequence)),
                        "transition_quality": _assess_transition_quality(rec, sequence)
                    }
                }
                formatted_sequence.append(sequence_item)
            
            return {
                "success": True,
                "analysis_type": "sequence_recommendation",
                "total_tracks": len(formatted_sequence),
                "session_directory": session_dir,
                "sequence": formatted_sequence,
                "summary": {
                    "total_duration_seconds": cumulative_time,
                    "total_duration_formatted": f"{int(cumulative_time//60):02d}:{int(cumulative_time%60):02d}",
                    "tempo_range": {
                        "min_bpm": min(item['musical_properties']['tempo_bpm'] for item in formatted_sequence),
                        "max_bpm": max(item['musical_properties']['tempo_bpm'] for item in formatted_sequence)
                    },
                    "energy_progression": {
                        "opening_energy": formatted_sequence[0]['musical_properties']['energy_level'],
                        "closing_energy": formatted_sequence[-1]['musical_properties']['energy_level'],
                        "progression_type": _analyze_energy_progression(formatted_sequence)
                    },
                    "mood_journey": [item['musical_properties']['primary_mood'] for item in formatted_sequence]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Sequence recommendation failed: {str(e)}",
                "details": ["Unexpected error during sequence generation"]
            }


    @mcp.tool()
    def analyze_audio_clusters(audio_files: List[Dict[str, Any]], 
                             n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform clustering analysis to group similar audio files.
        
        This tool uses K-means clustering to group tracks with similar musical
        characteristics, perfect for creating themed playlists, understanding
        compositional patterns, and organizing large collections.
        
        Args:
            audio_files: List of audio files with 'filename' and 'content' (base64 encoded)
            n_clusters: Number of clusters (optional, auto-determined if not provided)
        
        Returns:
            Dict with cluster analysis results and group characteristics
        """
        # Validate input files
        errors = analyzer_instance.validate_audio_files(audio_files)
        if errors:
            return {
                "success": False,
                "error": "Validation failed",
                "details": errors
            }
        
        try:
            # Set up analysis session
            session_dir = analyzer_instance.setup_analyzer(audio_files)
            
            # Run analysis
            df = analyzer_instance.analyzer.analyze_directory()
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Failed to analyze audio files",
                    "details": ["No features could be extracted from the provided files"]
                }
            
            # Determine cluster count
            if n_clusters is None:
                n_clusters = min(5, len(df))
            else:
                n_clusters = min(n_clusters, len(df))
            
            # Perform clustering
            cluster_labels, cluster_centers, feature_names = analyzer_instance.analyzer.perform_clustering(n_clusters=n_clusters)
            cluster_analysis = analyzer_instance.analyzer.cluster_analysis
            
            # Format cluster results
            formatted_clusters = {}
            for cluster_name, analysis in cluster_analysis.items():
                cluster_id = cluster_name.split('_')[1] if '_' in cluster_name else cluster_name
                
                formatted_clusters[cluster_id] = {
                    "cluster_info": {
                        "cluster_name": cluster_name,
                        "track_count": int(analysis.get('count', 0)),
                        "tracks": analysis.get('files', [])
                    },
                    "musical_characteristics": {
                        "avg_tempo_bpm": float(analysis.get('avg_tempo', 0)),
                        "avg_duration_seconds": float(analysis.get('avg_duration', 0)),
                        "avg_energy_level": float(analysis.get('avg_energy', 0)),
                        "avg_brightness_hz": float(analysis.get('avg_brightness', 0)),
                        "common_key": str(analysis.get('common_key', 'Unknown')),
                        "avg_phases": float(analysis.get('avg_phases', 0))
                    },
                    "creative_characteristics": {
                        "dominant_mood": str(analysis.get('dominant_mood', 'Unknown')),
                        "dominant_character": str(analysis.get('dominant_character', 'Unknown')),
                        "mood_diversity": int(analysis.get('mood_diversity', 0)),
                        "character_diversity": int(analysis.get('character_diversity', 0))
                    },
                    "structural_properties": {
                        "climax_percentage": float(analysis.get('has_climax_percent', 0)),
                        "breakdown_percentage": float(analysis.get('has_breakdown_percent', 0)),
                        "structural_coherence": str(analysis.get('musical_coherence', 'Unknown'))
                    },
                    "recommendations": _generate_cluster_recommendations(analysis)
                }
            
            return {
                "success": True,
                "analysis_type": "clustering_analysis",
                "total_tracks": len(df),
                "num_clusters": n_clusters,
                "session_directory": session_dir,
                "clusters": formatted_clusters,
                "summary": {
                    "largest_cluster_size": max(cluster['cluster_info']['track_count'] for cluster in formatted_clusters.values()),
                    "smallest_cluster_size": min(cluster['cluster_info']['track_count'] for cluster in formatted_clusters.values()),
                    "most_coherent_cluster": max(formatted_clusters.items(), 
                                               key=lambda x: x[1]['structural_properties']['climax_percentage'])[0],
                    "feature_count_used": len(feature_names)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Cluster analysis failed: {str(e)}",
                "details": ["Unexpected error during clustering analysis"]
            }


    @mcp.tool()
    def comprehensive_audio_analysis(audio_files: List[Dict[str, Any]], 
                                   export_format: str = "json") -> Dict[str, Any]:
        """
        Perform comprehensive analysis including all available analysis types.
        
        This tool runs the complete analysis pipeline including mood analysis,
        phase detection, clustering, and sequence recommendations, providing
        a complete picture of your musical collection.
        
        Args:
            audio_files: List of audio files with 'filename' and 'content' (base64 encoded)
            export_format: Format for results ("json", "summary", "detailed")
        
        Returns:
            Dict with complete analysis results across all analysis types
        """
        # Validate input files
        errors = analyzer_instance.validate_audio_files(audio_files)
        if errors:
            return {
                "success": False,
                "error": "Validation failed",
                "details": errors
            }
        
        try:
            # Set up analysis session
            session_dir = analyzer_instance.setup_analyzer(audio_files)
            
            # Run comprehensive analysis
            df = analyzer_instance.analyzer.analyze_directory()
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Failed to analyze audio files",
                    "details": ["No features could be extracted from the provided files"]
                }
            
            # Perform all analysis types
            n_clusters = min(5, len(df))
            cluster_labels, cluster_centers, feature_names = analyzer_instance.analyzer.perform_clustering(n_clusters=n_clusters)
            sequence_recommendations = analyzer_instance.analyzer.recommend_sequence()
            
            # Get all analysis results
            cluster_analysis = analyzer_instance.analyzer.cluster_analysis
            phase_data = analyzer_instance.analyzer.phase_data
            
            # Create comprehensive results
            comprehensive_results = {
                "success": True,
                "analysis_type": "comprehensive",
                "session_directory": session_dir,
                "collection_summary": {
                    "total_tracks": len(df),
                    "total_duration_minutes": float(df['duration'].sum()) / 60,
                    "unique_keys": df['key'].nunique() if 'key' in df.columns else (df['detected_key'].nunique() if 'detected_key' in df.columns else 0),
                    "avg_tempo_bpm": float(df['tempo'].mean()) if 'tempo' in df.columns else 0,
                    "dominant_mood": df['primary_mood'].mode().iloc[0] if 'primary_mood' in df.columns else 'Unknown',
                    "num_clusters": n_clusters,
                    "total_phases": sum(len(f['phases']) for f in phase_data),
                    "avg_phases_per_track": np.mean([len(f['phases']) for f in phase_data])
                }
            }
            
            # Add detailed results based on export format
            if export_format in ["json", "detailed"]:
                comprehensive_results["detailed_results"] = {
                    "track_analysis": df.to_dict('records'),
                    "phase_analysis": phase_data,
                    "cluster_analysis": cluster_analysis,
                    "sequence_recommendations": sequence_recommendations
                }
            
            if export_format in ["summary", "detailed"]:
                comprehensive_results["insights"] = {
                    "creative_insights": _generate_creative_insights(df),
                    "structural_insights": _generate_structural_insights(phase_data),
                    "clustering_insights": _generate_clustering_insights(cluster_analysis),
                    "sequencing_insights": _generate_sequencing_insights(sequence_recommendations)
                }
            
            return comprehensive_results
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comprehensive analysis failed: {str(e)}",
                "details": ["Unexpected error during comprehensive analysis"]
            }


    @mcp.tool()
    def get_supported_formats() -> Dict[str, Any]:
        """
        Get information about supported audio formats and analysis capabilities.
        
        Returns:
            Dict with format support, analysis capabilities, and usage guidelines
        """
        return {
            "supported_audio_formats": [".wav", ".aiff", ".aif", ".mp3"],
            "format_recommendations": {
                "best_quality": [".wav", ".aiff"],
                "most_compatible": [".mp3"],
                "recommended_for_analysis": [".wav"]
            },
            "file_constraints": {
                "max_file_size_mb": 100,
                "min_duration_seconds": 1,
                "max_duration_minutes": 30,
                "recommended_file_count": "1-20 files for optimal performance"
            },
            "analysis_capabilities": {
                "mood_descriptors": {
                    "core": ["spacey", "organic", "synthetic", "oozy", "pensive", "tense", "exuberant", "glitchy", "chaos"],
                    "extended": ["ethereal", "atmospheric", "crystalline", "warm", "melodic", "driving", "percussive", "droning"],
                    "total_count": 17
                },
                "character_tags": {
                    "synthesis": ["analog_synth", "digital_synth", "mellotron", "percussive_instrument", "acoustic_instrument"],
                    "texture": ["rich_texture", "pure_tone", "bright_harmonics", "warm_harmonics"],
                    "total_count": 9
                },
                "phase_types": [
                    "Intro/Ambient", "Introduction", "Build-up/Energetic", "Rhythmic/Percussive",
                    "Bright/Melodic", "Climax/Peak", "Development", "Breakdown/Quiet", 
                    "Conclusion", "Outro/Fade"
                ],
                "features_extracted_per_track": "80+"
            },
            "analysis_algorithms": {
                "phase_detection": "Energy and spectral change detection with music theory principles",
                "mood_classification": "Threshold-based classification optimized for synthesizer music",
                "character_analysis": "Spectral analysis combined with MFCC timbre fingerprinting",
                "clustering": "K-means with automatic cluster count optimization",
                "sequencing": "Musical flow principles considering key, tempo, energy, and mood"
            },
            "export_formats": ["json", "csv", "markdown"],
            "use_cases": [
                "Album sequencing and track ordering",
                "Playlist creation based on mood/energy",
                "Musical pattern analysis and insight generation",
                "Sound design palette analysis",
                "Compositional structure study",
                "DJ set preparation and flow optimization"
            ]
        }


    # Helper methods for MCP tools
    def _classify_structural_role(phase_type: str) -> str:
        """Classify the structural role of a phase type."""
        role_map = {
            "Intro/Ambient": "Opening/Atmospheric",
            "Introduction": "Opening",
            "Build-up/Energetic": "Development/Tension",
            "Rhythmic/Percussive": "Rhythmic Foundation",
            "Bright/Melodic": "Melodic Focus",
            "Climax/Peak": "Climactic/Peak Energy",
            "Development": "Musical Development",
            "Breakdown/Quiet": "Tension Release",
            "Conclusion": "Closing",
            "Outro/Fade": "Closing/Fade"
        }
        return role_map.get(phase_type, "Developmental")
    
    def _categorize_energy(energy: float) -> str:
        """Categorize energy level."""
        if energy < 0.02:
            return "Very Low"
        elif energy < 0.05:
            return "Low"
        elif energy < 0.08:
            return "Medium"
        elif energy < 0.12:
            return "High"
        else:
            return "Very High"
    
    def _categorize_brightness(brightness: float) -> str:
        """Categorize spectral brightness."""
        if brightness < 1000:
            return "Dark"
        elif brightness < 2000:
            return "Warm"
        elif brightness < 3000:
            return "Bright"
        else:
            return "Very Bright"
    
    def _categorize_tempo(tempo: float) -> str:
        """Categorize tempo."""
        if tempo < 80:
            return "Slow"
        elif tempo < 120:
            return "Medium"
        elif tempo < 140:
            return "Upbeat"
        else:
            return "Fast"
    
    def _assess_complexity(num_phases: int, duration: float) -> str:
        """Assess structural complexity."""
        complexity_score = num_phases / (duration / 60)  # phases per minute
        if complexity_score < 0.5:
            return "Simple"
        elif complexity_score < 1.0:
            return "Moderate"
        else:
            return "Complex"
    
    def _describe_position_in_arc(position: int, total: int) -> str:
        """Describe position in the overall arc."""
        ratio = position / total
        if ratio < 0.25:
            return "Opening Quarter"
        elif ratio < 0.5:
            return "First Half"
        elif ratio < 0.75:
            return "Second Half"
        else:
            return "Closing Quarter"
    
    def _assess_transition_quality(current_track: Dict, sequence: List[Dict]) -> str:
        """Assess the quality of transition to this track."""
        # Simplified transition quality assessment
        return "Good"  # Would implement actual transition scoring here
    
    def _analyze_energy_progression(sequence: List[Dict]) -> str:
        """Analyze the energy progression through the sequence."""
        energies = [track['musical_properties']['energy_level'] for track in sequence]
        start_energy = np.mean(energies[:2])
        end_energy = np.mean(energies[-2:])
        
        if end_energy > start_energy * 1.2:
            return "Building"
        elif end_energy < start_energy * 0.8:
            return "Declining"
        else:
            return "Stable"
    
    def _generate_cluster_recommendations(analysis: Dict) -> List[str]:
        """Generate recommendations for a cluster."""
        recommendations = []
        
        if analysis.get('has_climax_percent', 0) > 60:
            recommendations.append("High-energy cluster excellent for peak-time listening")
        
        if analysis.get('musical_coherence') == 'High Coherence':
            recommendations.append("Highly coherent - perfect for seamless playlists")
        
        if analysis.get('avg_tempo', 0) > 140:
            recommendations.append("Fast tempo cluster suitable for energetic contexts")
        
        return recommendations if recommendations else ["Versatile cluster for various contexts"]
    
    def _generate_creative_insights(df: pd.DataFrame) -> List[str]:
        """Generate creative insights from track analysis."""
        insights = []
        
        if 'primary_mood' in df.columns:
            dominant_mood = df['primary_mood'].mode().iloc[0]
            insights.append(f"Collection shows strong {dominant_mood} character")
        
        if 'tempo' in df.columns:
            tempo_std = df['tempo'].std()
            if tempo_std < 20:
                insights.append("Consistent tempo range suggests cohesive style")
            else:
                insights.append("Wide tempo range indicates diverse musical expression")
        
        return insights
    
    def _generate_structural_insights(phase_data: List[Dict]) -> List[str]:
        """Generate insights about musical structure."""
        insights = []
        
        avg_phases = np.mean([len(f['phases']) for f in phase_data])
        if avg_phases > 4:
            insights.append("Complex structural development with sophisticated phase progression")
        else:
            insights.append("Streamlined structure with clear, focused development")
        
        return insights
    
    def _generate_clustering_insights(cluster_analysis: Dict) -> List[str]:
        """Generate insights about clustering patterns."""
        insights = []
        
        cluster_sizes = [analysis.get('count', 0) for analysis in cluster_analysis.values()]
        if max(cluster_sizes) > len(cluster_sizes) * 0.6:
            insights.append("Strong stylistic consistency with dominant cluster")
        else:
            insights.append("Balanced distribution suggests diverse musical palette")
        
        return insights
    
    def _generate_sequencing_insights(sequence: List[Dict]) -> List[str]:
        """Generate insights about the recommended sequence."""
        insights = []
        
        tempos = [track['tempo'] for track in sequence]
        if tempos[-1] > tempos[0]:
            insights.append("Sequence builds energy from start to finish")
        else:
            insights.append("Sequence creates relaxing progression")
        
        return insights


# Make MCP server available for import
if MCP_AVAILABLE:
    __all__ = ['mcp', 'MCPAudioAnalyzer']
else:
    __all__ = ['MCPAudioAnalyzer']