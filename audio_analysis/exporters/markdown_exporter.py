"""
Markdown Export Module for Audio Analysis Results

This module generates comprehensive, human-readable reports in Markdown format.
These reports are designed for composers, producers, and musicians who want to
understand their musical collections through detailed analysis and insights.

The Markdown exporter creates rich, formatted reports that include:
- Executive summaries with key insights
- Detailed track-by-track analysis
- Phase-by-phase structural breakdowns
- Cluster analysis for track grouping
- Sequence recommendations for optimal listening
- Creative insights and compositional patterns

The reports are optimized for readability while maintaining technical accuracy,
bridging the gap between complex audio analysis and practical musical insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class MarkdownExporter:
    """
    Comprehensive Markdown report generator for audio analysis results.
    
    This class creates detailed, formatted reports that translate technical
    analysis results into actionable insights for composers and musicians.
    The reports are designed to be both informative and engaging.
    """
    
    def __init__(self):
        """Initialize the Markdown exporter with default settings."""
        pass
    
    def generate_comprehensive_report(self, df: pd.DataFrame, phase_data: List[Dict[str, Any]],
                                    cluster_analysis: Optional[Dict[str, Any]] = None,
                                    sequence_recommendations: Optional[List[Dict[str, Any]]] = None,
                                    output_path: Path = None) -> str:
        """
        Generate a comprehensive analysis report in Markdown format.
        
        This method creates a complete report covering all aspects of the
        audio analysis, from high-level insights to detailed breakdowns.
        The report is structured for easy navigation and understanding.
        
        Args:
            df: DataFrame containing audio features
            phase_data: List of phase analysis results
            cluster_analysis: Cluster analysis results (optional)
            sequence_recommendations: Sequence recommendations (optional)
            output_path: Path to save the report (optional)
            
        Returns:
            Markdown content as string
        """
        try:
            # Build the complete report
            report_sections = []
            
            # Header and metadata
            report_sections.append(self._generate_header())
            
            # Executive summary
            report_sections.append(self._generate_executive_summary(df, phase_data))
            
            # Collection overview
            report_sections.append(self._generate_collection_overview(df, phase_data))
            
            # Sequence recommendations (if available)
            if sequence_recommendations:
                report_sections.append(self._generate_sequence_section(sequence_recommendations))
            
            # Track analysis
            report_sections.append(self._generate_track_analysis(df))
            
            # Phase analysis
            report_sections.append(self._generate_phase_analysis(phase_data))
            
            # Cluster analysis (if available)
            if cluster_analysis:
                report_sections.append(self._generate_cluster_analysis(cluster_analysis))
            
            # Creative insights
            report_sections.append(self._generate_creative_insights(df, phase_data))
            
            # Technical appendix
            report_sections.append(self._generate_technical_appendix(df))
            
            # Combine all sections
            full_report = '\n\n'.join(report_sections)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                print(f"Comprehensive report generated: {output_path}")
            
            return full_report
            
        except Exception as e:
            print(f"Error generating comprehensive report: {str(e)}")
            return ""
    
    def _generate_header(self) -> str:
        """Generate the report header with metadata."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""# Synthesizer Music Analysis Report

*Generated on: {timestamp}*

---

This report provides comprehensive analysis of your synthesizer music collection, combining technical audio analysis with creative insights. The analysis covers musical structure, mood characteristics, and composition patterns to help you understand and organize your work.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Collection Overview](#collection-overview)
3. [Recommended Listening Sequence](#recommended-listening-sequence)
4. [Individual Track Analysis](#individual-track-analysis)
5. [Musical Phase Analysis](#musical-phase-analysis)
6. [Track Clustering Analysis](#track-clustering-analysis)
7. [Creative Insights](#creative-insights)
8. [Technical Details](#technical-details)

---"""
    
    def _generate_executive_summary(self, df: pd.DataFrame, 
                                   phase_data: List[Dict[str, Any]]) -> str:
        """Generate executive summary section."""
        total_tracks = len(df)
        total_duration = df['duration'].sum() / 60  # in minutes
        total_phases = sum(len(f['phases']) for f in phase_data)
        avg_phases = np.mean([len(f['phases']) for f in phase_data])
        
        # Get dominant characteristics
        dominant_mood = df['primary_mood'].mode().iloc[0] if 'primary_mood' in df.columns else 'Unknown'
        dominant_character = df['primary_character'].mode().iloc[0] if 'primary_character' in df.columns else 'Unknown'
        
        # Calculate key insights
        _key_col = 'key' if 'key' in df.columns else ('detected_key' if 'detected_key' in df.columns else None)
        unique_keys = df[_key_col].nunique() if _key_col else 0
        avg_tempo = df['tempo'].mean() if 'tempo' in df.columns else 0
        
        return f"""## Executive Summary

Your collection contains **{total_tracks} tracks** with a total duration of **{total_duration:.1f} minutes** ({total_duration/60:.1f} hours). The analysis reveals a rich musical landscape with {total_phases} distinct musical phases across all tracks.

### Key Findings

- **Musical Diversity**: {unique_keys} different keys detected, indicating harmonic variety
- **Structural Complexity**: Average of {avg_phases:.1f} phases per track, showing sophisticated composition
- **Dominant Mood**: **{dominant_mood}** - the most prevalent emotional character
- **Primary Character**: **{dominant_character}** - the main synthesis/instrument type
- **Average Tempo**: {avg_tempo:.1f} BPM - indicates the overall energy and pace

### Creative Highlights

The collection demonstrates strong compositional variety with clear structural development in most tracks. The predominance of **{dominant_mood}** moods suggests a cohesive artistic vision while maintaining enough diversity to create engaging listening experiences."""
    
    def _generate_collection_overview(self, df: pd.DataFrame, 
                                    phase_data: List[Dict[str, Any]]) -> str:
        """Generate collection overview section."""
        # Musical characteristics
        tempo_stats = df['tempo'].describe() if 'tempo' in df.columns else None
        duration_stats = df['duration'].describe() if 'duration' in df.columns else None
        
        # Key distribution
        _key_col2 = 'key' if 'key' in df.columns else ('detected_key' if 'detected_key' in df.columns else None)
        key_distribution = df[_key_col2].value_counts().head(5) if _key_col2 else None
        
        # Mood distribution
        mood_distribution = df['primary_mood'].value_counts().head(5) if 'primary_mood' in df.columns else None
        
        overview = """## Collection Overview

### Musical Characteristics

"""
        
        if tempo_stats is not None:
            overview += f"""**Tempo Range**: {tempo_stats['min']:.0f} - {tempo_stats['max']:.0f} BPM
- Average: {tempo_stats['mean']:.1f} BPM
- Most tracks fall between {tempo_stats['25%']:.0f} - {tempo_stats['75%']:.0f} BPM

"""
        
        if duration_stats is not None:
            overview += f"""**Track Lengths**: {duration_stats['min']:.1f} - {duration_stats['max']:.1f} seconds
- Average length: {duration_stats['mean']:.1f} seconds ({duration_stats['mean']/60:.1f} minutes)
- Most tracks are {duration_stats['25%']:.1f} - {duration_stats['75%']:.1f} seconds long

"""
        
        if key_distribution is not None:
            overview += """**Key Distribution**:
"""
            for key, count in key_distribution.items():
                percentage = (count / len(df)) * 100
                overview += f"- **{key}**: {count} tracks ({percentage:.1f}%)\n"
            overview += "\n"
        
        if mood_distribution is not None:
            overview += """**Mood Distribution**:
"""
            for mood, count in mood_distribution.items():
                percentage = (count / len(df)) * 100
                overview += f"- **{mood.title()}**: {count} tracks ({percentage:.1f}%)\n"
        
        return overview
    
    def _generate_sequence_section(self, sequence_recommendations: List[Dict[str, Any]]) -> str:
        """Generate sequence recommendations section."""
        sequence = """## Recommended Listening Sequence

*A carefully curated flow designed to showcase your music's emotional journey*

The following sequence is optimized for musical flow, considering key relationships, tempo progression, energy arcs, and mood transitions:

"""
        
        for i, rec in enumerate(sequence_recommendations):
            duration_min = int(rec['duration'] // 60)
            duration_sec = int(rec['duration'] % 60)
            
            # Calculate cumulative time
            cumulative_seconds = sum(r['duration'] for r in sequence_recommendations[:i+1])
            cumulative_min = int(cumulative_seconds // 60)
            cumulative_sec = int(cumulative_seconds % 60)
            
            sequence += f"""### {rec['position']}. {rec['filename']}

**{rec['mood'].title()}** • **{rec['character'].replace('_', ' ').title()}** • {duration_min}:{duration_sec:02d} • {rec['tempo']:.0f} BPM • {rec['key']}

*{rec['reasoning']}*

*Cumulative time: {cumulative_min}:{cumulative_sec:02d}*

"""
        
        return sequence
    
    def _generate_track_analysis(self, df: pd.DataFrame) -> str:
        """Generate individual track analysis section."""
        analysis = """## Individual Track Analysis

*Detailed breakdown of each track's characteristics and creative elements*

"""
        
        for _, row in df.iterrows():
            duration_min = int(row['duration'] // 60)
            duration_sec = int(row['duration'] % 60)
            
            track_section = f"""### {row['filename']}

**Primary Mood**: {row.get('primary_mood', 'Unknown').title()} | **Character**: {row.get('primary_character', 'Unknown').replace('_', ' ').title()}

**Technical Specifications**:
- **Duration**: {duration_min}:{duration_sec:02d}
- **Tempo**: {row.get('tempo', 0):.1f} BPM
- **Key**: {row.get('key', row.get('detected_key', 'Unknown'))}
- **Energy Level**: {row.get('rms_mean', 0):.3f}
- **Spectral Brightness**: {row.get('spectral_centroid_mean', 0):.0f} Hz

**Creative Characteristics**:
- **Mood Descriptors**: {row.get('mood_descriptors', 'None')}
- **Character Tags**: {row.get('character_tags', 'None')}
- **Musical Phases**: {row.get('num_phases', 0)} distinct sections

"""
            
            # Add structural information if available
            if row.get('has_climax', False):
                track_section += "- **Structure**: Contains climactic sections - excellent for dynamic storytelling\n"
            if row.get('has_breakdown', False):
                track_section += "- **Structure**: Features breakdown sections - great for DJ mixing and transitions\n"
            
            track_section += "\n---\n\n"
            
            analysis += track_section
        
        return analysis
    
    def _generate_phase_analysis(self, phase_data: List[Dict[str, Any]]) -> str:
        """Generate phase analysis section."""
        analysis = """## Musical Phase Analysis

*Detailed structural analysis showing how your compositions evolve over time*

This section breaks down each track into its constituent musical phases, providing insights into your compositional structure and development techniques.

"""
        
        for file_info in phase_data:
            total_min = int(file_info['total_duration'] // 60)
            total_sec = int(file_info['total_duration'] % 60)
            
            file_section = f"""### {file_info['filename']}

**Overall Structure**: {total_min}:{total_sec:02d} duration • {file_info['num_phases']} phases

"""
            
            for phase in file_info['phases']:
                start_min = int(phase['start_time'] // 60)
                start_sec = int(phase['start_time'] % 60)
                end_min = int(phase['end_time'] // 60)
                end_sec = int(phase['end_time'] % 60)
                
                mood_text = ', '.join(phase.get('mood_descriptors', []))
                if mood_text:
                    mood_text = f" • *{mood_text.title()}*"
                
                file_section += f"""**Phase {phase['phase_number']}**: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} | *{phase['phase_type']}*{mood_text}
- Energy: {phase['avg_energy']:.4f} | Brightness: {phase['avg_brightness']:.0f} Hz | Rhythm: {phase['onset_density']:.1f} events/sec

"""
            
            analysis += file_section + "\n"
        
        return analysis
    
    def _generate_cluster_analysis(self, cluster_analysis: Dict[str, Any]) -> str:
        """Generate cluster analysis section."""
        analysis = """## Track Clustering Analysis

*Musical groupings based on similarity analysis - perfect for playlist creation*

The clustering analysis groups your tracks based on musical similarity, helping you understand patterns in your composition and create themed collections.

"""
        
        for cluster_name, cluster_data in cluster_analysis.items():
            cluster_id = cluster_name.split('_')[1] if '_' in cluster_name else cluster_name
            
            cluster_section = f"""### Cluster {cluster_id}

**{cluster_data['count']} tracks** • Avg Tempo: {cluster_data.get('avg_tempo', 0):.1f} BPM • Common Key: {cluster_data.get('common_key', 'Unknown')}

**Characteristics**:
- **Energy Level**: {cluster_data.get('avg_energy', 0):.3f} (Std: {cluster_data.get('energy_std', 0):.3f})
- **Spectral Brightness**: {cluster_data.get('avg_brightness', 0):.0f} Hz
- **Average Duration**: {cluster_data.get('avg_duration', 0):.1f} seconds
- **Dominant Mood**: {cluster_data.get('dominant_mood', 'Unknown').title()}
- **Dominant Character**: {cluster_data.get('dominant_character', 'Unknown').replace('_', ' ').title()}

**Tracks in this cluster**:
"""
            
            for filename in cluster_data.get('files', []):
                cluster_section += f"- {filename}\n"
            
            # Add recommendations
            if cluster_data.get('has_climax_percent', 0) > 50:
                cluster_section += f"\n🔥 **High Energy Group** - {cluster_data['has_climax_percent']:.0f}% of tracks have climactic sections\n"
            
            if cluster_data.get('has_breakdown_percent', 0) > 50:
                cluster_section += f"🌊 **Dynamic Range** - {cluster_data['has_breakdown_percent']:.0f}% feature breakdown/quiet sections\n"
            
            coherence = cluster_data.get('musical_coherence', 'Unknown')
            if coherence == 'High Coherence':
                cluster_section += "✨ **Highly Coherent** - Excellent for seamless playlists and DJ sets\n"
            
            cluster_section += "\n"
            analysis += cluster_section
        
        return analysis
    
    def _generate_creative_insights(self, df: pd.DataFrame, 
                                  phase_data: List[Dict[str, Any]]) -> str:
        """Generate creative insights section."""
        insights = """## Creative Insights

*Patterns and recommendations for your compositional development*

"""
        
        # Analyze compositional patterns
        if 'primary_mood' in df.columns:
            mood_counts = df['primary_mood'].value_counts()
            if len(mood_counts) > 0:
                dominant_mood = mood_counts.index[0]
                dominant_count = mood_counts.iloc[0]
                dominant_percentage = (dominant_count/len(df)*100)
                insights += f"""### Mood Palette Analysis

Your compositional style shows a preference for **{dominant_mood}** atmospheres, appearing in {dominant_count} tracks ({dominant_percentage:.1f}% of your collection).

**Top 3 Moods**:
"""
                for i, (mood, count) in enumerate(mood_counts.head(3).items()):
                    percentage = (count / len(df)) * 100
                    insights += f"{i+1}. **{mood.title()}**: {count} tracks ({percentage:.1f}%)\n"
            
            insights += "\n"
        
        if 'primary_character' in df.columns:
            character_counts = df['primary_character'].value_counts()
            insights += f"""### Sonic Palette Analysis

Your sound design gravitates toward **{character_counts.index[0].replace('_', ' ')}** textures, suggesting a distinctive sonic signature.

**Character Distribution**:
"""
            for char, count in character_counts.head(3).items():
                percentage = (count / len(df)) * 100
                insights += f"- **{char.replace('_', ' ').title()}**: {count} tracks ({percentage:.1f}%)\n"
            
            insights += "\n"
        
        # Structural analysis
        phase_lengths = [len(f['phases']) for f in phase_data]
        avg_complexity = np.mean(phase_lengths)
        
        insights += f"""### Structural Complexity

Your compositions average **{avg_complexity:.1f} phases per track**, indicating """
        
        if avg_complexity > 4:
            insights += "sophisticated structural development with clear musical narratives."
        elif avg_complexity > 2:
            insights += "moderate structural complexity with distinct musical sections."
        else:
            insights += "focused, streamlined compositions with clear, direct development."
        
        # Tempo analysis
        if 'tempo' in df.columns:
            tempo_std = df['tempo'].std()
            tempo_mean = df['tempo'].mean()
            
            insights += f"""

### Rhythmic Character

With an average tempo of **{tempo_mean:.1f} BPM** and standard deviation of {tempo_std:.1f}, your collection shows """
            
            if tempo_std < 20:
                insights += "consistent rhythmic character - excellent for cohesive album experiences."
            elif tempo_std < 40:
                insights += "moderate tempo variety - good balance between consistency and diversity."
            else:
                insights += "significant tempo diversity - great for dynamic, varied listening experiences."
        
        insights += """

### Recommendations for Future Work

Based on your compositional patterns:

1. **Explore Contrasts**: Consider experimenting with moods that appear less frequently in your current work
2. **Structural Variation**: Try varying your phase complexity to create different types of listening experiences  
3. **Harmonic Exploration**: Branch out into keys that are underrepresented in your collection
4. **Dynamic Range**: Experiment with more extreme energy variations for dramatic effect
5. **Character Blending**: Consider combining your dominant character types for richer textures"""
        
        return insights
    
    def _generate_technical_appendix(self, df: pd.DataFrame) -> str:
        """Generate technical appendix section."""
        appendix = """## Technical Details

*Analysis methodology and feature explanations*

### Analysis Overview

This report is based on comprehensive audio analysis using advanced signal processing and machine learning techniques specifically optimized for synthesizer music.

### Features Analyzed

**Total Features Extracted**: """ + str(len(df.columns)) + """ per track

**Core Feature Categories**:
- **Spectral Features**: Brightness, bandwidth, rolloff, and harmonic content analysis
- **Temporal Features**: Energy, dynamics, and rhythmic characteristics  
- **Harmonic Features**: Key detection, chroma analysis, and tonal content
- **Structural Features**: Phase detection and compositional development
- **Creative Features**: Mood classification and character identification

### Methodology

**Phase Detection**: Musical sections are identified using energy and spectral change detection with 2-second smoothing windows and gradient analysis.

**Mood Classification**: 17 creative descriptors based on energy, spectral, and temporal characteristics specifically calibrated for electronic music.

**Character Analysis**: Synthesis type identification using spectral analysis, harmonic content, and MFCC timbre fingerprinting.

**Clustering**: K-means clustering with automatic cluster count optimization and feature standardization.

**Sequence Optimization**: Musical flow principles considering key relationships, tempo progression, energy arcs, and mood compatibility.

### Creative Descriptors

**Mood Categories** (17 total):
- Core: spacey, organic, synthetic, oozy, pensive, tense, exuberant, glitchy, chaos
- Extended: ethereal, atmospheric, crystalline, warm, melodic, driving, percussive, droning

**Character Categories**:
- Synthesis: analog_synth, digital_synth, mellotron, percussive_instrument, acoustic_instrument  
- Texture: rich_texture, pure_tone, bright_harmonics, warm_harmonics

---

*Generated by AudioAnalyzer v1.0.0 - Synthesizer Music Analysis Toolkit*"""
        
        return appendix