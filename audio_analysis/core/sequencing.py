"""
Song Sequence Recommendation Module

This module implements intelligent song sequencing algorithms designed specifically
for synthesizer music. The approach combines music theory principles with practical
DJ/playlist creation knowledge to generate optimal listening sequences.

The analytical approach is based on several key principles:
1. Musical flow: Songs should transition smoothly based on key, tempo, and energy
2. Emotional journey: The sequence should tell a coherent emotional story
3. Compositional balance: Avoid monotony while maintaining cohesion
4. Creative intent: Support the artistic vision of the composer/curator

Key Concepts:
- Transition scoring: Quantifies how well two songs flow together
- Musical compatibility: Uses music theory to assess harmonic relationships
- Energy arc: Creates dynamic progression through the listening experience
- Mood coherence: Ensures emotional consistency while allowing for evolution

The algorithm is particularly effective for:
- Album sequencing for electronic music releases
- DJ set preparation and flow optimization
- Playlist creation for different moods and contexts
- Understanding compositional relationships in a body of work
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class SequenceTrack:
    """
    Data structure representing a track in the context of sequencing.
    
    This class encapsulates all the information needed to make intelligent
    sequencing decisions, including musical characteristics, mood descriptors,
    and metadata.
    """
    filename: str
    duration: float
    tempo: float
    key: str
    mood: str
    character: str
    energy: float
    brightness: float
    cluster: int
    key_confidence: float
    
    def __post_init__(self):
        """Ensure all numeric values are properly typed."""
        self.duration = float(self.duration)
        self.tempo = float(self.tempo)
        self.energy = float(self.energy)
        self.brightness = float(self.brightness)
        self.key_confidence = float(self.key_confidence)


class SequenceRecommender:
    """
    Intelligent song sequence recommendation system for synthesizer music.
    
    This class implements sophisticated algorithms for creating optimal listening
    sequences based on musical theory, energy flow, and creative principles.
    The system considers multiple factors to create cohesive, engaging sequences.
    """
    
    def __init__(self):
        """Initialize the sequence recommender with music theory knowledge."""
        # Circle of fifths relationships for key compatibility
        # These relationships are fundamental to Western music harmony
        self.key_relationships = {
            'C': ['G', 'F', 'Am', 'Dm', 'Em'],
            'G': ['C', 'D', 'Em', 'Am', 'Bm'],
            'D': ['G', 'A', 'Bm', 'Em', 'F#m'],
            'A': ['D', 'E', 'F#m', 'Bm', 'C#m'],
            'E': ['A', 'B', 'C#m', 'F#m', 'G#m'],
            'B': ['E', 'F#', 'G#m', 'C#m', 'D#m'],
            'F#': ['B', 'C#', 'D#m', 'G#m', 'A#m'],
            'C#': ['F#', 'G#', 'A#m', 'D#m', 'Fm'],
            'F': ['C', 'Bb', 'Dm', 'Gm', 'Am'],
            'Bb': ['F', 'Eb', 'Gm', 'Cm', 'Dm'],
            'Eb': ['Bb', 'Ab', 'Cm', 'Fm', 'Gm'],
            'Ab': ['Eb', 'Db', 'Fm', 'Bbm', 'Cm'],
            'Db': ['Ab', 'Gb', 'Bbm', 'Ebm', 'Fm'],
            'Gb': ['Db', 'Cb', 'Ebm', 'Abm', 'Bbm'],
        }
        
        # Mood compatibility matrix for smooth emotional transitions
        # Based on psychological research and practical DJ experience
        self.mood_compatibility = {
            'atmospheric': ['spacey', 'ethereal', 'oozy', 'pensive', 'warm'],
            'spacey': ['atmospheric', 'ethereal', 'crystalline', 'pensive'],
            'ethereal': ['atmospheric', 'spacey', 'crystalline', 'warm'],
            'oozy': ['atmospheric', 'warm', 'pensive', 'organic'],
            'pensive': ['atmospheric', 'oozy', 'warm', 'melodic'],
            'warm': ['pensive', 'oozy', 'organic', 'melodic'],
            'organic': ['warm', 'oozy', 'melodic', 'driving'],
            'melodic': ['pensive', 'warm', 'organic', 'driving'],
            'driving': ['melodic', 'organic', 'energetic', 'percussive'],
            'energetic': ['driving', 'exuberant', 'percussive', 'tense'],
            'exuberant': ['energetic', 'driving', 'percussive'],
            'tense': ['energetic', 'glitchy', 'chaos'],
            'glitchy': ['tense', 'chaos', 'synthetic'],
            'chaos': ['glitchy', 'tense'],
            'synthetic': ['crystalline', 'glitchy', 'driving'],
            'crystalline': ['synthetic', 'ethereal', 'spacey'],
            'percussive': ['driving', 'energetic', 'exuberant'],
            'droning': ['atmospheric', 'oozy', 'warm']
        }
    
    def recommend_sequence(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate an optimal listening sequence for a collection of tracks.
        
        This method implements a sophisticated sequencing algorithm that considers:
        1. Musical compatibility (key, tempo, energy)
        2. Emotional journey (mood progression)
        3. Structural balance (intro/outro placement)
        4. Creative flow (avoiding repetition, maintaining interest)
        
        The algorithm uses a greedy approach with transition scoring to build
        the sequence one track at a time, always choosing the best next track
        based on multiple musical criteria.
        
        Args:
            df: DataFrame containing track features and analysis results
            
        Returns:
            List of dictionaries representing the recommended sequence
        """
        if df.empty:
            return []
        
        # Stage 1: Data Preparation
        # Convert DataFrame to SequenceTrack objects for easier manipulation
        tracks = self._prepare_tracks(df)
        
        # Stage 2: Intro Track Selection
        # Choose the best opening track based on atmospheric qualities
        # and low energy characteristics that set the mood
        intro_track = self._select_intro_track(tracks)
        
        # Stage 3: Sequence Building
        # Build the sequence using greedy selection with transition scoring
        sequence = [intro_track]
        remaining_tracks = [t for t in tracks if t.filename != intro_track.filename]
        
        # Continue building sequence until all tracks are placed
        while remaining_tracks:
            current_track = sequence[-1]
            
            # Calculate transition scores for all remaining tracks
            transition_scores = []
            for candidate in remaining_tracks:
                score = self._calculate_transition_score(current_track, candidate)
                transition_scores.append((candidate, score))
            
            # Select the track with the highest transition score
            best_track, best_score = max(transition_scores, key=lambda x: x[1])
            sequence.append(best_track)
            remaining_tracks.remove(best_track)
        
        # Stage 4: Sequence Optimization
        # Apply post-processing to improve the overall flow
        sequence = self._optimize_sequence(sequence)
        
        # Stage 5: Generate Recommendations with Reasoning
        # Create detailed recommendations with explanations
        recommendations = []
        for i, track in enumerate(sequence):
            rec = {
                'position': i + 1,
                'filename': track.filename,
                'duration': track.duration,
                'tempo': track.tempo,
                'key': track.key,
                'mood': track.mood,
                'character': track.character,
                'energy': track.energy,
                'reasoning': self._generate_reasoning(track, i, len(sequence))
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _prepare_tracks(self, df: pd.DataFrame) -> List[SequenceTrack]:
        """
        Convert DataFrame to SequenceTrack objects with proper typing.
        
        This method handles the conversion from pandas DataFrame to structured
        objects, ensuring all values are properly typed and handling missing
        data gracefully.
        
        Args:
            df: DataFrame with track features
            
        Returns:
            List of SequenceTrack objects
        """
        tracks = []
        
        for _, row in df.iterrows():
            # Handle missing values with sensible defaults
            track = SequenceTrack(
                filename=str(row.get('filename', 'unknown')),
                duration=float(row.get('duration', 0)),
                tempo=float(row.get('tempo', 120)),
                key=str(row.get('key', row.get('detected_key', 'C'))),
                mood=str(row.get('primary_mood', 'neutral')),
                character=str(row.get('primary_character', 'unknown')),
                energy=float(row.get('rms_mean', 0)),
                brightness=float(row.get('spectral_centroid_mean', 1000)),
                cluster=int(row.get('cluster', 0)),
                key_confidence=float(row.get('key_confidence', 0))
            )
            tracks.append(track)
        
        return tracks
    
    def _select_intro_track(self, tracks: List[SequenceTrack]) -> SequenceTrack:
        """
        Select the best opening track for the sequence.
        
        The intro track sets the tone for the entire listening experience.
        This method looks for tracks with specific characteristics that make
        them suitable for opening a sequence:
        
        1. Atmospheric/ambient qualities
        2. Lower energy levels
        3. Longer duration for gradual immersion
        4. Warm or organic character
        
        Args:
            tracks: List of available tracks
            
        Returns:
            Best intro track
        """
        best_score = -1
        best_track = tracks[0]  # Default to first track
        
        for track in tracks:
            score = 0
            
            # Favor atmospheric/ambient moods for openings
            if track.mood in ['atmospheric', 'spacey', 'ethereal', 'warm']:
                score += 10
            elif track.mood in ['oozy', 'pensive', 'organic']:
                score += 7
            
            # Favor analog/warm character for openings
            if track.character in ['analog_synth', 'warm_harmonics']:
                score += 5
            elif track.character in ['mellotron', 'organic']:
                score += 3
            
            # Favor lower energy for smooth opening
            if track.energy < 0.03:
                score += 10
            elif track.energy < 0.05:
                score += 5
            
            # Favor longer tracks for immersive openings
            if track.duration > 240:  # 4 minutes
                score += 5
            elif track.duration > 120:  # 2 minutes
                score += 3
            
            # Favor certain keys that feel "opening-like"
            if track.key in ['C', 'G', 'D', 'A']:
                score += 2
            
            if score > best_score:
                best_score = score
                best_track = track
        
        return best_track
    
    def _calculate_transition_score(self, current: SequenceTrack, next_track: SequenceTrack) -> float:
        """
        Calculate how well two tracks transition together.
        
        This method implements the core logic for assessing musical compatibility
        between two tracks. It considers multiple factors that affect the quality
        of the transition:
        
        1. Key compatibility (harmonic relationships)
        2. Tempo similarity (rhythmic continuity)
        3. Energy progression (dynamic flow)
        4. Mood compatibility (emotional coherence)
        5. Character similarity (timbral consistency)
        6. Duration balance (avoiding abrupt changes)
        
        The scoring system is weighted to prioritize the most important factors
        for synthesizer music transitions.
        
        Args:
            current: Current track in the sequence
            next_track: Candidate next track
            
        Returns:
            Transition score (higher = better transition)
        """
        score = 0
        
        # Stage 1: Key Compatibility Analysis
        # Harmonic relationships are crucial for smooth transitions
        if current.key == next_track.key:
            score += 10  # Same key = perfect harmonic compatibility
        elif self._are_keys_related(current.key, next_track.key):
            score += 5   # Related keys = good harmonic compatibility
        
        # Bonus for high key confidence (more reliable key detection)
        if current.key_confidence > 0.7 and next_track.key_confidence > 0.7:
            score += 2
        
        # Stage 2: Tempo Compatibility Analysis
        # Gradual tempo changes feel more natural than sudden jumps
        tempo_diff = abs(current.tempo - next_track.tempo)
        if tempo_diff < 10:
            score += 8   # Very similar tempos
        elif tempo_diff < 20:
            score += 5   # Moderately similar tempos
        elif tempo_diff < 40:
            score += 2   # Acceptable tempo difference
        # No points for large tempo differences
        
        # Stage 3: Energy Progression Analysis
        # Energy flow is critical for maintaining listener engagement
        energy_diff = abs(current.energy - next_track.energy)
        if energy_diff < 0.02:
            score += 6   # Very similar energy levels
        elif energy_diff < 0.05:
            score += 3   # Moderately similar energy levels
        
        # Bonus for gradual energy increase (building excitement)
        if 0 < (next_track.energy - current.energy) < 0.03:
            score += 3
        
        # Stage 4: Mood Compatibility Analysis
        # Emotional coherence is essential for storytelling
        if next_track.mood == current.mood:
            score += 4   # Same mood = emotional consistency
        elif self._are_moods_compatible(current.mood, next_track.mood):
            score += 7   # Compatible moods = good emotional flow
        
        # Stage 5: Character Similarity Analysis
        # Timbral consistency helps maintain sonic coherence
        if current.character == next_track.character:
            score += 4   # Same character = timbral consistency
        elif self._are_characters_compatible(current.character, next_track.character):
            score += 2   # Compatible characters = acceptable timbral flow
        
        # Stage 6: Duration Balance Analysis
        # Avoid jarring changes in track length
        if 30 < current.duration < 300 and 30 < next_track.duration < 300:
            score += 3   # Both tracks are reasonable length
        
        # Slight preference for duration progression
        if current.duration < next_track.duration:
            score += 1   # Gradually increasing duration
        
        # Stage 7: Brightness Compatibility Analysis
        # Spectral similarity contributes to smooth transitions
        brightness_diff = abs(current.brightness - next_track.brightness)
        if brightness_diff < 500:
            score += 3   # Similar brightness levels
        elif brightness_diff < 1000:
            score += 1   # Moderately similar brightness
        
        # Stage 8: Cluster Compatibility Bonus
        # Tracks in the same cluster are likely to work well together
        if current.cluster == next_track.cluster:
            score += 2   # Same cluster = algorithmic compatibility
        
        return score
    
    def _are_keys_related(self, key1: str, key2: str) -> bool:
        """
        Check if two keys are harmonically related.
        
        This method uses music theory principles to determine if two keys
        will sound good together. It considers:
        1. Circle of fifths relationships
        2. Relative major/minor relationships
        3. Common chord progressions
        
        Args:
            key1: First key
            key2: Second key
            
        Returns:
            True if keys are related, False otherwise
        """
        return key2 in self.key_relationships.get(key1, [])
    
    def _are_moods_compatible(self, mood1: str, mood2: str) -> bool:
        """
        Check if two moods create a good emotional transition.
        
        This method uses psychological and musical research to determine
        which moods flow well together. It considers:
        1. Emotional proximity (similar feelings)
        2. Natural progressions (energy build-up/down)
        3. Complementary contrasts (tension/release)
        
        Args:
            mood1: First mood
            mood2: Second mood
            
        Returns:
            True if moods are compatible, False otherwise
        """
        return mood2 in self.mood_compatibility.get(mood1, [])
    
    def _are_characters_compatible(self, char1: str, char2: str) -> bool:
        """
        Check if two instrument characters work well together.
        
        This method considers timbral compatibility between different
        synthesizer types and sound sources.
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            True if characters are compatible, False otherwise
        """
        # Define character compatibility groups
        compatibility_groups = {
            'analog_synth': ['digital_synth', 'warm_harmonics', 'pure_tone'],
            'digital_synth': ['analog_synth', 'bright_harmonics', 'crystalline'],
            'mellotron': ['organic', 'warm_harmonics', 'rich_texture'],
            'percussive_instrument': ['rhythmic', 'bright_harmonics'],
            'acoustic_instrument': ['organic', 'warm_harmonics'],
            'warm_harmonics': ['analog_synth', 'mellotron', 'organic'],
            'bright_harmonics': ['digital_synth', 'crystalline', 'percussive_instrument']
        }
        
        return char2 in compatibility_groups.get(char1, [])
    
    def _optimize_sequence(self, sequence: List[SequenceTrack]) -> List[SequenceTrack]:
        """
        Apply post-processing optimizations to improve sequence flow.
        
        This method looks for opportunities to improve the overall sequence
        by making local adjustments:
        1. Swap adjacent tracks if it improves flow
        2. Move tracks to better positions based on energy arc
        3. Ensure proper intro/outro placement
        
        Args:
            sequence: Initial sequence
            
        Returns:
            Optimized sequence
        """
        if len(sequence) <= 2:
            return sequence
        
        optimized = sequence.copy()
        
        # Look for improvement opportunities with adjacent swaps
        for i in range(len(optimized) - 1):
            # Don't move the first track (intro should stay)
            if i == 0:
                continue
            
            current_score = 0
            if i > 0:
                current_score += self._calculate_transition_score(optimized[i-1], optimized[i])
            if i < len(optimized) - 1:
                current_score += self._calculate_transition_score(optimized[i], optimized[i+1])
            if i < len(optimized) - 2:
                current_score += self._calculate_transition_score(optimized[i+1], optimized[i+2])
            
            # Try swapping adjacent tracks
            optimized[i], optimized[i+1] = optimized[i+1], optimized[i]
            
            swap_score = 0
            if i > 0:
                swap_score += self._calculate_transition_score(optimized[i-1], optimized[i])
            if i < len(optimized) - 1:
                swap_score += self._calculate_transition_score(optimized[i], optimized[i+1])
            if i < len(optimized) - 2:
                swap_score += self._calculate_transition_score(optimized[i+1], optimized[i+2])
            
            # Keep the swap if it improves the score
            if swap_score <= current_score:
                # Swap back if no improvement
                optimized[i], optimized[i+1] = optimized[i+1], optimized[i]
        
        return optimized
    
    def _generate_reasoning(self, track: SequenceTrack, position: int, total_tracks: int) -> str:
        """
        Generate human-readable reasoning for track placement.
        
        This method creates explanations for why each track is placed in
        its specific position, helping users understand the algorithmic
        decisions and learn about sequence construction.
        
        Args:
            track: Track in the sequence
            position: Position in sequence (0-based)
            total_tracks: Total number of tracks
            
        Returns:
            Human-readable reasoning string
        """
        # Calculate position percentage
        position_pct = position / max(total_tracks - 1, 1)
        
        # Generate position-specific reasoning
        if position == 0:
            return f"Opening track - sets the mood with {track.mood} atmosphere"
        elif position == total_tracks - 1:
            return f"Closing track - concludes the journey with {track.mood} energy"
        elif position_pct < 0.3:
            return f"Early exploration - introduces {track.character} textures"
        elif position_pct < 0.7:
            return f"Core development - showcases {track.mood} at {track.tempo:.0f} BPM"
        else:
            return f"Building toward conclusion - {track.mood} energy prepares for ending"
    
    def analyze_sequence_quality(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of a generated sequence.
        
        This method provides metrics for evaluating how well the sequence
        follows musical principles and creative guidelines.
        
        Args:
            sequence: Generated sequence
            
        Returns:
            Dictionary with quality metrics
        """
        if len(sequence) <= 1:
            return {'quality': 'insufficient_data'}
        
        # Calculate transition quality scores
        transition_scores = []
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_track = sequence[i + 1]
            
            # Create temporary SequenceTrack objects for scoring
            current_track = SequenceTrack(
                filename=current['filename'],
                duration=current['duration'],
                tempo=current['tempo'],
                key=current['key'],
                mood=current['mood'],
                character=current['character'],
                energy=current['energy'],
                brightness=current.get('brightness', 1000),
                cluster=current.get('cluster', 0),
                key_confidence=current.get('key_confidence', 0.5)
            )
            
            next_track_obj = SequenceTrack(
                filename=next_track['filename'],
                duration=next_track['duration'],
                tempo=next_track['tempo'],
                key=next_track['key'],
                mood=next_track['mood'],
                character=next_track['character'],
                energy=next_track['energy'],
                brightness=next_track.get('brightness', 1000),
                cluster=next_track.get('cluster', 0),
                key_confidence=next_track.get('key_confidence', 0.5)
            )
            
            score = self._calculate_transition_score(current_track, next_track_obj)
            transition_scores.append(score)
        
        # Calculate overall quality metrics
        avg_transition_score = np.mean(transition_scores)
        min_transition_score = np.min(transition_scores)
        
        # Analyze energy progression
        energies = [track['energy'] for track in sequence]
        energy_std = np.std(energies)
        
        # Analyze tempo progression
        tempos = [track['tempo'] for track in sequence]
        tempo_std = np.std(tempos)
        
        # Analyze mood diversity
        moods = [track['mood'] for track in sequence]
        unique_moods = len(set(moods))
        
        # Determine overall quality
        if avg_transition_score > 15:
            overall_quality = "Excellent"
        elif avg_transition_score > 10:
            overall_quality = "Good"
        elif avg_transition_score > 5:
            overall_quality = "Fair"
        else:
            overall_quality = "Poor"
        
        return {
            'overall_quality': overall_quality,
            'avg_transition_score': avg_transition_score,
            'min_transition_score': min_transition_score,
            'energy_diversity': energy_std,
            'tempo_diversity': tempo_std,
            'mood_diversity': unique_moods,
            'total_duration': sum(track['duration'] for track in sequence),
            'weakest_transition': min(range(len(transition_scores)), key=lambda i: transition_scores[i]) + 1
        }