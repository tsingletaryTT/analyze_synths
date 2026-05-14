"""
Mood Analysis Module for Synthesizer Music

This module implements creative mood classification using descriptors that
composers and electronic musicians can relate to. The analysis translates
technical audio features into emotional and atmospheric characteristics.

The analytical approach is based on:
1. Empirical analysis of synthesizer music characteristics
2. Mapping technical features to creative descriptors
3. Threshold-based classification optimized for electronic music
4. Multi-descriptor analysis for nuanced mood characterization

The mood analysis system recognizes 117 distinct moods organized into:
- Core moods: Fundamental emotional categories (9 descriptors)
- Extended moods: Nuanced atmospheric characteristics (8 descriptors)
- Advanced moods: Comprehensive creative vocabulary (100 descriptors)

Each mood is defined by specific combinations of:
- Energy levels (RMS power)
- Spectral brightness (spectral centroid)
- Texture roughness (zero-crossing rate)
- Rhythmic activity (onset density)
- Sustained characteristics (duration)
- Harmonic complexity (spectral bandwidth)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .descriptors import MoodDescriptors, MoodDescriptor


class MoodAnalyzer:
    """
    Creative mood analyzer for synthesizer music.
    
    This class implements sophisticated mood classification that goes beyond
    simple energy-based categorization. It considers multiple acoustic
    characteristics to identify the emotional and atmospheric qualities
    that define synthesizer music.
    
    The analysis process:
    1. Extracts relevant features from audio analysis
    2. Applies threshold-based classification for each mood
    3. Handles multi-mood assignments for complex textures
    4. Provides confidence scores for mood assignments
    5. Generates human-readable mood descriptions
    """
    
    def __init__(self):
        """Initialize the mood analyzer with descriptor definitions."""
        self.mood_descriptors = MoodDescriptors.get_all_descriptors()
        self.core_descriptors = MoodDescriptors.get_core_descriptor_names()
        self.extended_descriptors = MoodDescriptors.get_extended_descriptor_names()
        self.advanced_descriptors = MoodDescriptors.get_advanced_descriptor_names()
        
    def analyze_mood(self, phase_data: Dict[str, float], 
                    spectral_features: Dict[str, float],
                    confidence_threshold: float = 0.7) -> Tuple[List[str], Dict[str, float]]:
        """
        Analyze the mood of an audio segment using creative descriptors.
        
        This method implements the core mood analysis algorithm. It evaluates
        each potential mood descriptor against the audio characteristics and
        returns those that match the defined criteria.
        
        The analysis considers:
        - Energy characteristics: Overall power and dynamic range
        - Spectral characteristics: Brightness, bandwidth, harmonic content
        - Temporal characteristics: Rhythm, onset density, duration
        - Textural characteristics: Roughness, complexity, smoothness
        
        Args:
            phase_data: Dictionary containing phase-level audio characteristics
            spectral_features: Dictionary containing spectral analysis results
            confidence_threshold: Minimum confidence for mood assignment
            
        Returns:
            Tuple containing:
            - List of detected mood descriptors
            - Dictionary of confidence scores for each mood
        """
        # Convert input data to ensure proper numeric types
        from ..utils.type_conversion import convert_mood_analysis_input
        phase_data, spectral_features = convert_mood_analysis_input(phase_data, spectral_features)
        
        # Extract key metrics from converted input data
        from ..utils.type_conversion import safe_float_convert
        energy = safe_float_convert(phase_data.get('avg_energy', 0))
        brightness = safe_float_convert(phase_data.get('avg_brightness', 0))
        roughness = safe_float_convert(phase_data.get('avg_roughness', 0))
        onset_density = safe_float_convert(phase_data.get('onset_density', 0))
        duration = safe_float_convert(phase_data.get('duration', 0))
        
        # Extract spectral characteristics
        spectral_centroid = safe_float_convert(spectral_features.get('spectral_centroid_mean', brightness))
        spectral_bandwidth = safe_float_convert(spectral_features.get('spectral_bandwidth_mean', 0))
        spectral_rolloff = safe_float_convert(spectral_features.get('spectral_rolloff_mean', 0))
        zero_crossing_rate = safe_float_convert(spectral_features.get('zero_crossing_rate_mean', roughness))
        
        detected_moods = []
        confidence_scores = {}
        
        # Test each mood descriptor
        for mood_name, descriptor in self.mood_descriptors.items():
            # Exclude 'synthetic' from speech/voice content.
            # True synthesised sounds (pads, drones) have ZCR < 0.05; speech and
            # acoustic instruments alternate voiced/unvoiced frames keeping ZCR above 0.05.
            if mood_name == 'synthetic' and zero_crossing_rate > 0.05:
                confidence_scores[mood_name] = 0.0
                continue

            confidence = self._calculate_mood_confidence(
                descriptor, energy, brightness, roughness, onset_density,
                duration, spectral_centroid, spectral_bandwidth, spectral_rolloff,
                zero_crossing_rate
            )

            confidence_scores[mood_name] = confidence

            # Add mood if confidence exceeds threshold
            if confidence >= confidence_threshold:
                detected_moods.append(mood_name)
        
        # If no moods detected, use fallback analysis
        if not detected_moods:
            fallback_mood = self._get_fallback_mood(energy, brightness, onset_density)
            detected_moods.append(fallback_mood)
            confidence_scores[fallback_mood] = 0.5  # Medium confidence for fallback
        
        return detected_moods, confidence_scores
    
    def _calculate_mood_confidence(self, descriptor: MoodDescriptor, 
                                 energy: float, brightness: float, roughness: float,
                                 onset_density: float, duration: float,
                                 spectral_centroid: float, spectral_bandwidth: float,
                                 spectral_rolloff: float, zero_crossing_rate: float) -> float:
        """
        Calculate confidence score for a specific mood descriptor.
        
        This method evaluates how well the audio characteristics match
        the defined criteria for a specific mood. It uses a weighted
        scoring system that considers multiple factors.
        
        The confidence calculation:
        1. Tests each characteristic against the descriptor's ranges
        2. Calculates partial scores for each matching criterion
        3. Combines scores with appropriate weighting
        4. Returns normalized confidence value (0.0 to 1.0)
        
        Args:
            descriptor: MoodDescriptor to evaluate
            energy: RMS energy level
            brightness: Spectral centroid (brightness)
            roughness: Zero-crossing rate (roughness)
            onset_density: Rhythmic activity level
            duration: Duration of the segment
            spectral_centroid: Spectral centroid from features
            spectral_bandwidth: Spectral bandwidth from features
            spectral_rolloff: Spectral rolloff from features
            zero_crossing_rate: Zero-crossing rate from features
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.0
        max_score = 0.0
        
        # Energy criterion (weight: 3.0)
        # Energy is crucial for distinguishing between ambient and active moods
        energy_weight = 3.0
        if self._in_range(energy, descriptor.energy_range):
            score += energy_weight
        max_score += energy_weight
        
        # Brightness criterion (weight: 2.5)
        # Brightness distinguishes between dark and bright moods
        brightness_weight = 2.5
        brightness_value = max(brightness, spectral_centroid)
        if self._in_range(brightness_value, descriptor.brightness_range):
            score += brightness_weight
        max_score += brightness_weight
        
        # Roughness criterion (weight: 2.0)
        # Roughness distinguishes between smooth and textured moods
        roughness_weight = 2.0
        roughness_value = max(roughness, zero_crossing_rate)
        if self._in_range(roughness_value, descriptor.roughness_range):
            score += roughness_weight
        max_score += roughness_weight
        
        # Onset density criterion (weight: 2.0)
        # Onset density distinguishes between rhythmic and sustained moods
        onset_weight = 2.0
        if self._in_range(onset_density, descriptor.onset_density_range):
            score += onset_weight
        max_score += onset_weight
        
        # Duration criterion (weight: 1.0)
        # Duration is less critical but important for certain moods
        duration_weight = 1.0
        if duration >= descriptor.duration_threshold:
            score += duration_weight
        max_score += duration_weight
        
        # Spectral bandwidth criterion (weight: 1.5)
        # Bandwidth distinguishes between simple and complex textures
        bandwidth_weight = 1.5
        if self._in_range(spectral_bandwidth, descriptor.spectral_bandwidth_range):
            score += bandwidth_weight
        max_score += bandwidth_weight
        
        # Calculate normalized confidence
        confidence = score / max_score if max_score > 0 else 0.0
        
        # Apply bonus for strong matches
        if confidence > 0.8:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def _in_range(self, value: float, range_tuple: Tuple[float, float]) -> bool:
        """
        Check if a value falls within the specified range.
        
        This method handles both finite and infinite ranges, allowing
        for flexible descriptor definitions.
        
        Args:
            value: Value to check
            range_tuple: (min, max) tuple defining the range
            
        Returns:
            True if value is in range, False otherwise
        """
        from ..utils.validation import validate_range
        return validate_range(value, range_tuple)
    
    def _get_fallback_mood(self, energy: float, brightness: float, onset_density: float) -> str:
        """
        Provide fallback mood classification when no specific mood is detected.
        
        This method ensures that every audio segment receives some mood
        classification, even if it doesn't strongly match any specific
        descriptor. It uses simple heuristics based on the most important
        characteristics.
        
        Args:
            energy: RMS energy level
            brightness: Spectral brightness
            onset_density: Rhythmic activity
            
        Returns:
            Fallback mood descriptor
        """
        # Ensure all values are numeric for comparison
        from ..utils.type_conversion import safe_float_convert
        energy = safe_float_convert(energy, 0.0)
        brightness = safe_float_convert(brightness, 0.0)
        onset_density = safe_float_convert(onset_density, 0.0)
        # High energy moods - use advanced descriptors for more nuance
        if energy > 0.08:
            if onset_density > 4.0:
                return 'tempestuous' if brightness > 2500 else 'volcanic'
            elif onset_density > 2.0:
                return 'euphoric' if brightness > 2500 else 'surging'
            else:
                return 'triumphant' if brightness > 2200 else 'molten'
        
        # Low energy moods - ethereal and atmospheric categories
        elif energy < 0.02:
            if brightness > 3000:
                return 'celestial' if onset_density < 0.2 else 'gossamer'
            elif brightness > 2000:
                return 'luminous' if onset_density < 0.6 else 'spectral'
            else:
                return 'glacial' if onset_density < 0.1 else 'serene'
        
        # Medium energy moods - textural and emotional categories
        else:
            if brightness > 2500:
                return 'crystalline_hard' if onset_density > 1.5 else 'prismatic'
            elif onset_density > 2.0:
                return 'pulsating' if brightness > 1500 else 'granular'
            elif brightness < 1000:
                return 'velvet' if onset_density < 1.0 else 'wooden'
            else:
                return 'hopeful' if brightness > 1500 else 'contemplative'
    
    def analyze_track_mood(self, features: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, float]]:
        """
        Analyze the overall mood of a complete track.
        
        This method performs track-level mood analysis using the extracted
        features. It creates a comprehensive mood profile that represents
        the overall character of the track.
        
        Args:
            features: Dictionary containing all extracted track features
            
        Returns:
            Tuple containing:
            - List of detected mood descriptors
            - Primary mood (most confident)
            - Dictionary of confidence scores
        """
        # Prepare phase data for analysis
        phase_data = {
            'avg_energy': features.get('rms_mean', 0),
            'avg_brightness': features.get('spectral_centroid_mean', 0),
            'avg_roughness': features.get('zero_crossing_rate_mean', 0),
            'onset_density': features.get('onset_density', 0),
            'duration': features.get('duration', 0)
        }
        
        # Prepare spectral features
        spectral_features = {
            'spectral_centroid_mean': features.get('spectral_centroid_mean', 0),
            'spectral_bandwidth_mean': features.get('spectral_bandwidth_mean', 0),
            'spectral_rolloff_mean': features.get('spectral_rolloff_mean', 0),
            'zero_crossing_rate_mean': features.get('zero_crossing_rate_mean', 0)
        }
        
        # Analyze mood
        detected_moods, confidence_scores = self.analyze_mood(phase_data, spectral_features)
        
        # Determine primary mood (highest confidence)
        primary_mood = max(confidence_scores.items(), key=lambda x: x[1])[0]
        
        return detected_moods, primary_mood, confidence_scores
    
    def get_mood_description(self, mood_name: str) -> str:
        """
        Get human-readable description for a mood.
        
        Args:
            mood_name: Name of the mood
            
        Returns:
            Description string
        """
        descriptor = self.mood_descriptors.get(mood_name)
        if descriptor:
            return descriptor.description
        return f"Unknown mood: {mood_name}"
    
    def get_mood_characteristics(self, mood_name: str) -> Dict[str, Any]:
        """
        Get detailed characteristics for a mood.
        
        Args:
            mood_name: Name of the mood
            
        Returns:
            Dictionary with mood characteristics
        """
        descriptor = self.mood_descriptors.get(mood_name)
        if not descriptor:
            return {}
        
        return {
            'name': descriptor.name,
            'description': descriptor.description,
            'energy_range': descriptor.energy_range,
            'brightness_range': descriptor.brightness_range,
            'roughness_range': descriptor.roughness_range,
            'onset_density_range': descriptor.onset_density_range,
            'duration_threshold': descriptor.duration_threshold,
            'spectral_bandwidth_range': descriptor.spectral_bandwidth_range,
            'category': ('core' if mood_name in self.core_descriptors 
                         else 'extended' if mood_name in self.extended_descriptors 
                         else 'advanced')
        }
    
    def analyze_mood_distribution(self, track_moods: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of moods across a collection of tracks.
        
        This method provides insights into the overall mood characteristics
        of a musical collection, useful for understanding compositional
        patterns and creating themed playlists.
        
        Args:
            track_moods: List of primary moods for each track
            
        Returns:
            Dictionary with mood distribution analysis
        """
        if not track_moods:
            return {}
        
        # Count mood occurrences
        mood_counts = {}
        for mood in track_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        # Calculate percentages
        total_tracks = len(track_moods)
        mood_percentages = {mood: (count / total_tracks) * 100 
                           for mood, count in mood_counts.items()}
        
        # Identify dominant moods
        dominant_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize moods
        core_moods = [mood for mood in track_moods if mood in self.core_descriptors]
        extended_moods = [mood for mood in track_moods if mood in self.extended_descriptors]
        advanced_moods = [mood for mood in track_moods if mood in self.advanced_descriptors]
        
        return {
            'total_tracks': total_tracks,
            'unique_moods': len(mood_counts),
            'mood_counts': mood_counts,
            'mood_percentages': mood_percentages,
            'dominant_moods': dominant_moods,
            'core_mood_ratio': len(core_moods) / total_tracks,
            'extended_mood_ratio': len(extended_moods) / total_tracks,
            'advanced_mood_ratio': len(advanced_moods) / total_tracks,
            'mood_diversity': len(mood_counts) / len(self.mood_descriptors)
        }
    
    def suggest_mood_transitions(self, current_mood: str) -> List[str]:
        """
        Suggest moods that transition well from the current mood.
        
        This method provides recommendations for mood progression,
        useful for playlist creation and DJ set planning.
        
        Args:
            current_mood: Current mood to transition from
            
        Returns:
            List of compatible moods for transition
        """
        # Define mood transition compatibility - expanded for advanced descriptors
        transition_map = {
            # Original core and extended transitions
            'atmospheric': ['spacey', 'ethereal', 'oozy', 'pensive', 'warm', 'nebulous', 'vaporous', 'serene'],
            'spacey': ['atmospheric', 'ethereal', 'crystalline', 'pensive', 'celestial', 'gossamer', 'telescopic'],
            'ethereal': ['atmospheric', 'spacey', 'crystalline', 'warm', 'celestial', 'luminous', 'transcendent'],
            'oozy': ['atmospheric', 'warm', 'pensive', 'organic', 'viscous', 'gelatinous', 'molten'],
            'pensive': ['atmospheric', 'oozy', 'warm', 'melodic', 'contemplative', 'introspective', 'serene'],
            'warm': ['pensive', 'oozy', 'organic', 'melodic', 'velvet', 'compassionate', 'nostalgic'],
            'organic': ['warm', 'oozy', 'melodic', 'driving', 'wooden', 'fibrous', 'pastoral'],
            'melodic': ['pensive', 'warm', 'organic', 'driving', 'hopeful', 'flowing', 'legato'],
            'driving': ['melodic', 'organic', 'exuberant', 'percussive', 'pulsating', 'accelerating', 'ostinato'],
            'exuberant': ['driving', 'percussive', 'euphoric', 'triumphant', 'playful'],
            'tense': ['glitchy', 'chaos', 'anxious', 'vindictive', 'tempestuous'],
            'glitchy': ['tense', 'chaos', 'synthetic', 'granular', 'fractual', 'microscopic'],
            'chaos': ['glitchy', 'tense', 'tempestuous', 'churning', 'volcanic'],
            'synthetic': ['crystalline', 'glitchy', 'driving', 'plastic', 'metallic', 'ceramic'],
            'crystalline': ['synthetic', 'ethereal', 'spacey', 'crystalline_hard', 'prismatic', 'arctic'],
            'percussive': ['driving', 'exuberant', 'staccato', 'marcato', 'syncopated'],
            'droning': ['atmospheric', 'oozy', 'warm', 'glacial', 'fermata', 'subterranean'],
            
            # New advanced descriptor transitions
            'celestial': ['ethereal', 'spacey', 'luminous', 'transcendent', 'gossamer'],
            'gossamer': ['celestial', 'ethereal', 'iridescent', 'translucent'],
            'luminous': ['celestial', 'ethereal', 'hopeful', 'opalescent'],
            'nebulous': ['atmospheric', 'vaporous', 'mysterious', 'drifting'],
            'spectral': ['mysterious', 'brooding', 'arctic', 'telescopic'],
            'prismatic': ['crystalline', 'iridescent', 'dimensional', 'aurora'],
            'vaporous': ['nebulous', 'drifting', 'translucent', 'mirage'],
            'iridescent': ['gossamer', 'prismatic', 'aurora', 'opalescent'],
            'translucent': ['gossamer', 'vaporous', 'serene', 'flowing'],
            'opalescent': ['luminous', 'iridescent', 'warm', 'silky'],
            
            # Textural transitions
            'velvet': ['warm', 'silky', 'intimate', 'compassionate'],
            'silky': ['velvet', 'flowing', 'legato', 'opalescent'],
            'granular': ['glitchy', 'powdery', 'microscopic', 'urban'],
            'fibrous': ['organic', 'wooden', 'textured', 'woven'],
            'crystalline_hard': ['crystalline', 'metallic', 'arctic', 'ceramic'],
            'molten': ['oozy', 'volcanic', 'viscous', 'churning'],
            'metallic': ['crystalline_hard', 'synthetic', 'urban', 'industrial'],
            'ceramic': ['crystalline_hard', 'clean', 'resonant', 'precise'],
            'wooden': ['organic', 'warm', 'pastoral', 'natural'],
            'plastic': ['synthetic', 'uniform', 'artificial', 'clean'],
            'rubbery': ['elastic', 'bouncy', 'flexible', 'playful'],
            'spongy': ['absorbent', 'soft', 'textured', 'organic'],
            'viscous': ['oozy', 'molten', 'thick', 'slow'],
            'gelatinous': ['oozy', 'viscous', 'wobbling', 'soft'],
            'powdery': ['granular', 'dusty', 'diffuse', 'microscopic'],
            
            # Emotional transitions
            'euphoric': ['exuberant', 'ecstatic', 'triumphant', 'hopeful'],
            'melancholic': ['nostalgic', 'wistful', 'brooding', 'yearning'],
            'nostalgic': ['melancholic', 'wistful', 'warm', 'contemplative'],
            'anxious': ['tense', 'restless', 'claustrophobic', 'churning'],
            'serene': ['peaceful', 'calm', 'translucent', 'flowing'],
            'mysterious': ['enigmatic', 'spectral', 'brooding', 'dimensional'],
            'playful': ['whimsical', 'bouncy', 'rubbery', 'spiraling'],
            'brooding': ['melancholic', 'mysterious', 'dark', 'subterranean'],
            'hopeful': ['optimistic', 'luminous', 'rising', 'euphoric'],
            'longing': ['yearning', 'melancholic', 'reaching', 'distant'],
            'triumphant': ['victorious', 'euphoric', 'soaring', 'ecstatic'],
            'contemplative': ['thoughtful', 'pensive', 'introspective', 'meditative'],
            'yearning': ['longing', 'melancholic', 'poignant', 'reaching'],
            'ecstatic': ['euphoric', 'overwhelming', 'explosive', 'triumphant'],
            'introspective': ['contemplative', 'inward', 'pensive', 'quiet'],
            'wistful': ['nostalgic', 'melancholic', 'gentle', 'remembering'],
            'vindictive': ['vengeful', 'aggressive', 'sharp', 'intense'],
            'compassionate': ['gentle', 'caring', 'warm', 'enveloping'],
            'rebellious': ['defiant', 'challenging', 'irregular', 'chaotic'],
            'transcendent': ['spiritual', 'uplifting', 'celestial', 'ethereal']
        }
        
        return transition_map.get(current_mood, [])
    
    def get_mood_energy_profile(self, moods: List[str]) -> Dict[str, str]:
        """
        Get energy profile for a list of moods.
        
        This method categorizes moods by their energy characteristics,
        useful for understanding the dynamic range of a collection.
        
        Args:
            moods: List of mood names
            
        Returns:
            Dictionary mapping energy levels to mood lists
        """
        energy_profile = {
            'low_energy': [],
            'medium_energy': [],
            'high_energy': []
        }
        
        for mood in moods:
            descriptor = self.mood_descriptors.get(mood)
            if not descriptor:
                continue
            
            # Categorize based on energy range
            if descriptor.energy_range[1] <= 0.05:
                energy_profile['low_energy'].append(mood)
            elif descriptor.energy_range[0] >= 0.08:
                energy_profile['high_energy'].append(mood)
            else:
                energy_profile['medium_energy'].append(mood)
        
        return energy_profile