"""
Character Analysis Module for Synthesizer Music

This module implements instrument and synthesis character detection for
electronic music analysis. It identifies the type of sound source and
its characteristics, helping composers understand their sonic palette.

The analytical approach is based on:
1. Spectral analysis to identify synthesis types
2. Harmonic analysis to detect instrument characteristics
3. MFCC analysis for timbre fingerprinting
4. Texture analysis for sound source identification

The character analysis system recognizes:
- Synthesis types: analog_synth, digital_synth, mellotron, etc.
- Instrument types: percussive_instrument, acoustic_instrument
- Texture types: rich_texture, pure_tone, bright_harmonics, warm_harmonics

Each character is defined by specific spectral and harmonic characteristics
that distinguish different synthesizer types and sound sources.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .descriptors import CharacterTags, CharacterTag


class CharacterAnalyzer:
    """
    Instrument and synthesis character analyzer for electronic music.
    
    This class identifies the type of sound source and its characteristics
    based on spectral analysis. It distinguishes between different
    synthesizer types, instrument categories, and textural qualities.
    
    The analysis process:
    1. Analyzes spectral characteristics (centroid, bandwidth, rolloff)
    2. Evaluates harmonic content and distribution
    3. Assesses temporal characteristics (zero-crossing rate)
    4. Applies MFCC analysis for timbre identification
    5. Combines evidence to determine character tags
    """
    
    def __init__(self):
        """Initialize the character analyzer with tag definitions."""
        self.character_tags = CharacterTags.get_all_tags()
        self.synthesis_tags = CharacterTags.get_synthesis_tag_names()
        self.texture_tags = CharacterTags.get_texture_tag_names()
        self.processing_tags = CharacterTags.get_processing_tag_names()
        
    def analyze_character(self, spectral_features: Dict[str, float],
                         confidence_threshold: float = 0.6) -> Tuple[List[str], Dict[str, float]]:
        """
        Analyze the character of an audio segment.
        
        This method evaluates the spectral characteristics to determine
        what type of sound source produced the audio. It considers multiple
        factors to identify synthesis types and textural qualities.
        
        The analysis considers:
        - Spectral centroid: Indicates brightness and harmonic distribution
        - Spectral bandwidth: Indicates harmonic complexity and texture
        - Spectral rolloff: Indicates high-frequency content and brightness
        - Zero-crossing rate: Indicates roughness and texture
        - MFCC characteristics: Provides timbre fingerprint
        
        Args:
            spectral_features: Dictionary containing spectral analysis results
            confidence_threshold: Minimum confidence for character assignment
            
        Returns:
            Tuple containing:
            - List of detected character tags
            - Dictionary of confidence scores for each character
        """
        # Convert input data to ensure proper numeric types
        from ..utils.type_conversion import convert_spectral_features_types
        spectral_features = convert_spectral_features_types(spectral_features)
        
        # Extract key spectral characteristics
        spectral_centroid = spectral_features.get('spectral_centroid_mean', 0)
        spectral_bandwidth = spectral_features.get('spectral_bandwidth_mean', 0)
        spectral_rolloff = spectral_features.get('spectral_rolloff_mean', 0)
        zero_crossing_rate = spectral_features.get('zero_crossing_rate_mean', 0)
        
        # Extract MFCC features for timbre analysis
        mfcc_features = self._extract_mfcc_characteristics(spectral_features)
        
        detected_characters = []
        confidence_scores = {}
        
        # Test each character tag
        for tag_name, tag_definition in self.character_tags.items():
            confidence = self._calculate_character_confidence(
                tag_definition, spectral_centroid, spectral_bandwidth,
                spectral_rolloff, zero_crossing_rate, mfcc_features,
                spectral_features
            )

            confidence_scores[tag_name] = confidence

            # Per-tag threshold overrides the global caller-supplied threshold
            effective_threshold = tag_definition.confidence_threshold or confidence_threshold
            if confidence >= effective_threshold:
                detected_characters.append(tag_name)
        
        # If no characters detected, use fallback analysis
        if not detected_characters:
            fallback_character = self._get_fallback_character(
                spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate
            )
            detected_characters.append(fallback_character)
            confidence_scores[fallback_character] = 0.5  # Medium confidence for fallback
        
        return detected_characters, confidence_scores
    
    def _extract_mfcc_characteristics(self, spectral_features: Dict[str, float]) -> Dict[str, float]:
        """
        Extract relevant MFCC characteristics for timbre analysis.
        
        This method processes MFCC features to identify characteristics
        that are relevant for synthesizer character identification.
        
        Args:
            spectral_features: Dictionary containing all spectral features
            
        Returns:
            Dictionary with processed MFCC characteristics
        """
        mfcc_characteristics = {}
        
        # Extract MFCC coefficients (first 5 are most relevant for character)
        for i in range(1, 6):
            mean_key = f'mfcc_{i}_mean'
            std_key = f'mfcc_{i}_std'
            
            if mean_key in spectral_features:
                mfcc_characteristics[f'mfcc_{i}_mean'] = spectral_features[mean_key]
            if std_key in spectral_features:
                mfcc_characteristics[f'mfcc_{i}_std'] = spectral_features[std_key]
        
        # Calculate derived characteristics
        if 'mfcc_1_mean' in mfcc_characteristics and 'mfcc_2_mean' in mfcc_characteristics:
            # MFCC ratio can indicate synthesis type
            mfcc1 = mfcc_characteristics['mfcc_1_mean']
            mfcc2 = mfcc_characteristics['mfcc_2_mean']
            if mfcc2 != 0:
                mfcc_characteristics['mfcc_ratio'] = abs(mfcc1 / mfcc2)
        
        # Calculate MFCC variance for texture analysis
        mfcc_vars = [mfcc_characteristics.get(f'mfcc_{i}_std', 0) for i in range(1, 6)]
        mfcc_characteristics['mfcc_variance'] = np.mean(mfcc_vars)
        
        return mfcc_characteristics
    
    def _calculate_character_confidence(self, tag_definition: CharacterTag,
                                      spectral_centroid: float, spectral_bandwidth: float,
                                      spectral_rolloff: float, zero_crossing_rate: float,
                                      mfcc_characteristics: Dict[str, float],
                                      spectral_features: Dict[str, float] = None) -> float:
        """
        Calculate confidence score for a specific character tag.

        Core four features are always scored. Optional extended features
        (spectral_flatness_range, spectral_flux_range, stereo_width_range) are
        scored only when the tag defines them AND the feature is present in
        spectral_features.

        Args:
            tag_definition: CharacterTag to evaluate
            spectral_centroid: Spectral centroid value
            spectral_bandwidth: Spectral bandwidth value
            spectral_rolloff: Spectral rolloff value
            zero_crossing_rate: Zero-crossing rate value
            mfcc_characteristics: Dictionary with MFCC characteristics
            spectral_features: Full feature dict for extended feature access

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if spectral_features is None:
            spectral_features = {}

        score = 0.0
        max_score = 0.0

        # Spectral centroid criterion (weight: 3.0)
        centroid_weight = 3.0
        if self._in_range(spectral_centroid, tag_definition.spectral_centroid_range):
            score += centroid_weight
        max_score += centroid_weight

        # Spectral bandwidth criterion (weight: 3.5)
        bandwidth_weight = 3.5
        if self._in_range(spectral_bandwidth, tag_definition.spectral_bandwidth_range):
            score += bandwidth_weight
        max_score += bandwidth_weight

        # Spectral rolloff criterion (weight: 2.5)
        rolloff_weight = 2.5
        if self._in_range(spectral_rolloff, tag_definition.spectral_rolloff_range):
            score += rolloff_weight
        max_score += rolloff_weight

        # Zero-crossing rate criterion (weight: 2.0)
        zcr_weight = 2.0
        if self._in_range(zero_crossing_rate, tag_definition.zero_crossing_rate_range):
            score += zcr_weight
        max_score += zcr_weight

        # MFCC characteristics (weight: 1.5) — only when data is available
        mfcc_weight = 1.5
        if self._evaluate_mfcc_characteristics(tag_definition.mfcc_characteristics, mfcc_characteristics):
            score += mfcc_weight
        max_score += mfcc_weight

        # Optional extended feature criteria — scored only when tag defines a range
        # and the feature was actually extracted.

        # Spectral flatness (weight: 3.0): noise-like (1.0) vs tonal (0.0)
        if tag_definition.spectral_flatness_range is not None:
            flatness = spectral_features.get('spectral_flatness_mean')
            max_score += 3.0  # always counts against max so missing feature penalises
            if flatness is not None and self._in_range(flatness, tag_definition.spectral_flatness_range):
                score += 3.0

        # Spectral flux (weight: 2.5): transient energy between frames
        if tag_definition.spectral_flux_range is not None:
            flux = spectral_features.get('spectral_flux_mean')
            max_score += 2.5
            if flux is not None and self._in_range(flux, tag_definition.spectral_flux_range):
                score += 2.5

        # Stereo width (weight: 5.0): L-R divergence ratio
        # Always counts against max so the tag cannot fire when stereo_width is absent.
        if tag_definition.stereo_width_range is not None:
            width = spectral_features.get('stereo_width')
            max_score += 5.0
            if width is not None and self._in_range(width, tag_definition.stereo_width_range):
                score += 5.0

        # Calculate normalized confidence
        confidence = score / max_score if max_score > 0 else 0.0

        # Apply bonus for strong matches
        if confidence > 0.85:
            confidence = min(1.0, confidence * 1.05)

        return confidence
    
    def _in_range(self, value: float, range_tuple: Tuple[float, float]) -> bool:
        """
        Check if a value falls within the specified range.
        
        Args:
            value: Value to check
            range_tuple: (min, max) tuple defining the range
            
        Returns:
            True if value is in range, False otherwise
        """
        from ..utils.validation import validate_range
        return validate_range(value, range_tuple)
    
    def _evaluate_mfcc_characteristics(self, expected_chars, actual_chars: Dict[str, float]) -> bool:
        """
        Evaluate MFCC characteristics against expected patterns.

        New-style tags pass a string: 'neutral' (always True), 'bright' (mfcc_1_mean > -10),
        or 'dark' (mfcc_1_mean < -10).  Legacy tags pass a dict with 'precision' or
        'complexity' keys.  Legacy dict tags that specify neither key fall through to True
        (neutral behaviour) just as before.

        Args:
            expected_chars: str or dict describing expected MFCC characteristics
            actual_chars: Actual computed MFCC characteristics dict

        Returns:
            True if characteristics match, False otherwise
        """
        # New-style: string descriptor
        if isinstance(expected_chars, str):
            if expected_chars == 'neutral':
                return True
            mfcc1 = actual_chars.get('mfcc_1_mean')
            if mfcc1 is None:
                return True  # no data — don't penalize
            if expected_chars == 'bright':
                return mfcc1 > -10
            if expected_chars == 'dark':
                return mfcc1 < -10
            return True  # unknown string → neutral

        # Legacy-style: dict descriptor
        if not isinstance(expected_chars, dict):
            return True

        if 'precision' in expected_chars:
            precision_level = expected_chars['precision']
            mfcc_variance = actual_chars.get('mfcc_variance', 0)
            if precision_level == 'high':
                return mfcc_variance < 0.5
            if precision_level == 'low':
                return mfcc_variance > 0.5

        if 'complexity' in expected_chars:
            complexity_level = expected_chars['complexity']
            mfcc_variance = actual_chars.get('mfcc_variance', 0)
            if complexity_level == 'high':
                return mfcc_variance > 0.8
            if complexity_level == 'low':
                return mfcc_variance < 0.3

        # Dict has keys we don't recognise — neutral
        return True
    
    def _get_fallback_character(self, spectral_centroid: float, spectral_bandwidth: float,
                              spectral_rolloff: float, zero_crossing_rate: float) -> str:
        """
        Provide fallback character classification.
        
        This method ensures that every audio segment receives some character
        classification, even if it doesn't strongly match any specific tag.
        
        Args:
            spectral_centroid: Spectral centroid value
            spectral_bandwidth: Spectral bandwidth value
            spectral_rolloff: Spectral rolloff value
            zero_crossing_rate: Zero-crossing rate value
            
        Returns:
            Fallback character tag
        """
        # Synthesis type heuristics
        if spectral_bandwidth < 1000 and zero_crossing_rate < 0.1:
            return 'analog_synth'
        elif spectral_centroid > 2000 and spectral_bandwidth < 800:
            return 'digital_synth'
        elif spectral_bandwidth > 1800 and zero_crossing_rate > 0.1:
            return 'acoustic_instrument'
        
        # Texture heuristics
        elif spectral_bandwidth > 2000:
            return 'rich_texture'
        elif spectral_bandwidth < 600:
            return 'pure_tone'
        elif spectral_rolloff > 4000:
            return 'bright_harmonics'
        elif spectral_rolloff < 2000:
            return 'warm_harmonics'
        
        # Default fallback
        return 'unknown'
    
    def analyze_track_character(self, features: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, float]]:
        """
        Analyze the character of a complete track.
        
        This method performs track-level character analysis using the
        extracted features. It identifies the primary synthesis type
        and textural characteristics of the track.
        
        Args:
            features: Dictionary containing all extracted track features
            
        Returns:
            Tuple containing:
            - List of detected character tags
            - Primary character (most confident)
            - Dictionary of confidence scores
        """
        # Prepare spectral features for analysis
        spectral_features = {
            'spectral_centroid_mean': features.get('spectral_centroid_mean', 0),
            'spectral_bandwidth_mean': features.get('spectral_bandwidth_mean', 0),
            'spectral_rolloff_mean': features.get('spectral_rolloff_mean', 0),
            'zero_crossing_rate_mean': features.get('zero_crossing_rate_mean', 0),
        }

        # Pass new discriminative features when present
        for key in ('spectral_flatness_mean', 'spectral_flux_mean', 'stereo_width'):
            if key in features:
                spectral_features[key] = features[key]

        # Add MFCC features
        for i in range(1, 14):
            mean_key = f'mfcc_{i}_mean'
            std_key = f'mfcc_{i}_std'
            if mean_key in features:
                spectral_features[mean_key] = features[mean_key]
            if std_key in features:
                spectral_features[std_key] = features[std_key]
        
        # Analyze character
        detected_characters, confidence_scores = self.analyze_character(spectral_features)
        
        # Determine primary character (highest confidence)
        primary_character = max(confidence_scores.items(), key=lambda x: x[1])[0]
        
        return detected_characters, primary_character, confidence_scores
    
    def get_character_description(self, character_name: str) -> str:
        """
        Get human-readable description for a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Description string
        """
        tag_definition = self.character_tags.get(character_name)
        if tag_definition:
            return tag_definition.description
        return f"Unknown character: {character_name}"
    
    def get_character_characteristics(self, character_name: str) -> Dict[str, Any]:
        """
        Get detailed characteristics for a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Dictionary with character characteristics
        """
        tag_definition = self.character_tags.get(character_name)
        if not tag_definition:
            return {}
        
        return {
            'name': tag_definition.name,
            'description': tag_definition.description,
            'spectral_centroid_range': tag_definition.spectral_centroid_range,
            'spectral_bandwidth_range': tag_definition.spectral_bandwidth_range,
            'spectral_rolloff_range': tag_definition.spectral_rolloff_range,
            'zero_crossing_rate_range': tag_definition.zero_crossing_rate_range,
            'mfcc_characteristics': tag_definition.mfcc_characteristics,
            'category': ('synthesis' if character_name in self.synthesis_tags 
                         else 'texture' if character_name in self.texture_tags 
                         else 'processing')
        }
    
    def analyze_character_distribution(self, track_characters: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of characters across a collection of tracks.
        
        This method provides insights into the sonic palette of a musical
        collection, useful for understanding production techniques and
        sound design patterns.
        
        Args:
            track_characters: List of primary characters for each track
            
        Returns:
            Dictionary with character distribution analysis
        """
        if not track_characters:
            return {}
        
        # Count character occurrences
        character_counts = {}
        for character in track_characters:
            character_counts[character] = character_counts.get(character, 0) + 1
        
        # Calculate percentages
        total_tracks = len(track_characters)
        character_percentages = {char: (count / total_tracks) * 100 
                               for char, count in character_counts.items()}
        
        # Identify dominant characters
        dominant_characters = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize characters
        synthesis_characters = [char for char in track_characters if char in self.synthesis_tags]
        texture_characters = [char for char in track_characters if char in self.texture_tags]
        processing_characters = [char for char in track_characters if char in self.processing_tags]
        
        return {
            'total_tracks': total_tracks,
            'unique_characters': len(character_counts),
            'character_counts': character_counts,
            'character_percentages': character_percentages,
            'dominant_characters': dominant_characters,
            'synthesis_character_ratio': len(synthesis_characters) / total_tracks,
            'texture_character_ratio': len(texture_characters) / total_tracks,
            'processing_character_ratio': len(processing_characters) / total_tracks,
            'character_diversity': len(character_counts) / len(self.character_tags)
        }
    
    def get_synthesis_profile(self, characters: List[str]) -> Dict[str, Any]:
        """
        Get synthesis profile for a collection of characters.
        
        This method analyzes the synthesis techniques used in a collection,
        providing insights into production methods and sound design.
        
        Args:
            characters: List of character names
            
        Returns:
            Dictionary with synthesis profile analysis
        """
        synthesis_profile = {
            'analog_proportion': 0,
            'digital_proportion': 0,
            'acoustic_proportion': 0,
            'texture_complexity': 'medium',
            'harmonic_tendency': 'neutral'
        }
        
        # Count synthesis types
        analog_count = characters.count('analog_synth')
        digital_count = characters.count('digital_synth')
        acoustic_count = characters.count('acoustic_instrument') + characters.count('mellotron')
        
        total_synthesis = analog_count + digital_count + acoustic_count
        
        if total_synthesis > 0:
            synthesis_profile['analog_proportion'] = analog_count / total_synthesis
            synthesis_profile['digital_proportion'] = digital_count / total_synthesis
            synthesis_profile['acoustic_proportion'] = acoustic_count / total_synthesis
        
        # Analyze texture complexity
        rich_texture_count = characters.count('rich_texture')
        pure_tone_count = characters.count('pure_tone')
        
        if rich_texture_count > pure_tone_count:
            synthesis_profile['texture_complexity'] = 'high'
        elif pure_tone_count > rich_texture_count:
            synthesis_profile['texture_complexity'] = 'low'
        
        # Analyze harmonic tendency
        bright_harmonics_count = characters.count('bright_harmonics')
        warm_harmonics_count = characters.count('warm_harmonics')
        
        if bright_harmonics_count > warm_harmonics_count:
            synthesis_profile['harmonic_tendency'] = 'bright'
        elif warm_harmonics_count > bright_harmonics_count:
            synthesis_profile['harmonic_tendency'] = 'warm'
        
        return synthesis_profile
    
    def suggest_character_combinations(self, current_character: str) -> List[str]:
        """
        Suggest characters that work well in combination.
        
        This method provides recommendations for character combinations,
        useful for layering sounds and creating complex textures.
        
        Args:
            current_character: Current character to combine with
            
        Returns:
            List of compatible characters
        """
        # Define character compatibility
        compatibility_map = {
            'analog_synth': ['warm_harmonics', 'organic', 'rich_texture'],
            'digital_synth': ['bright_harmonics', 'crystalline', 'pure_tone'],
            'mellotron': ['warm_harmonics', 'organic', 'rich_texture'],
            'percussive_instrument': ['bright_harmonics', 'rhythmic'],
            'acoustic_instrument': ['warm_harmonics', 'organic', 'rich_texture'],
            'rich_texture': ['analog_synth', 'mellotron', 'organic'],
            'pure_tone': ['digital_synth', 'crystalline', 'precise'],
            'bright_harmonics': ['digital_synth', 'percussive_instrument', 'energetic'],
            'warm_harmonics': ['analog_synth', 'mellotron', 'acoustic_instrument']
        }
        
        return compatibility_map.get(current_character, [])