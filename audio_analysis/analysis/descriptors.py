"""
Creative Descriptors for Synthesizer Music Analysis

This module defines the creative vocabulary used to describe synthesizer music
characteristics. The descriptors are designed to be intuitive for composers
and producers, bridging the gap between technical analysis and artistic understanding.

The descriptor system is based on:
1. Creative language that artists naturally use
2. Empirical analysis of synthesizer music characteristics
3. Psychological research on musical emotion and perception
4. Practical experience from electronic music production

The descriptors are organized into two main categories:
- Mood Descriptors: Emotional and atmospheric characteristics
- Character Tags: Instrument and synthesis characteristics
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MoodDescriptor:
    """
    Definition of a mood descriptor with its characteristics and thresholds.
    
    This class encapsulates the rules and thresholds used to detect
    specific moods in synthesizer music. Each descriptor includes
    the technical criteria that must be met for identification.
    """
    name: str
    description: str
    energy_range: tuple
    brightness_range: tuple
    roughness_range: tuple
    onset_density_range: tuple
    duration_threshold: float
    spectral_bandwidth_range: tuple


@dataclass
class CharacterTag:
    """
    Definition of a character tag with its technical characteristics.

    This class defines the technical criteria used to identify
    different synthesizer types and sound sources.

    Optional extended features (spectral_flatness_range, spectral_flux_range,
    stereo_width_range) are scored in addition to the core four when present.
    confidence_threshold overrides the global threshold for this specific tag.
    """
    name: str
    description: str
    spectral_centroid_range: tuple
    spectral_bandwidth_range: tuple
    spectral_rolloff_range: tuple
    zero_crossing_rate_range: tuple
    mfcc_characteristics: Any  # str ('neutral', 'bright', 'dark') or legacy Dict
    # Optional extended feature ranges — None means not scored for this tag
    spectral_flatness_range: Optional[tuple] = None
    spectral_flux_range: Optional[tuple] = None
    stereo_width_range: Optional[tuple] = None
    # Per-tag confidence threshold; 0.0 means use the caller-supplied global threshold
    confidence_threshold: float = 0.0


class MoodDescriptors:
    """
    Complete definition of all mood descriptors for synthesizer music.
    
    This class contains the comprehensive set of mood descriptors
    used to characterize the emotional and atmospheric qualities
    of electronic music. Each descriptor is defined with specific
    technical thresholds derived from analysis of synthesizer music.
    """
    
    # Core mood descriptors - the fundamental emotional categories
    CORE_DESCRIPTORS = {
        'spacey': MoodDescriptor(
            name='spacey',
            description='Ethereal, otherworldly atmosphere with sustained tones and minimal rhythm',
            energy_range=(0.0, 0.03),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.0, 0.1),
            onset_density_range=(0.0, 0.5),
            duration_threshold=20.0,
            spectral_bandwidth_range=(0.0, 1500)
        ),
        
        'organic': MoodDescriptor(
            name='organic',
            description='Natural, textured sounds with harmonic richness and moderate energy',
            energy_range=(0.02, 0.08),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.1, float('inf')),
            onset_density_range=(0.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'synthetic': MoodDescriptor(
            name='synthetic',
            description='Precise, artificial textures with clean spectral characteristics',
            energy_range=(0.0, float('inf')),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.0, 0.05),
            onset_density_range=(0.0, float('inf')),
            duration_threshold=0.0,
            # Tightened from 1000 Hz to 800 Hz: speech codec narrowing sits above 800 Hz
            spectral_bandwidth_range=(0.0, 800)
        ),
        
        'oozy': MoodDescriptor(
            name='oozy',
            description='Slow, flowing textures with low energy and sustained character',
            energy_range=(0.0, 0.025),
            brightness_range=(0.0, 1500),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.0, 0.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'pensive': MoodDescriptor(
            name='pensive',
            description='Contemplative, balanced energy with moderate brightness and steady rhythm',
            energy_range=(0.03, 0.06),
            brightness_range=(1200, 2200),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.5, 1.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'tense': MoodDescriptor(
            name='tense',
            description='High energy with sharp spectral content and active rhythm',
            energy_range=(0.08, float('inf')),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(2.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'exuberant': MoodDescriptor(
            name='exuberant',
            description='High energy, bright, and rhythmically active with joyful character',
            energy_range=(0.1, float('inf')),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(1.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'glitchy': MoodDescriptor(
            name='glitchy',
            description='Irregular, fragmented textures with high roughness and complex rhythms',
            energy_range=(0.0, float('inf')),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.15, float('inf')),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'chaos': MoodDescriptor(
            name='chaos',
            description='Extreme energy and rhythm with unpredictable spectral characteristics',
            energy_range=(0.15, float('inf')),
            brightness_range=(0.0, 800),  # Can be very bright OR very dark
            roughness_range=(0.0, float('inf')),
            onset_density_range=(4.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        )
    }
    
    # Extended mood descriptors - more nuanced emotional categories
    EXTENDED_DESCRIPTORS = {
        'ethereal': MoodDescriptor(
            name='ethereal',
            description='Delicate, floating textures with very low energy and high brightness',
            energy_range=(0.0, 0.02),
            brightness_range=(3000, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.0, 0.3),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'atmospheric': MoodDescriptor(
            name='atmospheric',
            description='Ambient, environmental textures with minimal rhythm and wide spectrum',
            energy_range=(0.0, 0.015),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.0, 0.2),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'crystalline': MoodDescriptor(
            name='crystalline',
            description='Clear, precise textures with high brightness and low roughness',
            energy_range=(0.0, float('inf')),
            brightness_range=(2800, float('inf')),
            roughness_range=(0.0, 0.04),
            onset_density_range=(0.5, 1.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'warm': MoodDescriptor(
            name='warm',
            description='Comfortable, enveloping textures with moderate brightness and rich harmonics',
            energy_range=(0.03, 0.08),
            brightness_range=(800, 1800),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1200, float('inf'))
        ),
        
        'melodic': MoodDescriptor(
            name='melodic',
            description='Tuneful, balanced textures with moderate characteristics',
            energy_range=(0.04, 0.09),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.3, 1.2),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1000, 1600)
        ),
        
        'driving': MoodDescriptor(
            name='driving',
            description='Consistent, forward-moving energy with regular rhythm',
            energy_range=(0.05, 0.12),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(1.0, 2.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(800, 1800)
        ),
        
        'percussive': MoodDescriptor(
            name='percussive',
            description='Rhythmically active with high onset density and focused energy',
            energy_range=(0.06, float('inf')),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        ),
        
        'droning': MoodDescriptor(
            name='droning',
            description='Sustained, minimal rhythm with continuous energy',
            energy_range=(0.01, float('inf')),
            brightness_range=(0.0, float('inf')),
            roughness_range=(0.0, float('inf')),
            onset_density_range=(0.0, 0.1),
            duration_threshold=15.0,
            spectral_bandwidth_range=(0.0, float('inf'))
        )
    }
    
    # Advanced mood descriptors - comprehensive creative vocabulary (100 new descriptors)
    ADVANCED_DESCRIPTORS = {
        # Ethereal and atmospheric (10 descriptors)
        'celestial': MoodDescriptor(
            name='celestial',
            description='Heavenly, transcendent textures with very high brightness and minimal rhythm',
            energy_range=(0.0, 0.015),
            brightness_range=(3500, float('inf')),
            roughness_range=(0.0, 0.03),
            onset_density_range=(0.0, 0.2),
            duration_threshold=25.0,
            spectral_bandwidth_range=(0.0, 1200)
        ),
        
        'gossamer': MoodDescriptor(
            name='gossamer',
            description='Delicate, web-like textures that shimmer with high-frequency content',
            energy_range=(0.0, 0.02),
            brightness_range=(4000, float('inf')),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.0, 0.4),
            duration_threshold=15.0,
            spectral_bandwidth_range=(0.0, 1000)
        ),
        
        'luminous': MoodDescriptor(
            name='luminous',
            description='Glowing, radiant textures with sustained brightness and gentle energy',
            energy_range=(0.01, 0.04),
            brightness_range=(3200, float('inf')),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.0, 0.6),
            duration_threshold=20.0,
            spectral_bandwidth_range=(800, 1800)
        ),
        
        'nebulous': MoodDescriptor(
            name='nebulous',
            description='Cloud-like, undefined textures with soft edges and evolving character',
            energy_range=(0.005, 0.03),
            brightness_range=(1500, 2800),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.0, 0.5),
            duration_threshold=30.0,
            spectral_bandwidth_range=(1200, 2500)
        ),
        
        'spectral': MoodDescriptor(
            name='spectral',
            description='Ghostly, haunting textures with eerie brightness and sparse rhythm',
            energy_range=(0.0, 0.025),
            brightness_range=(2800, 4500),
            roughness_range=(0.0, 0.05),
            onset_density_range=(0.0, 0.3),
            duration_threshold=18.0,
            spectral_bandwidth_range=(600, 1400)
        ),
        
        'prismatic': MoodDescriptor(
            name='prismatic',
            description='Multi-faceted textures that split and refract like light through crystal',
            energy_range=(0.02, 0.06),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.04, 0.15),
            onset_density_range=(0.2, 0.8),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'vaporous': MoodDescriptor(
            name='vaporous',
            description='Steam-like, ephemeral textures that drift and dissipate',
            energy_range=(0.0, 0.02),
            brightness_range=(1800, 3500),
            roughness_range=(0.01, 0.08),
            onset_density_range=(0.0, 0.4),
            duration_threshold=22.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'iridescent': MoodDescriptor(
            name='iridescent',
            description='Color-shifting textures with shimmering high-frequency variations',
            energy_range=(0.01, 0.045),
            brightness_range=(3000, float('inf')),
            roughness_range=(0.05, 0.18),
            onset_density_range=(0.3, 1.0),
            duration_threshold=10.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'translucent': MoodDescriptor(
            name='translucent',
            description='Semi-transparent textures allowing underlying harmonics to show through',
            energy_range=(0.005, 0.035),
            brightness_range=(2200, 3800),
            roughness_range=(0.0, 0.07),
            onset_density_range=(0.0, 0.5),
            duration_threshold=16.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'opalescent': MoodDescriptor(
            name='opalescent',
            description='Milky, pearl-like textures with subtle color shifts and gentle luminosity',
            energy_range=(0.01, 0.04),
            brightness_range=(2000, 3200),
            roughness_range=(0.02, 0.09),
            onset_density_range=(0.1, 0.6),
            duration_threshold=14.0,
            spectral_bandwidth_range=(1200, 2000)
        ),
        
        # Textural and tactile (15 descriptors)
        'velvet': MoodDescriptor(
            name='velvet',
            description='Smooth, luxurious textures with rich mid-range warmth',
            energy_range=(0.03, 0.08),
            brightness_range=(600, 1600),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.2, 1.0),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1000, 1800)
        ),
        
        'silky': MoodDescriptor(
            name='silky',
            description='Ultra-smooth textures with flowing, continuous character',
            energy_range=(0.02, 0.07),
            brightness_range=(800, 2000),
            roughness_range=(0.0, 0.04),
            onset_density_range=(0.1, 0.8),
            duration_threshold=6.0,
            spectral_bandwidth_range=(800, 1500)
        ),
        
        'granular': MoodDescriptor(
            name='granular',
            description='Sandy, particulate textures with fine-grained roughness',
            energy_range=(0.02, 0.12),
            brightness_range=(1200, float('inf')),
            roughness_range=(0.08, 0.25),
            onset_density_range=(1.5, 4.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'fibrous': MoodDescriptor(
            name='fibrous',
            description='Thread-like textures with interwoven harmonic strands',
            energy_range=(0.025, 0.09),
            brightness_range=(1000, 2500),
            roughness_range=(0.06, 0.18),
            onset_density_range=(0.8, 2.2),
            duration_threshold=5.0,
            spectral_bandwidth_range=(1400, 2800)
        ),
        
        'crystalline_hard': MoodDescriptor(
            name='crystalline_hard',
            description='Diamond-hard textures with precise edges and brilliant clarity',
            energy_range=(0.04, 0.15),
            brightness_range=(3500, float('inf')),
            roughness_range=(0.0, 0.03),
            onset_density_range=(0.5, 2.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(0.0, 800)
        ),
        
        'molten': MoodDescriptor(
            name='molten',
            description='Lava-like textures that flow and bubble with intense heat',
            energy_range=(0.08, 0.2),
            brightness_range=(400, 1200),
            roughness_range=(0.1, 0.3),
            onset_density_range=(0.5, 2.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'metallic': MoodDescriptor(
            name='metallic',
            description='Cold, industrial textures with sharp harmonic edges',
            energy_range=(0.05, 0.18),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.03, 0.15),
            onset_density_range=(0.8, 3.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1200, 2500)
        ),
        
        'ceramic': MoodDescriptor(
            name='ceramic',
            description='Porcelain-like textures with clean, resonant characteristics',
            energy_range=(0.03, 0.1),
            brightness_range=(1800, 3500),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.4, 1.8),
            duration_threshold=3.0,
            spectral_bandwidth_range=(900, 1600)
        ),
        
        'wooden': MoodDescriptor(
            name='wooden',
            description='Organic, woody textures with natural resonance and warmth',
            energy_range=(0.04, 0.12),
            brightness_range=(500, 1400),
            roughness_range=(0.05, 0.2),
            onset_density_range=(0.3, 2.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'plastic': MoodDescriptor(
            name='plastic',
            description='Artificial, synthetic textures with uniform characteristics',
            energy_range=(0.03, 0.11),
            brightness_range=(1200, 2800),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.5, 2.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(600, 1200)
        ),
        
        'rubbery': MoodDescriptor(
            name='rubbery',
            description='Elastic, bouncy textures with flexible dynamic response',
            energy_range=(0.05, 0.15),
            brightness_range=(800, 2200),
            roughness_range=(0.04, 0.16),
            onset_density_range=(1.0, 3.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'spongy': MoodDescriptor(
            name='spongy',
            description='Absorbent, porous textures with soft attack and decay',
            energy_range=(0.02, 0.08),
            brightness_range=(600, 1800),
            roughness_range=(0.08, 0.22),
            onset_density_range=(0.5, 2.0),
            duration_threshold=4.0,
            spectral_bandwidth_range=(1300, 2600)
        ),
        
        'viscous': MoodDescriptor(
            name='viscous',
            description='Thick, syrupy textures that move slowly and deliberately',
            energy_range=(0.01, 0.06),
            brightness_range=(400, 1200),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.0, 1.0),
            duration_threshold=12.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'gelatinous': MoodDescriptor(
            name='gelatinous',
            description='Jelly-like textures that wobble and undulate with soft definition',
            energy_range=(0.02, 0.07),
            brightness_range=(700, 1600),
            roughness_range=(0.03, 0.14),
            onset_density_range=(0.2, 1.2),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'powdery': MoodDescriptor(
            name='powdery',
            description='Fine, dusty textures with soft, diffuse characteristics',
            energy_range=(0.01, 0.05),
            brightness_range=(1000, 2500),
            roughness_range=(0.1, 0.3),
            onset_density_range=(0.8, 3.0),
            duration_threshold=2.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        # Emotional and psychological (20 descriptors)
        'euphoric': MoodDescriptor(
            name='euphoric',
            description='Blissful, uplifting textures with soaring brightness and energetic rhythm',
            energy_range=(0.1, float('inf')),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.0, 0.1),
            onset_density_range=(1.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1200, float('inf'))
        ),
        
        'melancholic': MoodDescriptor(
            name='melancholic',
            description='Sorrowful, introspective textures with gentle sadness',
            energy_range=(0.02, 0.06),
            brightness_range=(800, 1800),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.2, 1.0),
            duration_threshold=10.0,
            spectral_bandwidth_range=(900, 1600)
        ),
        
        'nostalgic': MoodDescriptor(
            name='nostalgic',
            description='Wistful, memory-evoking textures with vintage warmth',
            energy_range=(0.025, 0.07),
            brightness_range=(600, 1600),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.3, 1.2),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1100, 2000)
        ),
        
        'anxious': MoodDescriptor(
            name='anxious',
            description='Restless, uneasy textures with irregular patterns and tension',
            energy_range=(0.06, 0.14),
            brightness_range=(1800, float('inf')),
            roughness_range=(0.08, 0.25),
            onset_density_range=(2.0, 5.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'serene': MoodDescriptor(
            name='serene',
            description='Peaceful, tranquil textures with balanced, calming characteristics',
            energy_range=(0.015, 0.05),
            brightness_range=(1000, 2200),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.1, 0.8),
            duration_threshold=15.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'mysterious': MoodDescriptor(
            name='mysterious',
            description='Enigmatic textures that hint at hidden depths and secrets',
            energy_range=(0.02, 0.08),
            brightness_range=(1200, 2800),
            roughness_range=(0.04, 0.16),
            onset_density_range=(0.3, 1.5),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1300, 2600)
        ),
        
        'playful': MoodDescriptor(
            name='playful',
            description='Whimsical, lighthearted textures with bouncy, cheerful character',
            energy_range=(0.06, 0.16),
            brightness_range=(1800, float('inf')),
            roughness_range=(0.02, 0.12),
            onset_density_range=(1.2, 3.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1000, 2200)
        ),
        
        'brooding': MoodDescriptor(
            name='brooding',
            description='Dark, contemplative textures with underlying intensity',
            energy_range=(0.03, 0.09),
            brightness_range=(400, 1200),
            roughness_range=(0.02, 0.15),
            onset_density_range=(0.2, 1.5),
            duration_threshold=15.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'hopeful': MoodDescriptor(
            name='hopeful',
            description='Optimistic textures with rising energy and brightening character',
            energy_range=(0.04, 0.12),
            brightness_range=(1500, 3000),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.5, 2.0),
            duration_threshold=6.0,
            spectral_bandwidth_range=(1100, 2000)
        ),
        
        'longing': MoodDescriptor(
            name='longing',
            description='Yearning textures with emotional pull and gentle ache',
            energy_range=(0.025, 0.08),
            brightness_range=(900, 2000),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.2, 1.2),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1000, 1800)
        ),
        
        'triumphant': MoodDescriptor(
            name='triumphant',
            description='Victorious textures with bold energy and soaring brightness',
            energy_range=(0.12, float('inf')),
            brightness_range=(2200, float('inf')),
            roughness_range=(0.0, 0.12),
            onset_density_range=(1.8, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'contemplative': MoodDescriptor(
            name='contemplative',
            description='Thoughtful, meditative textures that encourage reflection',
            energy_range=(0.02, 0.06),
            brightness_range=(800, 2000),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.1, 0.8),
            duration_threshold=18.0,
            spectral_bandwidth_range=(900, 1700)
        ),
        
        'yearning': MoodDescriptor(
            name='yearning',
            description='Deeply emotional textures with poignant, reaching quality',
            energy_range=(0.03, 0.1),
            brightness_range=(1200, 2500),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.3, 1.5),
            duration_threshold=10.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'ecstatic': MoodDescriptor(
            name='ecstatic',
            description='Overwhelming joy with explosive energy and brilliant brightness',
            energy_range=(0.15, float('inf')),
            brightness_range=(3000, float('inf')),
            roughness_range=(0.05, 0.2),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'introspective': MoodDescriptor(
            name='introspective',
            description='Inward-looking textures that promote self-examination',
            energy_range=(0.015, 0.05),
            brightness_range=(600, 1600),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.1, 0.6),
            duration_threshold=20.0,
            spectral_bandwidth_range=(800, 1400)
        ),
        
        'wistful': MoodDescriptor(
            name='wistful',
            description='Gently sad textures tinged with sweet remembrance',
            energy_range=(0.02, 0.06),
            brightness_range=(900, 1900),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.2, 1.0),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1000, 1800)
        ),
        
        'vindictive': MoodDescriptor(
            name='vindictive',
            description='Vengeful textures with sharp edges and aggressive intent',
            energy_range=(0.1, float('inf')),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.12, 0.4),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'compassionate': MoodDescriptor(
            name='compassionate',
            description='Gentle, caring textures with warm, enveloping character',
            energy_range=(0.03, 0.08),
            brightness_range=(800, 2000),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.2, 1.2),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1200, 2000)
        ),
        
        'rebellious': MoodDescriptor(
            name='rebellious',
            description='Defiant textures that challenge conventions with irregular patterns',
            energy_range=(0.08, 0.2),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.1, 0.35),
            onset_density_range=(2.0, 6.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'transcendent': MoodDescriptor(
            name='transcendent',
            description='Spiritually uplifting textures that rise above earthly concerns',
            energy_range=(0.02, 0.1),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.1, 1.0),
            duration_threshold=25.0,
            spectral_bandwidth_range=(800, 1800)
        ),
        
        # Dynamic and kinetic (15 descriptors)
        'pulsating': MoodDescriptor(
            name='pulsating',
            description='Rhythmic, throbbing textures with regular energy variations',
            energy_range=(0.04, 0.15),
            brightness_range=(1000, float('inf')),
            roughness_range=(0.02, 0.12),
            onset_density_range=(1.5, 4.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1000, 2500)
        ),
        
        'undulating': MoodDescriptor(
            name='undulating',
            description='Wave-like textures with smooth, flowing energy variations',
            energy_range=(0.03, 0.1),
            brightness_range=(800, 2500),
            roughness_range=(0.01, 0.08),
            onset_density_range=(0.5, 2.0),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'oscillating': MoodDescriptor(
            name='oscillating',
            description='Regular back-and-forth motion with periodic variations',
            energy_range=(0.05, 0.18),
            brightness_range=(1200, 3500),
            roughness_range=(0.03, 0.15),
            onset_density_range=(2.0, 5.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1300, 2800)
        ),
        
        'spiraling': MoodDescriptor(
            name='spiraling',
            description='Circular, twisting textures that wind through frequency space',
            energy_range=(0.06, 0.16),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.05, 0.2),
            onset_density_range=(1.8, 4.5),
            duration_threshold=5.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'cascading': MoodDescriptor(
            name='cascading',
            description='Waterfall-like textures with descending energy flows',
            energy_range=(0.08, 0.2),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.1, 0.3),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'surging': MoodDescriptor(
            name='surging',
            description='Powerful forward motion with building energy waves',
            energy_range=(0.1, float('inf')),
            brightness_range=(1800, float('inf')),
            roughness_range=(0.05, 0.18),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'rippling': MoodDescriptor(
            name='rippling',
            description='Gentle wave textures with soft, expanding patterns',
            energy_range=(0.02, 0.08),
            brightness_range=(1200, 2800),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.8, 2.5),
            duration_threshold=6.0,
            spectral_bandwidth_range=(1200, 2400)
        ),
        
        'churning': MoodDescriptor(
            name='churning',
            description='Turbulent, agitated textures with chaotic energy patterns',
            energy_range=(0.08, 0.22),
            brightness_range=(1000, float('inf')),
            roughness_range=(0.15, 0.4),
            onset_density_range=(3.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'flowing': MoodDescriptor(
            name='flowing',
            description='Smooth, continuous motion with seamless transitions',
            energy_range=(0.03, 0.1),
            brightness_range=(800, 2200),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.5, 1.5),
            duration_threshold=10.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'jolting': MoodDescriptor(
            name='jolting',
            description='Sudden, shocking textures with abrupt energy changes',
            energy_range=(0.1, float('inf')),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.2, 0.5),
            onset_density_range=(4.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'drifting': MoodDescriptor(
            name='drifting',
            description='Aimless, floating textures with gentle, directionless motion',
            energy_range=(0.01, 0.04),
            brightness_range=(1000, 2500),
            roughness_range=(0.0, 0.05),
            onset_density_range=(0.0, 0.6),
            duration_threshold=20.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'accelerating': MoodDescriptor(
            name='accelerating',
            description='Speeding textures with increasing tempo and energy',
            energy_range=(0.06, 0.18),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.03, 0.15),
            onset_density_range=(2.0, 6.0),
            duration_threshold=4.0,
            spectral_bandwidth_range=(1400, float('inf'))
        ),
        
        'decelerating': MoodDescriptor(
            name='decelerating',
            description='Slowing textures with decreasing energy and tempo',
            energy_range=(0.02, 0.08),
            brightness_range=(600, 2000),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.5, 2.0),
            duration_threshold=8.0,
            spectral_bandwidth_range=(900, 1800)
        ),
        
        'pendular': MoodDescriptor(
            name='pendular',
            description='Swinging textures with regular back-and-forth motion',
            energy_range=(0.04, 0.12),
            brightness_range=(1000, 2800),
            roughness_range=(0.02, 0.1),
            onset_density_range=(1.0, 3.0),
            duration_threshold=6.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'gyrating': MoodDescriptor(
            name='gyrating',
            description='Rotating, spinning textures with circular energy patterns',
            energy_range=(0.08, 0.2),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.08, 0.25),
            onset_density_range=(2.5, 5.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        # Spatial and dimensional (10 descriptors)
        'cavernous': MoodDescriptor(
            name='cavernous',
            description='Deep, hollow textures with vast, echoing spaces',
            energy_range=(0.02, 0.08),
            brightness_range=(400, 1200),
            roughness_range=(0.05, 0.2),
            onset_density_range=(0.2, 1.5),
            duration_threshold=15.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'intimate': MoodDescriptor(
            name='intimate',
            description='Close, personal textures with warm, enveloping presence',
            energy_range=(0.025, 0.07),
            brightness_range=(800, 1800),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.3, 1.2),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1000, 1800)
        ),
        
        'expansive': MoodDescriptor(
            name='expansive',
            description='Wide, broad textures that fill large sonic spaces',
            energy_range=(0.05, 0.15),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.5, 2.5),
            duration_threshold=12.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'compressed': MoodDescriptor(
            name='compressed',
            description='Tight, constrained textures with limited dynamic range',
            energy_range=(0.06, 0.12),
            brightness_range=(1200, 2500),
            roughness_range=(0.0, 0.08),
            onset_density_range=(1.5, 4.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'panoramic': MoodDescriptor(
            name='panoramic',
            description='Wide-angle textures with sweeping, cinematic scope',
            energy_range=(0.04, 0.12),
            brightness_range=(1000, float('inf')),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.5, 2.0),
            duration_threshold=20.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'claustrophobic': MoodDescriptor(
            name='claustrophobic',
            description='Oppressive, confined textures with restricted frequency range',
            energy_range=(0.08, 0.18),
            brightness_range=(1000, 2000),
            roughness_range=(0.1, 0.3),
            onset_density_range=(2.0, 5.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(600, 1200)
        ),
        
        'dimensional': MoodDescriptor(
            name='dimensional',
            description='Multi-layered textures with distinct spatial planes',
            energy_range=(0.06, 0.16),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.05, 0.18),
            onset_density_range=(1.5, 4.0),
            duration_threshold=8.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'microscopic': MoodDescriptor(
            name='microscopic',
            description='Tiny, detailed textures that reveal intricate sonic structures',
            energy_range=(0.01, 0.06),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.15, 0.4),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'telescopic': MoodDescriptor(
            name='telescopic',
            description='Far-reaching textures that extend into sonic distance',
            energy_range=(0.02, 0.08),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.1, 1.0),
            duration_threshold=25.0,
            spectral_bandwidth_range=(1200, 2500)
        ),
        
        'fractal': MoodDescriptor(
            name='fractal',
            description='Self-similar textures with recursive patterns at multiple scales',
            energy_range=(0.04, 0.14),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.1, 0.35),
            onset_density_range=(2.0, 6.0),
            duration_threshold=10.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        # Temporal and rhythmic (15 descriptors)
        'syncopated': MoodDescriptor(
            name='syncopated',
            description='Off-beat textures with displaced rhythmic emphasis',
            energy_range=(0.06, 0.18),
            brightness_range=(1200, float('inf')),
            roughness_range=(0.05, 0.2),
            onset_density_range=(2.5, 6.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1300, 2800)
        ),
        
        'polyrhythmic': MoodDescriptor(
            name='polyrhythmic',
            description='Multiple interlocking rhythms creating complex temporal patterns',
            energy_range=(0.08, 0.2),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.1, 0.3),
            onset_density_range=(4.0, float('inf')),
            duration_threshold=5.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'rubato': MoodDescriptor(
            name='rubato',
            description='Flexible timing with expressive tempo variations',
            energy_range=(0.03, 0.12),
            brightness_range=(800, 2500),
            roughness_range=(0.02, 0.15),
            onset_density_range=(0.5, 2.5),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1000, 2200)
        ),
        
        'staccato': MoodDescriptor(
            name='staccato',
            description='Short, detached textures with crisp articulation',
            energy_range=(0.08, 0.2),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.1, 0.3),
            onset_density_range=(4.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1500, float('inf'))
        ),
        
        'legato': MoodDescriptor(
            name='legato',
            description='Smooth, connected textures with flowing continuity',
            energy_range=(0.03, 0.1),
            brightness_range=(600, 2000),
            roughness_range=(0.0, 0.06),
            onset_density_range=(0.3, 1.5),
            duration_threshold=10.0,
            spectral_bandwidth_range=(900, 1800)
        ),
        
        'accelerando': MoodDescriptor(
            name='accelerando',
            description='Gradually speeding textures with building momentum',
            energy_range=(0.05, 0.18),
            brightness_range=(1200, float('inf')),
            roughness_range=(0.03, 0.15),
            onset_density_range=(1.5, 5.0),
            duration_threshold=6.0,
            spectral_bandwidth_range=(1300, float('inf'))
        ),
        
        'ritardando': MoodDescriptor(
            name='ritardando',
            description='Gradually slowing textures with decreasing energy',
            energy_range=(0.02, 0.08),
            brightness_range=(600, 1800),
            roughness_range=(0.01, 0.1),
            onset_density_range=(0.5, 2.0),
            duration_threshold=10.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'tremolo': MoodDescriptor(
            name='tremolo',
            description='Rapid amplitude fluctuations creating shimmering effects',
            energy_range=(0.04, 0.14),
            brightness_range=(1000, 3000),
            roughness_range=(0.08, 0.25),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1200, 2500)
        ),
        
        'vibrato': MoodDescriptor(
            name='vibrato',
            description='Pitch variations creating warm, expressive modulation',
            energy_range=(0.04, 0.12),
            brightness_range=(800, 2200),
            roughness_range=(0.05, 0.18),
            onset_density_range=(1.5, 4.0),
            duration_threshold=4.0,
            spectral_bandwidth_range=(1100, 2200)
        ),
        
        'sforzando': MoodDescriptor(
            name='sforzando',
            description='Sudden emphasis with explosive energy bursts',
            energy_range=(0.15, float('inf')),
            brightness_range=(2000, float('inf')),
            roughness_range=(0.1, 0.4),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'tenuto': MoodDescriptor(
            name='tenuto',
            description='Held, sustained textures with full value and weight',
            energy_range=(0.04, 0.12),
            brightness_range=(600, 2000),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.2, 1.2),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'marcato': MoodDescriptor(
            name='marcato',
            description='Strongly marked textures with emphatic articulation',
            energy_range=(0.1, 0.22),
            brightness_range=(1800, float('inf')),
            roughness_range=(0.08, 0.25),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'fermata': MoodDescriptor(
            name='fermata',
            description='Extended, held textures that suspend time',
            energy_range=(0.02, 0.1),
            brightness_range=(800, 2500),
            roughness_range=(0.0, 0.1),
            onset_density_range=(0.0, 0.5),
            duration_threshold=30.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'ostinato': MoodDescriptor(
            name='ostinato',
            description='Persistently repeated patterns with hypnotic repetition',
            energy_range=(0.05, 0.15),
            brightness_range=(1000, 2800),
            roughness_range=(0.02, 0.12),
            onset_density_range=(2.0, 5.0),
            duration_threshold=15.0,
            spectral_bandwidth_range=(1200, 2400)
        ),
        
        'hemiola': MoodDescriptor(
            name='hemiola',
            description='Conflicting rhythmic groupings creating temporal ambiguity',
            energy_range=(0.06, 0.16),
            brightness_range=(1300, 3000),
            roughness_range=(0.08, 0.22),
            onset_density_range=(3.0, 6.5),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1400, 2800)
        ),
        
        # Climatic and environmental (15 descriptors)
        'tempestuous': MoodDescriptor(
            name='tempestuous',
            description='Storm-like textures with chaotic, violent energy',
            energy_range=(0.15, float('inf')),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.2, 0.5),
            onset_density_range=(4.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(2000, float('inf'))
        ),
        
        'arctic': MoodDescriptor(
            name='arctic',
            description='Freezing, crystalline textures with sparse, icy characteristics',
            energy_range=(0.0, 0.03),
            brightness_range=(3000, float('inf')),
            roughness_range=(0.0, 0.05),
            onset_density_range=(0.0, 0.3),
            duration_threshold=25.0,
            spectral_bandwidth_range=(0.0, 1000)
        ),
        
        'tropical': MoodDescriptor(
            name='tropical',
            description='Lush, humid textures with rich, vibrant characteristics',
            energy_range=(0.06, 0.16),
            brightness_range=(1200, 2800),
            roughness_range=(0.08, 0.22),
            onset_density_range=(1.5, 4.0),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'desert': MoodDescriptor(
            name='desert',
            description='Dry, sparse textures with vast, empty spaces',
            energy_range=(0.01, 0.05),
            brightness_range=(1500, 3500),
            roughness_range=(0.02, 0.1),
            onset_density_range=(0.0, 0.8),
            duration_threshold=30.0,
            spectral_bandwidth_range=(800, 1600)
        ),
        
        'oceanic': MoodDescriptor(
            name='oceanic',
            description='Deep, flowing textures with tidal rhythms and aquatic motion',
            energy_range=(0.03, 0.12),
            brightness_range=(600, 1800),
            roughness_range=(0.05, 0.2),
            onset_density_range=(0.5, 2.5),
            duration_threshold=20.0,
            spectral_bandwidth_range=(1400, 2800)
        ),
        
        'volcanic': MoodDescriptor(
            name='volcanic',
            description='Explosive, molten textures with intense heat and pressure',
            energy_range=(0.12, float('inf')),
            brightness_range=(1000, 2500),
            roughness_range=(0.15, 0.4),
            onset_density_range=(3.0, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'glacial': MoodDescriptor(
            name='glacial',
            description='Slowly moving, massive textures with ancient, frozen quality',
            energy_range=(0.01, 0.06),
            brightness_range=(800, 2000),
            roughness_range=(0.0, 0.08),
            onset_density_range=(0.0, 0.5),
            duration_threshold=40.0,
            spectral_bandwidth_range=(1200, 2400)
        ),
        
        'pastoral': MoodDescriptor(
            name='pastoral',
            description='Peaceful, countryside textures with natural, rustic charm',
            energy_range=(0.03, 0.08),
            brightness_range=(800, 2200),
            roughness_range=(0.04, 0.16),
            onset_density_range=(0.5, 2.0),
            duration_threshold=12.0,
            spectral_bandwidth_range=(1200, 2200)
        ),
        
        'urban': MoodDescriptor(
            name='urban',
            description='City-like textures with mechanical, industrial characteristics',
            energy_range=(0.08, 0.2),
            brightness_range=(1500, float('inf')),
            roughness_range=(0.1, 0.3),
            onset_density_range=(2.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'subterranean': MoodDescriptor(
            name='subterranean',
            description='Underground textures with deep, earthy resonance',
            energy_range=(0.02, 0.08),
            brightness_range=(200, 800),
            roughness_range=(0.05, 0.2),
            onset_density_range=(0.2, 1.5),
            duration_threshold=20.0,
            spectral_bandwidth_range=(1000, 2000)
        ),
        
        'celestial_storms': MoodDescriptor(
            name='celestial_storms',
            description='Cosmic tempests with otherworldly electrical activity',
            energy_range=(0.1, float('inf')),
            brightness_range=(3000, float('inf')),
            roughness_range=(0.15, 0.4),
            onset_density_range=(3.5, float('inf')),
            duration_threshold=0.0,
            spectral_bandwidth_range=(2200, float('inf'))
        ),
        
        'aurora': MoodDescriptor(
            name='aurora',
            description='Northern lights textures with shimmering, dancing colors',
            energy_range=(0.02, 0.08),
            brightness_range=(2500, float('inf')),
            roughness_range=(0.08, 0.25),
            onset_density_range=(1.5, 4.0),
            duration_threshold=15.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'mirage': MoodDescriptor(
            name='mirage',
            description='Illusory textures that shimmer and waver like heat distortion',
            energy_range=(0.02, 0.06),
            brightness_range=(2000, 3500),
            roughness_range=(0.1, 0.3),
            onset_density_range=(2.0, 5.0),
            duration_threshold=8.0,
            spectral_bandwidth_range=(1600, float('inf'))
        ),
        
        'monsoon': MoodDescriptor(
            name='monsoon',
            description='Heavy, rhythmic textures like seasonal rain patterns',
            energy_range=(0.08, 0.18),
            brightness_range=(1000, 2500),
            roughness_range=(0.15, 0.35),
            onset_density_range=(3.0, 6.0),
            duration_threshold=10.0,
            spectral_bandwidth_range=(1800, float('inf'))
        ),
        
        'eclipse': MoodDescriptor(
            name='eclipse',
            description='Shadow-casting textures with gradual darkening and revelation',
            energy_range=(0.03, 0.1),
            brightness_range=(600, 1800),
            roughness_range=(0.02, 0.12),
            onset_density_range=(0.5, 2.0),
            duration_threshold=20.0,
            spectral_bandwidth_range=(1000, 2200)
        )
    }
    
    @classmethod
    def get_all_descriptors(cls) -> Dict[str, MoodDescriptor]:
        """Get all mood descriptors combined."""
        all_descriptors = {}
        all_descriptors.update(cls.CORE_DESCRIPTORS)
        all_descriptors.update(cls.EXTENDED_DESCRIPTORS)
        all_descriptors.update(cls.ADVANCED_DESCRIPTORS)
        return all_descriptors
    
    @classmethod
    def get_descriptor_names(cls) -> List[str]:
        """Get list of all descriptor names."""
        return list(cls.get_all_descriptors().keys())
    
    @classmethod
    def get_core_descriptor_names(cls) -> List[str]:
        """Get list of core descriptor names."""
        return list(cls.CORE_DESCRIPTORS.keys())
    
    @classmethod
    def get_extended_descriptor_names(cls) -> List[str]:
        """Get list of extended descriptor names."""
        return list(cls.EXTENDED_DESCRIPTORS.keys())
    
    @classmethod
    def get_advanced_descriptor_names(cls) -> List[str]:
        """Get list of advanced descriptor names."""
        return list(cls.ADVANCED_DESCRIPTORS.keys())


class CharacterTags:
    """
    Complete definition of all character tags for synthesizer music.
    
    This class contains the comprehensive set of character tags
    used to identify different synthesizer types and sound sources.
    Each tag is defined with specific technical characteristics.
    
    The character system recognizes 59 distinct tags organized into:
    - Synthesis types: Specific synthesizer and instrument categories (25 tags)
    - Texture types: Surface characteristics and material qualities (20 tags)
    - Processing types: Effects and signal processing characteristics (14 tags)
    """
    
    # Synthesis type tags (25 total)
    SYNTHESIS_TAGS = {
        'analog_synth': CharacterTag(
            name='analog_synth',
            description='Analog synthesizer with warm, organic characteristics',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, 1000),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, 0.1),
            mfcc_characteristics={'warmth': 'high', 'precision': 'low'}
        ),
        
        'digital_synth': CharacterTag(
            name='digital_synth',
            description='Digital synthesizer with precise, clean characteristics',
            spectral_centroid_range=(2000, float('inf')),
            spectral_bandwidth_range=(0.0, 800),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'precision': 'high', 'warmth': 'low'}
        ),
        
        'mellotron': CharacterTag(
            name='mellotron',
            description='Mellotron/sampled instrument with irregular spectral content',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1200, 2000),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.05, 0.15),
            mfcc_characteristics={'organic': 'high', 'irregularity': 'high'}
        ),
        
        'percussive_instrument': CharacterTag(
            name='percussive_instrument',
            description='Percussive instrument with bright, transient characteristics',
            spectral_centroid_range=(2500, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(4000, float('inf')),
            zero_crossing_rate_range=(0.1, float('inf')),
            mfcc_characteristics={'attack': 'sharp', 'sustain': 'low'}
        ),
        
        'acoustic_instrument': CharacterTag(
            name='acoustic_instrument',
            description='Acoustic instrument with rich, complex spectral content',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.12, float('inf')),
            mfcc_characteristics={'complexity': 'high', 'naturalness': 'high'}
        ),
        
        'fm_synth': CharacterTag(
            name='fm_synth',
            description='FM synthesizer with complex harmonic interactions',
            spectral_centroid_range=(1500, 3000),
            spectral_bandwidth_range=(1000, 2500),
            spectral_rolloff_range=(3000, float('inf')),
            zero_crossing_rate_range=(0.08, 0.25),
            mfcc_characteristics={'harmonics': 'complex', 'modulation': 'high'}
        ),
        
        'am_synth': CharacterTag(
            name='am_synth',
            description='Amplitude modulation synthesizer with tremolo characteristics',
            spectral_centroid_range=(800, 2500),
            spectral_bandwidth_range=(600, 1800),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.05, 0.2),
            mfcc_characteristics={'modulation': 'amplitude', 'tremolo': 'present'}
        ),
        
        'granular_synth': CharacterTag(
            name='granular_synth',
            description='Granular synthesizer with fragmented, particulate texture',
            spectral_centroid_range=(1000, float('inf')),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.2, float('inf')),
            mfcc_characteristics={'fragmentation': 'high', 'grain_density': 'variable'}
        ),
        
        'wavetable_synth': CharacterTag(
            name='wavetable_synth',
            description='Wavetable synthesizer with morphing spectral evolution',
            spectral_centroid_range=(1200, 4000),
            spectral_bandwidth_range=(1000, 3000),
            spectral_rolloff_range=(2500, float('inf')),
            zero_crossing_rate_range=(0.06, 0.3),
            mfcc_characteristics={'evolution': 'dynamic', 'morphing': 'present'}
        ),
        
        'physical_modeling': CharacterTag(
            name='physical_modeling',
            description='Physical modeling synthesis with realistic acoustic behavior',
            spectral_centroid_range=(600, 3500),
            spectral_bandwidth_range=(1200, 2800),
            spectral_rolloff_range=(1800, 5000),
            zero_crossing_rate_range=(0.08, 0.18),
            mfcc_characteristics={'realism': 'high', 'physics': 'modeled'}
        ),
        
        'drum_machine': CharacterTag(
            name='drum_machine',
            description='Drum machine with quantized rhythmic patterns',
            spectral_centroid_range=(1500, float('inf')),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(3500, float('inf')),
            zero_crossing_rate_range=(0.15, float('inf')),
            mfcc_characteristics={'rhythm': 'quantized', 'percussion': 'electronic'}
        ),
        
        'sampler': CharacterTag(
            name='sampler',
            description='Digital sampler with variable source characteristics',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(800, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'source': 'variable', 'sampling': 'digital'}
        ),
        
        'subtractive_synth': CharacterTag(
            name='subtractive_synth',
            description='Subtractive synthesizer with filtered spectral shaping',
            spectral_centroid_range=(800, 2800),
            spectral_bandwidth_range=(600, 2000),
            spectral_rolloff_range=(1500, 4000),
            zero_crossing_rate_range=(0.0, 0.15),
            mfcc_characteristics={'filtering': 'subtractive', 'resonance': 'variable'}
        ),
        
        'additive_synth': CharacterTag(
            name='additive_synth',
            description='Additive synthesizer with precise harmonic control',
            spectral_centroid_range=(1000, 4000),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(2000, float('inf')),
            zero_crossing_rate_range=(0.02, 0.12),
            mfcc_characteristics={'harmonics': 'controlled', 'precision': 'very_high'}
        ),
        
        'modular_synth': CharacterTag(
            name='modular_synth',
            description='Modular synthesizer with experimental characteristics',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'unpredictability': 'high', 'modulation': 'complex'}
        ),
        
        'string_machine': CharacterTag(
            name='string_machine',
            description='String synthesizer with ensemble characteristics',
            spectral_centroid_range=(800, 2200),
            spectral_bandwidth_range=(1200, 2400),
            spectral_rolloff_range=(2000, 4500),
            zero_crossing_rate_range=(0.03, 0.12),
            mfcc_characteristics={'ensemble': 'string', 'chorus': 'built_in'}
        ),
        
        'organ': CharacterTag(
            name='organ',
            description='Organ with harmonic drawbar characteristics',
            spectral_centroid_range=(400, 1800),
            spectral_bandwidth_range=(800, 2000),
            spectral_rolloff_range=(1200, 3000),
            zero_crossing_rate_range=(0.0, 0.08),
            mfcc_characteristics={'harmonics': 'drawbar', 'sustain': 'infinite'}
        ),
        
        'piano': CharacterTag(
            name='piano',
            description='Piano with attack-decay envelope and rich harmonics',
            spectral_centroid_range=(600, 2500),
            spectral_bandwidth_range=(1500, 3500),
            spectral_rolloff_range=(2000, 5000),
            zero_crossing_rate_range=(0.08, 0.2),
            mfcc_characteristics={'attack': 'percussive', 'decay': 'natural'}
        ),
        
        'choir': CharacterTag(
            name='choir',
            description='Vocal ensemble with formant characteristics',
            spectral_centroid_range=(600, 2000),
            spectral_bandwidth_range=(1000, 2200),
            spectral_rolloff_range=(1800, 3500),
            zero_crossing_rate_range=(0.05, 0.15),
            mfcc_characteristics={'formants': 'vocal', 'ensemble': 'human'}
        ),
        
        'brass': CharacterTag(
            name='brass',
            description='Brass instruments with bright, dynamic harmonics',
            spectral_centroid_range=(1200, 3500),
            spectral_bandwidth_range=(1300, 3000),
            spectral_rolloff_range=(3000, float('inf')),
            zero_crossing_rate_range=(0.08, 0.2),
            mfcc_characteristics={'brightness': 'dynamic', 'expression': 'high'}
        ),
        
        'woodwind': CharacterTag(
            name='woodwind',
            description='Woodwind instruments with breathy characteristics',
            spectral_centroid_range=(800, 2800),
            spectral_bandwidth_range=(1100, 2400),
            spectral_rolloff_range=(2200, 4500),
            zero_crossing_rate_range=(0.1, 0.25),
            mfcc_characteristics={'breathiness': 'present', 'reed': 'characteristic'}
        ),
        
        'pad_synth': CharacterTag(
            name='pad_synth',
            description='Pad synthesizer with sustained, evolving textures',
            spectral_centroid_range=(600, 2200),
            spectral_bandwidth_range=(1200, 2800),
            spectral_rolloff_range=(1800, 4000),
            zero_crossing_rate_range=(0.0, 0.1),
            mfcc_characteristics={'evolution': 'slow', 'sustain': 'long'}
        ),
        
        'lead_synth': CharacterTag(
            name='lead_synth',
            description='Lead synthesizer with prominent melodic characteristics',
            spectral_centroid_range=(1500, 4000),
            spectral_bandwidth_range=(800, 2200),
            spectral_rolloff_range=(2500, float('inf')),
            zero_crossing_rate_range=(0.02, 0.15),
            mfcc_characteristics={'prominence': 'high', 'melodic': 'focused'}
        ),
        
        'bass_synth': CharacterTag(
            name='bass_synth',
            description='Bass synthesizer with low-frequency emphasis',
            spectral_centroid_range=(0.0, 800),
            spectral_bandwidth_range=(200, 1200),
            spectral_rolloff_range=(400, 1500),
            zero_crossing_rate_range=(0.0, 0.08),
            mfcc_characteristics={'frequency': 'low', 'fundamental': 'strong'}
        ),
        
        'arp_synth': CharacterTag(
            name='arp_synth',
            description='Arpeggiator synthesizer with repetitive melodic patterns',
            spectral_centroid_range=(1200, 3500),
            spectral_bandwidth_range=(800, 2000),
            spectral_rolloff_range=(2000, 5000),
            zero_crossing_rate_range=(0.05, 0.2),
            mfcc_characteristics={'pattern': 'repetitive', 'timing': 'precise'}
        ),

        # ── New content-type tags ──────────────────────────────────────────────

        'spoken_word': CharacterTag(
            name='spoken_word',
            description='Spoken voice content — lectures, poetry, narration',
            spectral_centroid_range=(200, 3800),
            spectral_bandwidth_range=(800, 4000),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.04, 0.28),
            mfcc_characteristics='neutral',
            spectral_flatness_range=(0.05, 0.65),
            confidence_threshold=0.55,
        ),

        'guitar': CharacterTag(
            name='guitar',
            description='Guitar (acoustic or electric) with transient-rich attack',
            spectral_centroid_range=(500, 4000),
            spectral_bandwidth_range=(1000, 3500),
            spectral_rolloff_range=(1500, float('inf')),
            zero_crossing_rate_range=(0.04, 0.25),
            mfcc_characteristics='neutral',
            spectral_flatness_range=(0.05, 0.35),
            confidence_threshold=0.55,
        ),

        'live_drums': CharacterTag(
            name='live_drums',
            description='Acoustic drum kit or live percussion with noisy transients',
            spectral_centroid_range=(1000, float('inf')),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(2500, float('inf')),
            zero_crossing_rate_range=(0.10, float('inf')),
            mfcc_characteristics='neutral',
            spectral_flatness_range=(0.10, 0.70),
            spectral_flux_range=(50.0, float('inf')),
            confidence_threshold=0.60,
        ),

        'laughter': CharacterTag(
            name='laughter',
            description='Human laughter — bursty, breathy, voiced/unvoiced mix',
            spectral_centroid_range=(600, 3500),
            spectral_bandwidth_range=(1000, 4000),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.10, 0.40),
            mfcc_characteristics='neutral',
            spectral_flatness_range=(0.15, 0.70),
            confidence_threshold=0.60,
        ),

        'wide_stereo': CharacterTag(
            name='wide_stereo',
            description='Meaningfully wide stereo field — L/R divergence > 0.25',
            # These four ranges are fully open so they never subtract from the score.
            # The only gate is stereo_width_range, which carries all the weight.
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics='neutral',
            stereo_width_range=(0.25, float('inf')),
            confidence_threshold=0.80,
        ),
    }
    
    # Texture tags (20 total)
    TEXTURE_TAGS = {
        'rich_texture': CharacterTag(
            name='rich_texture',
            description='Complex, layered texture with wide spectral bandwidth',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(2200, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'complexity': 'high', 'richness': 'high'}
        ),
        
        'pure_tone': CharacterTag(
            name='pure_tone',
            description='Simple, clean texture with narrow spectral bandwidth',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, 600),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'simplicity': 'high', 'purity': 'high'}
        ),
        
        'bright_harmonics': CharacterTag(
            name='bright_harmonics',
            description='Bright harmonic content with high spectral rolloff',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(5000, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'brightness': 'high', 'harmonics': 'upper'}
        ),
        
        'warm_harmonics': CharacterTag(
            name='warm_harmonics',
            description='Warm harmonic content with low spectral rolloff',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, 2000),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'warmth': 'high', 'harmonics': 'lower'}
        ),
        
        'smooth_texture': CharacterTag(
            name='smooth_texture',
            description='Silky smooth texture with minimal roughness',
            spectral_centroid_range=(400, 2500),
            spectral_bandwidth_range=(600, 1800),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, 0.05),
            mfcc_characteristics={'smoothness': 'high', 'roughness': 'minimal'}
        ),
        
        'rough_texture': CharacterTag(
            name='rough_texture',
            description='Gritty, coarse texture with high roughness',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.15, float('inf')),
            mfcc_characteristics={'roughness': 'high', 'grittiness': 'present'}
        ),
        
        'crystalline_texture': CharacterTag(
            name='crystalline_texture',
            description='Glass-like, precise texture with sharp definition',
            spectral_centroid_range=(2500, float('inf')),
            spectral_bandwidth_range=(0.0, 1000),
            spectral_rolloff_range=(4000, float('inf')),
            zero_crossing_rate_range=(0.0, 0.06),
            mfcc_characteristics={'precision': 'crystalline', 'definition': 'sharp'}
        ),
        
        'organic_texture': CharacterTag(
            name='organic_texture',
            description='Natural, breathing texture with irregular variations',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1200, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.08, 0.2),
            mfcc_characteristics={'naturalness': 'high', 'variation': 'irregular'}
        ),
        
        'mechanical_texture': CharacterTag(
            name='mechanical_texture',
            description='Machine-like texture with precise, industrial character',
            spectral_centroid_range=(1500, 4000),
            spectral_bandwidth_range=(1000, 2500),
            spectral_rolloff_range=(3000, float('inf')),
            zero_crossing_rate_range=(0.1, 0.3),
            mfcc_characteristics={'precision': 'mechanical', 'character': 'industrial'}
        ),
        
        'liquid_texture': CharacterTag(
            name='liquid_texture',
            description='Flowing, fluid texture with smooth modulation',
            spectral_centroid_range=(600, 2200),
            spectral_bandwidth_range=(800, 2000),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.02, 0.12),
            mfcc_characteristics={'flow': 'smooth', 'modulation': 'fluid'}
        ),
        
        'gaseous_texture': CharacterTag(
            name='gaseous_texture',
            description='Ethereal, cloud-like texture with diffuse characteristics',
            spectral_centroid_range=(1200, 3500),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.05, 0.18),
            mfcc_characteristics={'diffusion': 'high', 'density': 'variable'}
        ),
        
        'metallic_texture': CharacterTag(
            name='metallic_texture',
            description='Cold, resonant texture with industrial character',
            spectral_centroid_range=(2000, float('inf')),
            spectral_bandwidth_range=(1200, 2800),
            spectral_rolloff_range=(3500, float('inf')),
            zero_crossing_rate_range=(0.08, 0.25),
            mfcc_characteristics={'resonance': 'metallic', 'coldness': 'present'}
        ),
        
        'wooden_texture': CharacterTag(
            name='wooden_texture',
            description='Warm, natural texture with organic resonance',
            spectral_centroid_range=(400, 1600),
            spectral_bandwidth_range=(800, 2200),
            spectral_rolloff_range=(1200, 3000),
            zero_crossing_rate_range=(0.05, 0.15),
            mfcc_characteristics={'warmth': 'natural', 'resonance': 'wooden'}
        ),
        
        'glassy_texture': CharacterTag(
            name='glassy_texture',
            description='Brittle, transparent texture with fragile character',
            spectral_centroid_range=(3000, float('inf')),
            spectral_bandwidth_range=(400, 1200),
            spectral_rolloff_range=(5000, float('inf')),
            zero_crossing_rate_range=(0.0, 0.08),
            mfcc_characteristics={'transparency': 'high', 'fragility': 'present'}
        ),
        
        'fabric_texture': CharacterTag(
            name='fabric_texture',
            description='Soft, woven texture with gentle characteristics',
            spectral_centroid_range=(600, 1800),
            spectral_bandwidth_range=(1000, 2200),
            spectral_rolloff_range=(1500, 3500),
            zero_crossing_rate_range=(0.06, 0.16),
            mfcc_characteristics={'softness': 'high', 'weaving': 'present'}
        ),
        
        'sandy_texture': CharacterTag(
            name='sandy_texture',
            description='Granular, particulate texture with gritty character',
            spectral_centroid_range=(1200, float('inf')),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.2, float('inf')),
            mfcc_characteristics={'granularity': 'high', 'particles': 'present'}
        ),
        
        'waxy_texture': CharacterTag(
            name='waxy_texture',
            description='Smooth but impure texture with slight irregularities',
            spectral_centroid_range=(800, 2000),
            spectral_bandwidth_range=(800, 1600),
            spectral_rolloff_range=(1800, 3500),
            zero_crossing_rate_range=(0.03, 0.1),
            mfcc_characteristics={'smoothness': 'moderate', 'purity': 'low'}
        ),
        
        'rubbery_texture': CharacterTag(
            name='rubbery_texture',
            description='Elastic, bouncy texture with flexible response',
            spectral_centroid_range=(1000, 2800),
            spectral_bandwidth_range=(1200, 2400),
            spectral_rolloff_range=(2000, 4000),
            zero_crossing_rate_range=(0.08, 0.2),
            mfcc_characteristics={'elasticity': 'high', 'bounce': 'present'}
        ),
        
        'ceramic_texture': CharacterTag(
            name='ceramic_texture',
            description='Hard, resonant texture with precise characteristics',
            spectral_centroid_range=(1800, 3500),
            spectral_bandwidth_range=(600, 1400),
            spectral_rolloff_range=(3000, 5000),
            zero_crossing_rate_range=(0.02, 0.1),
            mfcc_characteristics={'hardness': 'high', 'resonance': 'precise'}
        ),
        
        'plastic_texture': CharacterTag(
            name='plastic_texture',
            description='Artificial, uniform texture with synthetic character',
            spectral_centroid_range=(1200, 2800),
            spectral_bandwidth_range=(600, 1200),
            spectral_rolloff_range=(2200, 4000),
            zero_crossing_rate_range=(0.0, 0.08),
            mfcc_characteristics={'artificiality': 'high', 'uniformity': 'present'}
        )
    }
    
    # Processing type tags (14 total)
    PROCESSING_TAGS = {
        'reverbed': CharacterTag(
            name='reverbed',
            description='Spacious texture with echo and reverberation',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1200, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'space': 'large', 'decay': 'extended'}
        ),
        
        'delayed': CharacterTag(
            name='delayed',
            description='Texture with distinct echo patterns and rhythmic repeats',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'echo': 'distinct', 'rhythm': 'delayed'}
        ),
        
        'chorused': CharacterTag(
            name='chorused',
            description='Thick, ensemble-like texture with modulation',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1400, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.05, 0.2),
            mfcc_characteristics={'thickness': 'high', 'modulation': 'chorus'}
        ),
        
        'flanged': CharacterTag(
            name='flanged',
            description='Sweeping, jet-like texture with comb filtering',
            spectral_centroid_range=(1000, float('inf')),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.1, 0.3),
            mfcc_characteristics={'sweep': 'present', 'comb': 'filtering'}
        ),
        
        'phased': CharacterTag(
            name='phased',
            description='Notched texture with frequency sweeping',
            spectral_centroid_range=(800, float('inf')),
            spectral_bandwidth_range=(1200, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.08, 0.25),
            mfcc_characteristics={'notches': 'moving', 'phase': 'shifting'}
        ),
        
        'distorted': CharacterTag(
            name='distorted',
            description='Saturated texture with harmonic distortion',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.12, float('inf')),
            mfcc_characteristics={'saturation': 'present', 'harmonics': 'added'}
        ),
        
        'compressed': CharacterTag(
            name='compressed',
            description='Controlled dynamics with consistent level',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'dynamics': 'controlled', 'level': 'consistent'}
        ),
        
        'filtered': CharacterTag(
            name='filtered',
            description='Frequency-shaped texture with resonant characteristics',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(0.0, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'frequency': 'shaped', 'resonance': 'variable'}
        ),
        
        'pitched': CharacterTag(
            name='pitched',
            description='Frequency-shifted texture with harmonization',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1000, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'frequency': 'shifted', 'harmony': 'added'}
        ),
        
        'ring_modulated': CharacterTag(
            name='ring_modulated',
            description='Metallic texture with bell-like inharmonic content',
            spectral_centroid_range=(1500, float('inf')),
            spectral_bandwidth_range=(1800, float('inf')),
            spectral_rolloff_range=(3000, float('inf')),
            zero_crossing_rate_range=(0.15, float('inf')),
            mfcc_characteristics={'inharmonics': 'present', 'metallic': 'bell_like'}
        ),
        
        'bit_crushed': CharacterTag(
            name='bit_crushed',
            description='Digital degradation with lo-fi character',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1200, float('inf')),
            spectral_rolloff_range=(0.0, 8000),
            zero_crossing_rate_range=(0.1, float('inf')),
            mfcc_characteristics={'degradation': 'digital', 'fidelity': 'low'}
        ),
        
        'tape_saturated': CharacterTag(
            name='tape_saturated',
            description='Vintage warmth with subtle compression',
            spectral_centroid_range=(400, 2500),
            spectral_bandwidth_range=(800, 2200),
            spectral_rolloff_range=(0.0, 6000),
            zero_crossing_rate_range=(0.0, 0.12),
            mfcc_characteristics={'warmth': 'vintage', 'compression': 'tape'}
        ),
        
        'tube_saturated': CharacterTag(
            name='tube_saturated',
            description='Warm harmonic distortion from tube saturation',
            spectral_centroid_range=(600, 3000),
            spectral_bandwidth_range=(1000, 2400),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.02, 0.15),
            mfcc_characteristics={'warmth': 'tube', 'harmonics': 'even_order'}
        ),
        
        'multiband_processed': CharacterTag(
            name='multiband_processed',
            description='Frequency-specific processing across multiple bands',
            spectral_centroid_range=(0.0, float('inf')),
            spectral_bandwidth_range=(1500, float('inf')),
            spectral_rolloff_range=(0.0, float('inf')),
            zero_crossing_rate_range=(0.0, float('inf')),
            mfcc_characteristics={'processing': 'multiband', 'bands': 'specific'}
        )
    }
    
    @classmethod
    def get_all_tags(cls) -> Dict[str, CharacterTag]:
        """Get all character tags combined."""
        all_tags = {}
        all_tags.update(cls.SYNTHESIS_TAGS)
        all_tags.update(cls.TEXTURE_TAGS)
        all_tags.update(cls.PROCESSING_TAGS)
        return all_tags
    
    @classmethod
    def get_tag_names(cls) -> List[str]:
        """Get list of all tag names."""
        return list(cls.get_all_tags().keys())
    
    @classmethod
    def get_synthesis_tag_names(cls) -> List[str]:
        """Get list of synthesis tag names."""
        return list(cls.SYNTHESIS_TAGS.keys())
    
    @classmethod
    def get_texture_tag_names(cls) -> List[str]:
        """Get list of texture tag names."""
        return list(cls.TEXTURE_TAGS.keys())
        
    @classmethod
    def get_processing_tag_names(cls) -> List[str]:
        """Get list of processing tag names."""
        return list(cls.PROCESSING_TAGS.keys())
    
    @classmethod
    def get_tag_descriptions(cls) -> Dict[str, str]:
        """Get descriptions for all tags."""
        return {name: tag.description for name, tag in cls.get_all_tags().items()}


# Utility functions for working with descriptors
def get_descriptor_info() -> Dict[str, Any]:
    """
    Get comprehensive information about all available descriptors.
    
    Returns:
        Dictionary with descriptor information for documentation
    """
    return {
        'mood_descriptors': {
            'core': MoodDescriptors.get_core_descriptor_names(),
            'extended': MoodDescriptors.get_extended_descriptor_names(),
            'total_count': len(MoodDescriptors.get_all_descriptors())
        },
        'character_tags': {
            'synthesis': CharacterTags.get_synthesis_tag_names(),
            'texture': CharacterTags.get_texture_tag_names(),
            'total_count': len(CharacterTags.get_all_tags())
        },
        'descriptions': {
            'moods': {name: desc.description for name, desc in MoodDescriptors.get_all_descriptors().items()},
            'characters': CharacterTags.get_tag_descriptions()
        }
    }


def validate_descriptor_thresholds() -> Dict[str, Any]:
    """
    Validate that all descriptor thresholds are properly defined.
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check mood descriptors
    for name, desc in MoodDescriptors.get_all_descriptors().items():
        if desc.energy_range[0] > desc.energy_range[1]:
            issues.append(f"Mood {name}: Invalid energy range")
        if desc.brightness_range[0] > desc.brightness_range[1]:
            issues.append(f"Mood {name}: Invalid brightness range")
    
    # Check character tags
    for name, tag in CharacterTags.get_all_tags().items():
        if tag.spectral_centroid_range[0] > tag.spectral_centroid_range[1]:
            issues.append(f"Character {name}: Invalid spectral centroid range")
        if tag.spectral_bandwidth_range[0] > tag.spectral_bandwidth_range[1]:
            issues.append(f"Character {name}: Invalid spectral bandwidth range")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_descriptors': len(MoodDescriptors.get_all_descriptors()) + len(CharacterTags.get_all_tags())
    }