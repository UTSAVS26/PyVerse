"""
Tests for the mood mapper module.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion.mood_mapper import MoodMapper


class TestMoodMapper:
    """Test cases for MoodMapper class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.mapper = MoodMapper()
    
    def test_initialization(self):
        """Test MoodMapper initialization."""
        assert hasattr(self.mapper, 'emotion_colors')
        assert hasattr(self.mapper, 'emotion_music_tags')
        assert hasattr(self.mapper, 'emotion_emojis')
        assert hasattr(self.mapper, 'emotion_descriptions')
        
        # Check that all emotions are covered
        expected_emotions = ['happy', 'sad', 'angry', 'calm', 'excited', 'tired', 'neutral']
        for emotion in expected_emotions:
            assert emotion in self.mapper.emotion_colors
            assert emotion in self.mapper.emotion_music_tags
            assert emotion in self.mapper.emotion_emojis
            assert emotion in self.mapper.emotion_descriptions
    
    def test_get_emotion_color(self):
        """Test getting emotion colors."""
        # Test with full confidence
        color = self.mapper.get_emotion_color('happy', 1.0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        
        # Test with partial confidence
        color = self.mapper.get_emotion_color('happy', 0.5)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        
        # Test unknown emotion
        color = self.mapper.get_emotion_color('unknown', 1.0)
        assert color == self.mapper.emotion_colors['neutral']
    
    def test_get_emotion_gradient(self):
        """Test getting emotion gradients."""
        gradient = self.mapper.get_emotion_gradient('happy', 5)
        
        assert isinstance(gradient, list)
        assert len(gradient) == 5
        
        for color in gradient:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_get_music_tags(self):
        """Test getting music tags."""
        tags = self.mapper.get_music_tags('happy', 3)
        
        assert isinstance(tags, list)
        assert len(tags) == 3
        
        for tag in tags:
            assert isinstance(tag, str)
            assert len(tag) > 0
        
        # Test with more tags than available
        tags = self.mapper.get_music_tags('happy', 10)
        assert len(tags) <= len(self.mapper.emotion_music_tags['happy'])
    
    def test_get_emotion_emoji(self):
        """Test getting emotion emojis."""
        emoji = self.mapper.get_emotion_emoji('happy')
        assert isinstance(emoji, str)
        assert len(emoji) > 0
        
        # Test unknown emotion
        emoji = self.mapper.get_emotion_emoji('unknown')
        assert emoji == self.mapper.emotion_emojis['neutral']
    
    def test_get_emotion_description(self):
        """Test getting emotion descriptions."""
        description = self.mapper.get_emotion_description('happy')
        assert isinstance(description, str)
        assert len(description) > 0
        
        # Test unknown emotion
        description = self.mapper.get_emotion_description('unknown')
        assert description == self.mapper.emotion_descriptions['neutral']
    
    def test_get_mood_enhancement_suggestions(self):
        """Test getting mood enhancement suggestions."""
        suggestions = self.mapper.get_mood_enhancement_suggestions('happy')
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
        
        # Test unknown emotion
        suggestions = self.mapper.get_mood_enhancement_suggestions('unknown')
        assert suggestions == self.mapper.get_mood_enhancement_suggestions('neutral')
    
    def test_get_mood_transition_suggestions(self):
        """Test getting mood transition suggestions."""
        suggestions = self.mapper.get_mood_transition_suggestions('sad', 'happy')
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
        
        # Test unknown transition
        suggestions = self.mapper.get_mood_transition_suggestions('unknown', 'happy')
        assert len(suggestions) > 0
    
    def test_get_visual_feedback_config(self):
        """Test getting visual feedback configuration."""
        config = self.mapper.get_visual_feedback_config('happy', 0.8)
        
        assert isinstance(config, dict)
        assert 'color' in config
        assert 'gradient' in config
        assert 'emoji' in config
        assert 'description' in config
        assert 'confidence' in config
        assert 'intensity' in config
        assert 'animation_speed' in config
        
        assert config['confidence'] == 0.8
        assert config['emoji'] == self.mapper.get_emotion_emoji('happy')
        assert config['description'] == self.mapper.get_emotion_description('happy')
    
    def test_get_music_recommendation_config_match(self):
        """Test getting music recommendation config with match strategy."""
        config = self.mapper.get_music_recommendation_config('happy', 'match')
        
        assert isinstance(config, dict)
        assert 'tags' in config
        assert 'description' in config
        assert 'strategy' in config
        assert 'target_emotion' in config
        
        assert config['strategy'] == 'match'
        assert config['target_emotion'] == 'happy'
        assert 'happy' in config['description'].lower()
    
    def test_get_music_recommendation_config_modulate(self):
        """Test getting music recommendation config with modulate strategy."""
        config = self.mapper.get_music_recommendation_config('sad', 'modulate')
        
        assert isinstance(config, dict)
        assert 'tags' in config
        assert 'description' in config
        assert 'strategy' in config
        assert 'target_emotion' in config
        
        assert config['strategy'] == 'modulate'
        assert config['target_emotion'] == 'happy'  # sad -> happy
        assert 'happy' in config['description'].lower()
    
    def test_color_blending(self):
        """Test color blending with confidence."""
        # Test full confidence
        full_color = self.mapper.get_emotion_color('happy', 1.0)
        assert full_color == self.mapper.emotion_colors['happy']
        
        # Test zero confidence
        zero_color = self.mapper.get_emotion_color('happy', 0.0)
        assert zero_color == self.mapper.emotion_colors['neutral']
        
        # Test partial confidence
        partial_color = self.mapper.get_emotion_color('happy', 0.5)
        assert partial_color != full_color
        assert partial_color != zero_color
    
    def test_gradient_generation(self):
        """Test gradient generation."""
        gradient = self.mapper.get_emotion_gradient('happy', 10)
        
        # Should start with neutral and end with emotion color
        assert gradient[0] == self.mapper.emotion_colors['neutral']
        assert gradient[-1] == self.mapper.emotion_colors['happy']
        
        # Should have smooth transition
        for i in range(1, len(gradient)):
            # Each step should be different from the previous
            assert gradient[i] != gradient[i-1]
    
    def test_music_tags_consistency(self):
        """Test that music tags are consistent."""
        for emotion in self.mapper.emotion_music_tags:
            tags = self.mapper.emotion_music_tags[emotion]
            assert isinstance(tags, list)
            assert len(tags) > 0
            
            for tag in tags:
                assert isinstance(tag, str)
                assert len(tag) > 0
    
    def test_emotion_descriptions_consistency(self):
        """Test that emotion descriptions are consistent."""
        for emotion in self.mapper.emotion_descriptions:
            description = self.mapper.emotion_descriptions[emotion]
            assert isinstance(description, str)
            assert len(description) > 0
            assert emotion.lower() in description.lower() or 'tone' in description.lower()
    
    def test_suggestions_consistency(self):
        """Test that suggestions are consistent."""
        for emotion in ['happy', 'sad', 'angry', 'calm', 'excited', 'tired', 'neutral']:
            suggestions = self.mapper.get_mood_enhancement_suggestions(emotion)
            assert isinstance(suggestions, list)
            assert len(suggestions) >= 2  # Should have at least 2 suggestions
            
            for suggestion in suggestions:
                assert isinstance(suggestion, str)
                assert len(suggestion) > 10  # Should be meaningful suggestions
    
    def test_transition_suggestions(self):
        """Test specific transition suggestions."""
        # Test sad to happy transition
        sad_to_happy = self.mapper.get_mood_transition_suggestions('sad', 'happy')
        assert len(sad_to_happy) > 0
        assert any('music' in suggestion.lower() for suggestion in sad_to_happy)
        
        # Test angry to calm transition
        angry_to_calm = self.mapper.get_mood_transition_suggestions('angry', 'calm')
        assert len(angry_to_calm) > 0
        assert any('breathing' in suggestion.lower() for suggestion in angry_to_calm)
    
    def test_intensity_calculation(self):
        """Test intensity calculation."""
        config = self.mapper.get_visual_feedback_config('happy', 0.8)
        
        # Intensity should be scaled confidence
        expected_intensity = min(0.8 * 1.5, 1.0)
        assert abs(config['intensity'] - expected_intensity) < 0.001
    
    def test_animation_speed_calculation(self):
        """Test animation speed calculation."""
        config = self.mapper.get_visual_feedback_config('excited', 0.9)
        
        # Animation speed should be emotion-specific
        assert config['animation_speed'] > 0
        assert config['animation_speed'] > 1.0  # Excited should be fast
    
        # Test calm emotion
        calm_config = self.mapper.get_visual_feedback_config('calm', 0.9)
        assert calm_config['animation_speed'] < config['animation_speed']
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with confidence > 1.0
        config = self.mapper.get_visual_feedback_config('happy', 1.5)
        assert config['intensity'] == 1.0
        
        # Test with confidence < 0.0
        config = self.mapper.get_visual_feedback_config('happy', -0.5)
        assert config['intensity'] == 0.0
        
        # Test with empty emotion
        config = self.mapper.get_visual_feedback_config('', 0.8)
        assert config['emoji'] == self.mapper.emotion_emojis['neutral']
    
    def test_modulation_mapping(self):
        """Test emotion modulation mapping."""
        # Test known modulation pairs
        test_cases = [
            ('sad', 'happy'),
            ('angry', 'calm'),
            ('tired', 'excited'),
            ('excited', 'calm')
        ]
        
        for from_emotion, expected_to_emotion in test_cases:
            config = self.mapper.get_music_recommendation_config(from_emotion, 'modulate')
            assert config['target_emotion'] == expected_to_emotion
        
        # Test unknown emotion
        config = self.mapper.get_music_recommendation_config('unknown', 'modulate')
        assert config['target_emotion'] == 'calm'  # Default fallback
