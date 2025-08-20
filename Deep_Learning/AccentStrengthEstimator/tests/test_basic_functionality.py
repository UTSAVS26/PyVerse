"""
Basic functionality tests that don't require external dependencies.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.audio.reference_generator import ReferenceGenerator
from src.analysis.phoneme_analyzer import PhonemeAnalyzer
from src.scoring.feedback_generator import FeedbackGenerator


class TestBasicFunctionality:
    """Test basic functionality without external dependencies."""
    
    def test_reference_generator_init(self):
        """Test ReferenceGenerator initialization."""
        generator = ReferenceGenerator()
        assert generator.sample_rate == 22050
        assert generator.reference_data == {}
    
    def test_reference_generator_phoneme_generation(self):
        """Test phoneme generation."""
        generator = ReferenceGenerator()
        
        # Test simple text
        phonemes = generator.generate_phonemes("hello")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
        
        # Test empty text
        phonemes = generator.generate_phonemes("")
        assert isinstance(phonemes, list)
    
    def test_reference_generator_difficulty_assessment(self):
        """Test difficulty assessment."""
        generator = ReferenceGenerator()
        
        # Test easy phrase
        difficulty = generator._assess_difficulty("Hi", ["h", "aɪ"])
        assert difficulty == "easy"
        
        # Test hard phrase
        difficulty = generator._assess_difficulty("The quick brown fox jumps over the lazy dog", ["ð", "ə", "kwɪk", "braʊn", "fɑks", "dʒʌmps", "oʊvər", "ðə", "leɪzi", "dɔɡ"])
        assert difficulty == "hard"
    
    def test_phoneme_analyzer_init(self):
        """Test PhonemeAnalyzer initialization."""
        analyzer = PhonemeAnalyzer()
        assert analyzer.similarity_matrix is not None
        assert isinstance(analyzer.similarity_matrix, dict)
    
    def test_phoneme_similarity(self):
        """Test phoneme similarity computation."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical phonemes
        similarity = analyzer.compute_phoneme_similarity('θ', 'θ')
        assert similarity == 1.0
        
        # Test different phonemes
        similarity = analyzer.compute_phoneme_similarity('θ', 't')
        assert 0.0 <= similarity <= 1.0
    
    def test_sequence_similarity(self):
        """Test sequence similarity computation."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical sequences
        seq1 = ['h', 'ə', 'l', 'oʊ']
        seq2 = ['h', 'ə', 'l', 'oʊ']
        similarity = analyzer.compute_sequence_similarity(seq1, seq2)
        assert similarity == 1.0
        
        # Test different sequences
        seq1 = ['h', 'ə', 'l', 'oʊ']
        seq2 = ['w', 'ə', 'r', 'l', 'd']
        similarity = analyzer.compute_sequence_similarity(seq1, seq2)
        assert 0.0 <= similarity <= 1.0
    
    def test_feedback_generator_init(self):
        """Test FeedbackGenerator initialization."""
        generator = FeedbackGenerator()
        assert generator.feedback_templates is not None
        assert isinstance(generator.feedback_templates, dict)
    
    def test_feedback_generation(self):
        """Test feedback generation."""
        generator = FeedbackGenerator()
        
        # Mock analysis results
        analysis_results = {
            'overall_score': 0.75,
            'accent_level': 'Mild accent',
            'phoneme_accuracy': 0.8,
            'pitch_similarity': 0.7,
            'duration_similarity': 0.6,
            'stress_pattern_accuracy': 0.9
        }
        
        feedback = generator.generate_comprehensive_feedback(analysis_results)
        
        assert 'overall_assessment' in feedback
        assert 'specific_tips' in feedback
        assert 'improvement_areas' in feedback
        assert 'strengths' in feedback
        assert 'practice_recommendations' in feedback
        assert 'encouragement' in feedback
    
    def test_overall_assessment(self):
        """Test overall assessment generation."""
        generator = FeedbackGenerator()
        
        # Test high score
        results = {'overall_score': 0.9, 'accent_level': 'Native-like'}
        assessment = generator._generate_overall_assessment(results)
        assert 'excellent' in assessment.lower()
        
        # Test low score
        results = {'overall_score': 0.4, 'accent_level': 'Very strong accent'}
        assessment = generator._generate_overall_assessment(results)
        assert 'room for improvement' in assessment.lower()
    
    def test_encouragement_generation(self):
        """Test encouragement generation."""
        generator = FeedbackGenerator()
        
        # Test high score
        results = {'overall_score': 0.9}
        encouragement = generator._generate_encouragement(results)
        assert 'excellent' in encouragement.lower()
        
        # Test low score
        results = {'overall_score': 0.4}
        encouragement = generator._generate_encouragement(results)
        assert 'discouraged' in encouragement.lower()
    
    def test_improvement_areas_identification(self):
        """Test improvement area identification."""
        generator = FeedbackGenerator()
        
        # Test with low scores
        results = {
            'phoneme_accuracy': 0.5,
            'pitch_similarity': 0.4,
            'duration_similarity': 0.3,
            'stress_pattern_accuracy': 0.6
        }
        
        areas = generator._identify_improvement_areas(results)
        assert isinstance(areas, list)
        assert len(areas) > 0
        
        # Test with high scores
        results = {
            'phoneme_accuracy': 0.9,
            'pitch_similarity': 0.8,
            'duration_similarity': 0.9,
            'stress_pattern_accuracy': 0.8
        }
        
        areas = generator._identify_improvement_areas(results)
        assert isinstance(areas, list)
    
    def test_strength_identification(self):
        """Test strength identification."""
        generator = FeedbackGenerator()
        
        # Test with high scores
        results = {
            'phoneme_accuracy': 0.9,
            'pitch_similarity': 0.8,
            'duration_similarity': 0.9,
            'stress_pattern_accuracy': 0.8
        }
        
        strengths = generator._identify_strengths(results)
        assert isinstance(strengths, list)
        assert len(strengths) > 0
        
        # Test with low scores
        results = {
            'phoneme_accuracy': 0.5,
            'pitch_similarity': 0.4,
            'duration_similarity': 0.3,
            'stress_pattern_accuracy': 0.6
        }
        
        strengths = generator._identify_strengths(results)
        assert isinstance(strengths, list)


if __name__ == "__main__":
    pytest.main([__file__])
