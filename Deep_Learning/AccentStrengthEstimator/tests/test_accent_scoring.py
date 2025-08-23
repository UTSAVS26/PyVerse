"""
Tests for accent scoring components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.scoring.accent_scorer import AccentScorer
from src.scoring.feedback_generator import FeedbackGenerator


class TestAccentScorer:
    """Test cases for AccentScorer class."""
    
    def test_init(self):
        """Test AccentScorer initialization."""
        scorer = AccentScorer()
        assert scorer.weights is not None
        assert 'phoneme_accuracy' in scorer.weights
        assert 'pitch_similarity' in scorer.weights
        assert 'duration_similarity' in scorer.weights
        assert 'stress_pattern' in scorer.weights
    
    def test_init_custom_weights(self):
        """Test AccentScorer initialization with custom weights."""
        custom_weights = {
            'phoneme_accuracy': 0.5,
            'pitch_similarity': 0.3,
            'duration_similarity': 0.2
        }
        scorer = AccentScorer(weights=custom_weights)
        assert scorer.weights == custom_weights
    
    def test_compute_overall_score(self):
        """Test overall score computation."""
        scorer = AccentScorer()
        
        # Test with perfect scores
        analysis_results = {
            'phoneme_accuracy': 1.0,
            'pitch_similarity': 1.0,
            'duration_similarity': 1.0,
            'stress_pattern_accuracy': 1.0
        }
        
        scores = scorer.compute_overall_score(analysis_results)
        assert scores['overall_score'] == 1.0
        assert scores['accent_level'] == "Native-like"
        
        # Test with mixed scores
        analysis_results = {
            'phoneme_accuracy': 0.8,
            'pitch_similarity': 0.6,
            'duration_similarity': 0.7,
            'stress_pattern_accuracy': 0.9
        }
        
        scores = scorer.compute_overall_score(analysis_results)
        assert 0.0 <= scores['overall_score'] <= 1.0
        assert scores['accent_level'] in ["Native-like", "Very mild accent", "Mild accent", "Moderate accent", "Strong accent", "Very strong accent"]
        
        # Test with missing components
        analysis_results = {
            'phoneme_accuracy': 0.8,
            'pitch_similarity': 0.6
        }
        
        scores = scorer.compute_overall_score(analysis_results)
        assert 0.0 <= scores['overall_score'] <= 1.0
    
    def test_classify_accent_level(self):
        """Test accent level classification."""
        scorer = AccentScorer()
        
        # Test different score ranges
        assert scorer._classify_accent_level(0.95) == "Native-like"
        assert scorer._classify_accent_level(0.85) == "Very mild accent"
        assert scorer._classify_accent_level(0.75) == "Mild accent"
        assert scorer._classify_accent_level(0.65) == "Moderate accent"
        assert scorer._classify_accent_level(0.55) == "Strong accent"
        assert scorer._classify_accent_level(0.45) == "Very strong accent"
    
    def test_analyze_phrase(self):
        """Test phrase analysis."""
        scorer = AccentScorer()
        
        # Mock audio data
        audio_data = np.random.rand(22050)  # 1 second at 22050 Hz
        
        # Mock reference data
        reference_data = {
            'text': 'Hello world',
            'phonemes': ['h', 'ə', 'l', 'oʊ', 'w', 'ɜr', 'l', 'd'],
            'stress_pattern': [1, 0, 1, 0]
        }
        
        # Mock user phonemes
        user_phonemes = ['h', 'ə', 'l', 'oʊ', 'w', 'ɜr', 'l', 'd']
        
        # Test analysis
        result = scorer.analyze_phrase(audio_data, reference_data, user_phonemes)
        
        assert 'phoneme_accuracy' in result
        assert 'pitch_similarity' in result
        assert 'duration_similarity' in result
        assert 'stress_pattern_accuracy' in result
        assert 'overall_score' in result
        assert 'accent_level' in result
        
        # Check score ranges
        assert 0.0 <= result['phoneme_accuracy'] <= 1.0
        assert 0.0 <= result['pitch_similarity'] <= 1.0
        assert 0.0 <= result['duration_similarity'] <= 1.0
        assert 0.0 <= result['stress_pattern_accuracy'] <= 1.0
        assert 0.0 <= result['overall_score'] <= 1.0
    
    def test_analyze_multiple_phrases(self):
        """Test multiple phrase analysis."""
        scorer = AccentScorer()
        
        # Mock phrase results
        phrase_results = [
            {
                'phoneme_accuracy': 0.8,
                'pitch_similarity': 0.7,
                'duration_similarity': 0.6,
                'stress_pattern_accuracy': 0.9,
                'error_patterns': [
                    {'type': 'substitution', 'pattern': 'θ -> t', 'frequency': 2}
                ],
                'phoneme_errors': {
                    'total_phonemes': 10,
                    'correct_phonemes': 8,
                    'substitution_errors': [{'reference': 'θ', 'user': 't'}],
                    'deletion_errors': [],
                    'insertion_errors': []
                }
            },
            {
                'phoneme_accuracy': 0.9,
                'pitch_similarity': 0.8,
                'duration_similarity': 0.7,
                'stress_pattern_accuracy': 0.8,
                'error_patterns': [
                    {'type': 'substitution', 'pattern': 'θ -> t', 'frequency': 1}
                ],
                'phoneme_errors': {
                    'total_phonemes': 12,
                    'correct_phonemes': 11,
                    'substitution_errors': [{'reference': 'θ', 'user': 't'}],
                    'deletion_errors': [],
                    'insertion_errors': []
                }
            }
        ]
        
        # Test analysis
        result = scorer.analyze_multiple_phrases(phrase_results)
        
        assert 'phoneme_accuracy' in result
        assert 'pitch_similarity' in result
        assert 'duration_similarity' in result
        assert 'stress_pattern_accuracy' in result
        assert 'overall_score' in result
        assert 'accent_level' in result
        assert 'common_error_patterns' in result
        assert 'overall_phoneme_accuracy' in result
        
        # Check that scores are averages
        expected_phoneme_accuracy = (0.8 + 0.9) / 2
        assert abs(result['phoneme_accuracy'] - expected_phoneme_accuracy) < 0.01
    
    def test_analyze_multiple_phrases_empty(self):
        """Test multiple phrase analysis with empty results."""
        scorer = AccentScorer()
        
        result = scorer.analyze_multiple_phrases([])
        assert result == {}
    
    def test_get_score_breakdown(self):
        """Test score breakdown generation."""
        scorer = AccentScorer()
        
        # Mock scores
        scores = {
            'overall_score': 0.75,
            'accent_level': 'Mild accent',
            'phoneme_accuracy': 0.8,
            'pitch_similarity': 0.7,
            'duration_similarity': 0.6,
            'stress_pattern_accuracy': 0.9
        }
        
        breakdown = scorer.get_score_breakdown(scores)
        
        assert breakdown['overall_score'] == 0.75
        assert breakdown['accent_level'] == 'Mild accent'
        assert 'component_scores' in breakdown
        assert 'strengths' in breakdown
        assert 'weaknesses' in breakdown
        
        # Check component scores
        component_scores = breakdown['component_scores']
        assert 'phoneme_accuracy' in component_scores
        assert 'pitch_similarity' in component_scores
        assert 'duration_similarity' in component_scores
        assert 'stress_pattern' in component_scores
        
        # Check that strengths and weaknesses are identified
        assert isinstance(breakdown['strengths'], list)
        assert isinstance(breakdown['weaknesses'], list)


class TestFeedbackGenerator:
    """Test cases for FeedbackGenerator class."""
    
    def test_init(self):
        """Test FeedbackGenerator initialization."""
        generator = FeedbackGenerator()
        assert generator.feedback_templates is not None
        assert isinstance(generator.feedback_templates, dict)
    
    def test_generate_comprehensive_feedback(self):
        """Test comprehensive feedback generation."""
        generator = FeedbackGenerator()
        
        # Mock analysis results
        analysis_results = {
            'overall_score': 0.75,
            'accent_level': 'Mild accent',
            'phoneme_accuracy': 0.8,
            'pitch_similarity': 0.7,
            'duration_similarity': 0.6,
            'stress_pattern_accuracy': 0.9,
            'error_patterns': [
                {'type': 'substitution', 'pattern': 'θ -> t', 'suggestions': ['Practice th sound']}
            ],
            'pitch_analysis': {'intonation_type': 'declarative'},
            'timing_analysis': {'speech_rate_category': 'normal'}
        }
        
        feedback = generator.generate_comprehensive_feedback(analysis_results)
        
        assert 'overall_assessment' in feedback
        assert 'specific_tips' in feedback
        assert 'improvement_areas' in feedback
        assert 'strengths' in feedback
        assert 'practice_recommendations' in feedback
        assert 'encouragement' in feedback
        
        # Check that feedback is not empty
        assert len(feedback['overall_assessment']) > 0
        assert len(feedback['encouragement']) > 0
    
    def test_generate_overall_assessment(self):
        """Test overall assessment generation."""
        generator = FeedbackGenerator()
        
        # Test different score levels
        results = {'overall_score': 0.95, 'accent_level': 'Native-like'}
        assessment = generator._generate_overall_assessment(results)
        assert 'excellent' in assessment.lower()
        
        results = {'overall_score': 0.85, 'accent_level': 'Very mild accent'}
        assessment = generator._generate_overall_assessment(results)
        assert 'very good' in assessment.lower()
        
        results = {'overall_score': 0.45, 'accent_level': 'Very strong accent'}
        assessment = generator._generate_overall_assessment(results)
        assert 'room for improvement' in assessment.lower()
    
    def test_generate_specific_tips(self):
        """Test specific tips generation."""
        generator = FeedbackGenerator()
        
        # Mock results with error patterns
        results = {
            'error_patterns': [
                {'type': 'substitution', 'pattern': 'θ -> t', 'suggestions': ['Practice th sound']},
                {'type': 'deletion', 'pattern': 'ŋ', 'suggestions': ['Don\'t skip ng sound']}
            ],
            'pitch_analysis': {'intonation_type': 'interrogative'},
            'timing_analysis': {'speech_rate_category': 'slow'}
        }
        
        tips = generator._generate_specific_tips(results)
        assert isinstance(tips, list)
        assert len(tips) > 0
    
    def test_identify_improvement_areas(self):
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
    
    def test_identify_strengths(self):
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
    
    def test_generate_practice_recommendations(self):
        """Test practice recommendation generation."""
        generator = FeedbackGenerator()
        
        # Mock results
        results = {
            'error_patterns': [
                {'type': 'substitution', 'pattern': 'θ -> t'}
            ],
            'phoneme_errors': {'substitution_errors': [{'reference': 'θ', 'user': 't'}]},
            'pitch_similarity': 0.6,
            'duration_similarity': 0.5
        }
        
        recommendations = generator._generate_practice_recommendations(results)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_generate_encouragement(self):
        """Test encouragement generation."""
        generator = FeedbackGenerator()
        
        # Test different score levels
        results = {'overall_score': 0.9}
        encouragement = generator._generate_encouragement(results)
        assert 'excellent' in encouragement.lower()
        
        results = {'overall_score': 0.7}
        encouragement = generator._generate_encouragement(results)
        assert 'progress' in encouragement.lower()
        
        results = {'overall_score': 0.4}
        encouragement = generator._generate_encouragement(results)
        assert 'discouraged' in encouragement.lower()
    
    def test_format_feedback_report(self):
        """Test feedback report formatting."""
        generator = FeedbackGenerator()
        
        # Mock feedback
        feedback = {
            'overall_assessment': 'Good pronunciation with a mild accent.',
            'specific_tips': ['Practice th sound', 'Work on intonation'],
            'improvement_areas': ['phoneme pronunciation'],
            'strengths': ['stress patterns'],
            'practice_recommendations': ['Practice minimal pairs'],
            'encouragement': 'Keep up the great work!'
        }
        
        report = generator.format_feedback_report(feedback)
        assert isinstance(report, str)
        assert 'Accent Strength Estimator Results' in report
        assert 'Good pronunciation' in report
        assert 'Practice th sound' in report
    
    def test_generate_quick_feedback(self):
        """Test quick feedback generation."""
        generator = FeedbackGenerator()
        
        # Test different score levels
        results = {'overall_score': 0.9}
        feedback = generator.generate_quick_feedback(results)
        assert 'great' in feedback.lower()
        
        results = {'overall_score': 0.7}
        feedback = generator.generate_quick_feedback(results)
        assert 'effort' in feedback.lower()
        
        results = {'overall_score': 0.4}
        feedback = generator.generate_quick_feedback(results)
        assert 'practicing' in feedback.lower()
    
    def test_generate_phoneme_specific_feedback(self):
        """Test phoneme-specific feedback generation."""
        generator = FeedbackGenerator()
        
        # Mock error analysis
        error_analysis = {
            'substitution_errors': [
                {'reference': 'θ', 'user': 't', 'similarity': 0.3},
                {'reference': 'ð', 'user': 'd', 'similarity': 0.4}
            ],
            'deletion_errors': ['ŋ', 'r']
        }
        
        feedback = generator.generate_phoneme_specific_feedback(error_analysis)
        assert isinstance(feedback, list)
        assert len(feedback) > 0


if __name__ == "__main__":
    pytest.main([__file__])
