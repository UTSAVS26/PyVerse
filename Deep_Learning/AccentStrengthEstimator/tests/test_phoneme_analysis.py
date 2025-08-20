"""
Tests for phoneme analysis components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis.phoneme_analyzer import PhonemeAnalyzer


class TestPhonemeAnalyzer:
    """Test cases for PhonemeAnalyzer class."""
    
    def test_init(self):
        """Test PhonemeAnalyzer initialization."""
        analyzer = PhonemeAnalyzer()
        assert analyzer.similarity_matrix is not None
        assert isinstance(analyzer.similarity_matrix, dict)
    
    def test_compute_phoneme_similarity(self):
        """Test phoneme similarity computation."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical phonemes
        similarity = analyzer.compute_phoneme_similarity('θ', 'θ')
        assert similarity == 1.0
        
        # Test similar phonemes
        similarity = analyzer.compute_phoneme_similarity('θ', 'ð')
        assert 0.0 <= similarity <= 1.0
        
        # Test different phonemes
        similarity = analyzer.compute_phoneme_similarity('θ', 't')
        assert 0.0 <= similarity <= 1.0
        
        # Test unknown phonemes
        similarity = analyzer.compute_phoneme_similarity('x', 'y')
        assert similarity == 0.0
    
    def test_compute_sequence_similarity(self):
        """Test sequence similarity computation."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical sequences
        seq1 = ['h', 'ə', 'l', 'oʊ']
        seq2 = ['h', 'ə', 'l', 'oʊ']
        similarity = analyzer.compute_sequence_similarity(seq1, seq2)
        assert similarity == 1.0
        
        # Test similar sequences
        seq1 = ['h', 'ə', 'l', 'oʊ']
        seq2 = ['h', 'ə', 'l', 'u']
        similarity = analyzer.compute_sequence_similarity(seq1, seq2)
        assert 0.0 <= similarity <= 1.0
        
        # Test different sequences
        seq1 = ['h', 'ə', 'l', 'oʊ']
        seq2 = ['w', 'ə', 'r', 'l', 'd']
        similarity = analyzer.compute_sequence_similarity(seq1, seq2)
        assert 0.0 <= similarity <= 1.0
        
        # Test empty sequences
        similarity = analyzer.compute_sequence_similarity([], [])
        assert similarity == 0.0
        
        similarity = analyzer.compute_sequence_similarity(['h', 'ə'], [])
        assert similarity == 0.0
    
    def test_analyze_phoneme_errors(self):
        """Test phoneme error analysis."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical sequences
        ref_phonemes = ['h', 'ə', 'l', 'oʊ']
        user_phonemes = ['h', 'ə', 'l', 'oʊ']
        analysis = analyzer.analyze_phoneme_errors(ref_phonemes, user_phonemes)
        
        assert analysis['total_phonemes'] == 4
        assert analysis['correct_phonemes'] == 4
        assert analysis['overall_accuracy'] == 1.0
        assert len(analysis['substitution_errors']) == 0
        assert len(analysis['deletion_errors']) == 0
        assert len(analysis['insertion_errors']) == 0
        
        # Test substitution errors
        ref_phonemes = ['θ', 'i', 'ŋ', 'k']
        user_phonemes = ['t', 'i', 'ŋ', 'k']
        analysis = analyzer.analyze_phoneme_errors(ref_phonemes, user_phonemes)
        
        assert analysis['total_phonemes'] == 4
        assert analysis['correct_phonemes'] == 3
        assert analysis['overall_accuracy'] == 0.75
        assert len(analysis['substitution_errors']) == 1
        assert analysis['substitution_errors'][0]['reference'] == 'θ'
        assert analysis['substitution_errors'][0]['user'] == 't'
        
        # Test deletion errors
        ref_phonemes = ['h', 'ə', 'l', 'oʊ']
        user_phonemes = ['h', 'ə', 'oʊ']
        analysis = analyzer.analyze_phoneme_errors(ref_phonemes, user_phonemes)
        
        assert analysis['total_phonemes'] == 4
        assert analysis['correct_phonemes'] == 3
        assert len(analysis['deletion_errors']) == 1
        assert analysis['deletion_errors'][0] == 'l'
        
        # Test insertion errors
        ref_phonemes = ['h', 'ə', 'l', 'oʊ']
        user_phonemes = ['h', 'ə', 'l', 'ə', 'oʊ']
        analysis = analyzer.analyze_phoneme_errors(ref_phonemes, user_phonemes)
        
        assert analysis['total_phonemes'] == 4
        assert analysis['correct_phonemes'] == 4
        assert len(analysis['insertion_errors']) == 1
        assert analysis['insertion_errors'][0] == 'ə'
    
    def test_get_common_error_patterns(self):
        """Test common error pattern identification."""
        analyzer = PhonemeAnalyzer()
        
        # Create mock error analysis
        error_analysis = {
            'substitution_errors': [
                {'reference': 'θ', 'user': 't', 'similarity': 0.4},
                {'reference': 'θ', 'user': 't', 'similarity': 0.4},
                {'reference': 'ð', 'user': 'd', 'similarity': 0.3},
                {'reference': 'ð', 'user': 'd', 'similarity': 0.3},
                {'reference': 'ð', 'user': 'd', 'similarity': 0.3}
            ],
            'deletion_errors': ['ŋ', 'ŋ', 'r'],
            'insertion_errors': []
        }
        
        patterns = analyzer.get_common_error_patterns(error_analysis)
        
        assert len(patterns) > 0
        
        # Check that patterns are sorted by frequency
        frequencies = [pattern['frequency'] for pattern in patterns]
        assert frequencies == sorted(frequencies, reverse=True)
        
        # Check that patterns with frequency >= 2 are included
        for pattern in patterns:
            assert pattern['frequency'] >= 2
    
    def test_get_pronunciation_suggestion(self):
        """Test pronunciation suggestion generation."""
        analyzer = PhonemeAnalyzer()
        
        # Test known substitutions
        suggestion = analyzer._get_pronunciation_suggestion('θ', 't')
        assert 'th' in suggestion.lower()
        assert 'tongue' in suggestion.lower()
        
        suggestion = analyzer._get_pronunciation_suggestion('ð', 'd')
        assert 'th' in suggestion.lower()
        assert 'voiced' in suggestion.lower()
        
        # Test unknown substitution
        suggestion = analyzer._get_pronunciation_suggestion('x', 'y')
        assert 'practice' in suggestion.lower()
        assert 'x' in suggestion
    
    def test_compute_stress_pattern_similarity(self):
        """Test stress pattern similarity computation."""
        analyzer = PhonemeAnalyzer()
        
        # Test identical patterns
        ref_stress = [1, 0, 1, 0]
        user_stress = [1, 0, 1, 0]
        similarity = analyzer.compute_stress_pattern_similarity(ref_stress, user_stress)
        assert similarity == 1.0
        
        # Test different patterns
        ref_stress = [1, 0, 1, 0]
        user_stress = [0, 1, 0, 1]
        similarity = analyzer.compute_stress_pattern_similarity(ref_stress, user_stress)
        assert 0.0 <= similarity <= 1.0
        
        # Test different lengths
        ref_stress = [1, 0, 1]
        user_stress = [1, 0, 1, 0, 1]
        similarity = analyzer.compute_stress_pattern_similarity(ref_stress, user_stress)
        assert 0.0 <= similarity <= 1.0
        
        # Test empty patterns
        similarity = analyzer.compute_stress_pattern_similarity([], [])
        assert similarity == 0.0
        
        similarity = analyzer.compute_stress_pattern_similarity([1, 0], [])
        assert similarity == 0.0
    
    def test_similarity_matrix_structure(self):
        """Test similarity matrix structure."""
        analyzer = PhonemeAnalyzer()
        matrix = analyzer.similarity_matrix
        
        # Check that matrix is not empty
        assert len(matrix) > 0
        
        # Check that all entries are dictionaries
        for phoneme, similarities in matrix.items():
            assert isinstance(similarities, dict)
            assert isinstance(phoneme, str)
            
            # Check that similarity scores are between 0 and 1
            for similar_phoneme, score in similarities.items():
                assert isinstance(similar_phoneme, str)
                assert 0.0 <= score <= 1.0
        
        # Check that matrix is symmetric (if A->B exists, B->A should exist)
        for phoneme, similarities in matrix.items():
            for similar_phoneme, score in similarities.items():
                if similar_phoneme in matrix:
                    assert phoneme in matrix[similar_phoneme]
                    assert matrix[similar_phoneme][phoneme] == score
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        analyzer = PhonemeAnalyzer()
        
        # Test with very long sequences
        long_seq1 = ['h'] * 1000
        long_seq2 = ['h'] * 1000
        similarity = analyzer.compute_sequence_similarity(long_seq1, long_seq2)
        assert similarity == 1.0
        
        # Test with single phoneme
        similarity = analyzer.compute_sequence_similarity(['h'], ['h'])
        assert similarity == 1.0
        
        similarity = analyzer.compute_sequence_similarity(['h'], ['t'])
        assert 0.0 <= similarity <= 1.0
        
        # Test with special characters
        similarity = analyzer.compute_phoneme_similarity('θ', 'ð')
        assert 0.0 <= similarity <= 1.0
        
        # Test with numbers (should be treated as unknown)
        similarity = analyzer.compute_phoneme_similarity('1', '2')
        assert similarity == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
