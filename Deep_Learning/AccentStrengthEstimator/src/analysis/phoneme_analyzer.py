"""
Phoneme analysis and comparison functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from difflib import SequenceMatcher


class PhonemeAnalyzer:
    """Analyzes and compares phoneme sequences."""
    
    def __init__(self):
        """Initialize the phoneme analyzer."""
        self.similarity_matrix = self._create_similarity_matrix()
    
    def _create_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Create a similarity matrix for phoneme comparison."""
        # Define phoneme similarity scores (0-1)
        # Higher values indicate more similar phonemes
        similarity_matrix = {
            # Vowels
            'i': {'ɪ': 0.8, 'iː': 0.9, 'e': 0.6, 'ɛ': 0.5},
            'ɪ': {'i': 0.8, 'iː': 0.7, 'e': 0.7, 'ɛ': 0.6},
            'e': {'ɛ': 0.8, 'æ': 0.6, 'ɪ': 0.7, 'i': 0.6},
            'ɛ': {'e': 0.8, 'æ': 0.7, 'ɪ': 0.6, 'i': 0.5},
            'æ': {'ɛ': 0.7, 'ɑ': 0.6, 'e': 0.6},
            'ɑ': {'æ': 0.6, 'ɔ': 0.7, 'ʌ': 0.6},
            'ɔ': {'ɑ': 0.7, 'oʊ': 0.8, 'ʌ': 0.6},
            'oʊ': {'ɔ': 0.8, 'u': 0.7, 'ʊ': 0.6},
            'u': {'ʊ': 0.8, 'oʊ': 0.7, 'ʌ': 0.5},
            'ʊ': {'u': 0.8, 'oʊ': 0.6, 'ʌ': 0.6},
            'ʌ': {'ɑ': 0.6, 'ɔ': 0.6, 'ʊ': 0.6, 'u': 0.5},
            
            # Consonants - similar sounds
            'θ': {'ð': 0.9, 't': 0.4, 'f': 0.3},
            'ð': {'θ': 0.9, 'd': 0.4, 'v': 0.3},
            'ʃ': {'ʒ': 0.9, 's': 0.4, 'tʃ': 0.6},
            'ʒ': {'ʃ': 0.9, 'z': 0.4, 'dʒ': 0.6},
            'tʃ': {'dʒ': 0.9, 'ʃ': 0.6, 't': 0.5},
            'dʒ': {'tʃ': 0.9, 'ʒ': 0.6, 'd': 0.5},
            'ŋ': {'n': 0.7, 'g': 0.4, 'k': 0.3},
            'r': {'ɹ': 0.9, 'l': 0.4, 'w': 0.3},
            'l': {'r': 0.4, 'w': 0.5, 'j': 0.3},
            'w': {'l': 0.5, 'r': 0.3, 'u': 0.6},
            'j': {'i': 0.6, 'l': 0.3, 'w': 0.2},
        }
        
        # Add reverse mappings
        for phoneme, similarities in list(similarity_matrix.items()):
            for similar_phoneme, score in similarities.items():
                if similar_phoneme not in similarity_matrix:
                    similarity_matrix[similar_phoneme] = {}
                similarity_matrix[similar_phoneme][phoneme] = score
        
        return similarity_matrix
    
    def compute_phoneme_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """
        Compute similarity between two phonemes.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if phoneme1 == phoneme2:
            return 1.0
        
        # Check if phonemes are in similarity matrix
        if phoneme1 in self.similarity_matrix and phoneme2 in self.similarity_matrix[phoneme1]:
            return self.similarity_matrix[phoneme1][phoneme2]
        
        # Default similarity for unknown phonemes
        return 0.0
    
    def compute_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """
        Compute similarity between two phoneme sequences using dynamic programming.
        
        Args:
            seq1: First phoneme sequence
            seq2: Second phoneme sequence
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Use Levenshtein distance with phoneme similarity
        n, m = len(seq1), len(seq2)
        dp = np.zeros((n + 1, m + 1))
        
        # Initialize first row and column
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Calculate substitution cost based on phoneme similarity
                    substitution_cost = 1.0 - self.compute_phoneme_similarity(seq1[i-1], seq2[j-1])
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # deletion
                        dp[i][j-1] + 1,      # insertion
                        dp[i-1][j-1] + substitution_cost  # substitution
                    )
        
        # Convert distance to similarity
        max_distance = max(n, m)
        similarity = 1.0 - (dp[n][m] / max_distance)
        return max(0.0, min(1.0, similarity))
    
    def analyze_phoneme_errors(self, reference: List[str], user: List[str]) -> Dict[str, Any]:
        """
        Analyze specific phoneme errors in user pronunciation.
        
        Args:
            reference: Reference phoneme sequence
            user: User phoneme sequence
            
        Returns:
            dict: Analysis results with error details
        """
        analysis = {
            'total_phonemes': len(reference),
            'correct_phonemes': 0,
            'substitution_errors': [],
            'insertion_errors': [],
            'deletion_errors': [],
            'overall_accuracy': 0.0
        }
        
        # Use sequence matcher to find differences
        matcher = SequenceMatcher(None, reference, user)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                analysis['correct_phonemes'] += (i2 - i1)
            elif tag == 'replace':
                # Substitution errors
                ref_phonemes = reference[i1:i2]
                user_phonemes = user[j1:j2]
                for ref_phoneme, user_phoneme in zip(ref_phonemes, user_phonemes):
                    analysis['substitution_errors'].append({
                        'reference': ref_phoneme,
                        'user': user_phoneme,
                        'similarity': self.compute_phoneme_similarity(ref_phoneme, user_phoneme)
                    })
            elif tag == 'delete':
                # Deletion errors
                for phoneme in reference[i1:i2]:
                    analysis['deletion_errors'].append(phoneme)
            elif tag == 'insert':
                # Insertion errors
                for phoneme in user[j1:j2]:
                    analysis['insertion_errors'].append(phoneme)
        
        # Calculate overall accuracy
        total_errors = (len(analysis['substitution_errors']) + 
                       len(analysis['deletion_errors']) + 
                       len(analysis['insertion_errors']))
        analysis['overall_accuracy'] = analysis['correct_phonemes'] / analysis['total_phonemes']
        
        return analysis
    
    def get_common_error_patterns(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify common error patterns from phoneme analysis.
        
        Args:
            error_analysis: Results from analyze_phoneme_errors
            
        Returns:
            list: Common error patterns with suggestions
        """
        patterns = []
        
        # Analyze substitution errors
        substitution_counts = {}
        for error in error_analysis['substitution_errors']:
            key = f"{error['reference']} -> {error['user']}"
            substitution_counts[key] = substitution_counts.get(key, 0) + 1
        
        # Find most common substitutions
        for substitution, count in sorted(substitution_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
            if count >= 2:  # Only include patterns that occur multiple times
                ref_phoneme, user_phoneme = substitution.split(' -> ')
                patterns.append({
                    'type': 'substitution',
                    'pattern': substitution,
                    'frequency': count,
                    'suggestion': self._get_pronunciation_suggestion(ref_phoneme, user_phoneme)
                })
        
        # Analyze deletion errors
        deletion_counts = {}
        for phoneme in error_analysis['deletion_errors']:
            deletion_counts[phoneme] = deletion_counts.get(phoneme, 0) + 1
        
        for phoneme, count in sorted(deletion_counts.items(), 
                                   key=lambda x: x[1], reverse=True):
            if count >= 2:
                patterns.append({
                    'type': 'deletion',
                    'pattern': phoneme,
                    'frequency': count,
                    'suggestion': f"Practice pronouncing the '{phoneme}' sound"
                })
        
        return patterns
    
    def _get_pronunciation_suggestion(self, ref_phoneme: str, user_phoneme: str) -> str:
        """
        Get pronunciation suggestion for a phoneme substitution.
        
        Args:
            ref_phoneme: Reference phoneme
            user_phoneme: User's pronunciation
            
        Returns:
            str: Pronunciation suggestion
        """
        suggestions = {
            'θ -> t': "Practice the 'th' sound by placing your tongue between your teeth",
            'ð -> d': "Practice the voiced 'th' sound by placing your tongue between your teeth",
            'ʃ -> s': "Practice the 'sh' sound by rounding your lips and raising your tongue",
            'ʒ -> z': "Practice the voiced 'zh' sound like in 'vision'",
            'ŋ -> n': "Practice the 'ng' sound by keeping your mouth open and using your nose",
            'r -> l': "Practice the 'r' sound by curling your tongue back",
            'æ -> ɛ': "Practice the 'a' sound in 'cat' by opening your mouth wider",
            'ɪ -> i': "Practice the short 'i' sound by keeping your mouth more closed"
        }
        
        key = f"{ref_phoneme} -> {user_phoneme}"
        return suggestions.get(key, f"Practice the '{ref_phoneme}' sound")
    
    def compute_stress_pattern_similarity(self, ref_stress: List[int], user_stress: List[int]) -> float:
        """
        Compute similarity between stress patterns.
        
        Args:
            ref_stress: Reference stress pattern
            user_stress: User stress pattern
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not ref_stress or not user_stress:
            return 0.0
        
        # Pad shorter sequence with zeros
        max_len = max(len(ref_stress), len(user_stress))
        ref_padded = ref_stress + [0] * (max_len - len(ref_stress))
        user_padded = user_stress + [0] * (max_len - len(user_stress))
        
        # Calculate correlation
        correlation = np.corrcoef(ref_padded, user_padded)[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # Convert to similarity score
        similarity = (correlation + 1) / 2
        return max(0.0, min(1.0, similarity))
