"""
Accent strength scoring functionality.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from ..analysis.phoneme_analyzer import PhonemeAnalyzer
from ..analysis.pitch_analyzer import PitchAnalyzer
from ..analysis.duration_analyzer import DurationAnalyzer


class AccentScorer:
    """Computes overall accent strength scores from various analysis components."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the accent scorer.
        
        Args:
            weights: Dictionary of weights for different scoring components
        """
        self.weights = weights or {
            'phoneme_accuracy': 0.4,
            'pitch_similarity': 0.25,
            'duration_similarity': 0.2,
            'stress_pattern': 0.15
        }
        
        # Initialize analyzers
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.pitch_analyzer = PitchAnalyzer()
        self.duration_analyzer = DurationAnalyzer()
        
    def compute_overall_score(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall accent strength score from analysis results.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            dict: Overall scoring results
        """
        scores = {}
        
        # Extract individual scores
        scores['phoneme_accuracy'] = analysis_results.get('phoneme_accuracy', 0.0)
        scores['pitch_similarity'] = analysis_results.get('pitch_similarity', 0.0)
        scores['duration_similarity'] = analysis_results.get('duration_similarity', 0.0)
        scores['stress_pattern'] = analysis_results.get('stress_pattern_accuracy', 0.0)
        
        # Compute weighted average
        overall_score = 0.0
        total_weight = 0.0
        
        for component, weight in self.weights.items():
            if component in scores:
                overall_score += scores[component] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 0.0
        
        scores['overall_score'] = overall_score
        scores['accent_level'] = self._classify_accent_level(overall_score)
        
        return scores
    
    def _classify_accent_level(self, score: float) -> str:
        """
        Classify accent level based on score.
        
        Args:
            score: Overall accent score (0-1)
            
        Returns:
            str: Accent level classification
        """
        if score >= 0.9:
            return "Native-like"
        elif score >= 0.8:
            return "Very mild accent"
        elif score >= 0.7:
            return "Mild accent"
        elif score >= 0.6:
            return "Moderate accent"
        elif score >= 0.5:
            return "Strong accent"
        else:
            return "Very strong accent"
    
    def analyze_phrase(self, user_audio: np.ndarray, reference_data: Dict[str, Any],
                       user_phonemes: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a single phrase and compute scores.
        
        Args:
            user_audio: User's recorded audio
            reference_data: Reference data for the phrase
            user_phonemes: User's phoneme sequence (optional)
            
        Returns:
            dict: Complete analysis results
        """
        results = {}
        
        # Phoneme analysis
        if user_phonemes and 'phonemes' in reference_data:
            ref_phonemes = reference_data['phonemes']
            phoneme_similarity = self.phoneme_analyzer.compute_sequence_similarity(
                ref_phonemes, user_phonemes
            )
            results['phoneme_accuracy'] = phoneme_similarity
            
            # Detailed phoneme error analysis
            error_analysis = self.phoneme_analyzer.analyze_phoneme_errors(
                ref_phonemes, user_phonemes
            )
            results['phoneme_errors'] = error_analysis
            
            # Common error patterns
            error_patterns = self.phoneme_analyzer.get_common_error_patterns(error_analysis)
            results['error_patterns'] = error_patterns
        
        # Pitch analysis
        if len(user_audio) > 0:
            user_pitch = self.pitch_analyzer.extract_pitch_contour(user_audio)
            if len(user_pitch) > 0:
                # For now, we'll use a simplified pitch similarity
                # In a real implementation, you'd compare with reference pitch
                pitch_features = self.pitch_analyzer.extract_pitch_features(user_pitch)
                pitch_analysis = self.pitch_analyzer.analyze_intonation_pattern(
                    user_pitch, reference_data.get('text', '')
                )
                
                # Use pitch variation as a proxy for similarity
                pitch_variance = pitch_features.get('pitch_variance', 0)
                pitch_similarity = min(1.0, pitch_variance / 1000)  # Normalize
                results['pitch_similarity'] = pitch_similarity
                results['pitch_analysis'] = pitch_analysis
        
        # Duration analysis
        if len(user_audio) > 0:
            duration_features = self.duration_analyzer.extract_duration_features(user_audio)
            timing_analysis = self.duration_analyzer.analyze_timing_patterns(
                user_audio, reference_data.get('text', '')
            )
            
            # Use timing consistency as a proxy for similarity
            timing_consistency = timing_analysis.get('timing_consistency', 0)
            results['duration_similarity'] = timing_consistency
            results['timing_analysis'] = timing_analysis
        
        # Stress pattern analysis
        if 'stress_pattern' in reference_data:
            # For now, use a simplified stress pattern similarity
            # In a real implementation, you'd extract stress from user audio
            stress_similarity = 0.7  # Placeholder
            results['stress_pattern_accuracy'] = stress_similarity
        
        # Compute overall score
        overall_scores = self.compute_overall_score(results)
        results.update(overall_scores)
        
        return results
    
    def analyze_multiple_phrases(self, phrase_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results from multiple phrases and compute aggregate scores.
        
        Args:
            phrase_results: List of results from individual phrases
            
        Returns:
            dict: Aggregate analysis results
        """
        if not phrase_results:
            return {}
        
        # Aggregate scores
        aggregate_scores = {
            'phoneme_accuracy': [],
            'pitch_similarity': [],
            'duration_similarity': [],
            'stress_pattern_accuracy': [],
            'overall_score': []
        }
        
        # Collect all error patterns
        all_error_patterns = []
        all_phoneme_errors = []
        
        for result in phrase_results:
            for score_type in aggregate_scores:
                if score_type in result:
                    aggregate_scores[score_type].append(result[score_type])
            
            # Collect error patterns
            if 'error_patterns' in result:
                all_error_patterns.extend(result['error_patterns'])
            
            if 'phoneme_errors' in result:
                all_phoneme_errors.append(result['phoneme_errors'])
        
        # Compute averages
        final_scores = {}
        for score_type, scores in aggregate_scores.items():
            if scores:
                final_scores[score_type] = np.mean(scores)
            else:
                final_scores[score_type] = 0.0
        
        # Compute overall score
        overall_scores = self.compute_overall_score(final_scores)
        final_scores.update(overall_scores)
        
        # Analyze common patterns across all phrases
        final_scores['common_error_patterns'] = self._analyze_common_patterns(all_error_patterns)
        final_scores['overall_phoneme_accuracy'] = self._compute_overall_phoneme_accuracy(all_phoneme_errors)
        
        return final_scores
    
    def _analyze_common_patterns(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze common error patterns across multiple phrases.
        
        Args:
            all_patterns: List of error patterns from all phrases
            
        Returns:
            list: Aggregated common patterns
        """
        if not all_patterns:
            return []
        
        # Count pattern frequencies
        pattern_counts = {}
        for pattern in all_patterns:
            key = f"{pattern['type']}_{pattern['pattern']}"
            if key not in pattern_counts:
                pattern_counts[key] = {
                    'type': pattern['type'],
                    'pattern': pattern['pattern'],
                    'frequency': 0,
                    'suggestions': []
                }
            
            pattern_counts[key]['frequency'] += pattern['frequency']
            if pattern.get('suggestion') and pattern['suggestion'] not in pattern_counts[key]['suggestions']:
                pattern_counts[key]['suggestions'].append(pattern['suggestion'])
        
        # Sort by frequency and return top patterns
        sorted_patterns = sorted(pattern_counts.values(), 
                               key=lambda x: x['frequency'], reverse=True)
        
        return sorted_patterns[:5]  # Return top 5 patterns
    
    def _compute_overall_phoneme_accuracy(self, all_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute overall phoneme accuracy across all phrases.
        
        Args:
            all_errors: List of phoneme error analyses from all phrases
            
        Returns:
            dict: Overall phoneme accuracy statistics
        """
        if not all_errors:
            return {}
        
        total_phonemes = sum(error.get('total_phonemes', 0) for error in all_errors)
        total_correct = sum(error.get('correct_phonemes', 0) for error in all_errors)
        
        overall_accuracy = total_correct / total_phonemes if total_phonemes > 0 else 0.0
        
        # Count error types
        total_substitutions = sum(len(error.get('substitution_errors', [])) for error in all_errors)
        total_deletions = sum(len(error.get('deletion_errors', [])) for error in all_errors)
        total_insertions = sum(len(error.get('insertion_errors', [])) for error in all_errors)
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_phonemes': total_phonemes,
            'correct_phonemes': total_correct,
            'substitution_errors': total_substitutions,
            'deletion_errors': total_deletions,
            'insertion_errors': total_insertions,
            'error_rate': (total_substitutions + total_deletions + total_insertions) / total_phonemes if total_phonemes > 0 else 0.0
        }
    
    def get_score_breakdown(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed breakdown of scores for reporting.
        
        Args:
            scores: Scoring results
            
        Returns:
            dict: Detailed score breakdown
        """
        breakdown = {
            'overall_score': scores.get('overall_score', 0.0),
            'accent_level': scores.get('accent_level', 'Unknown'),
            'component_scores': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # Component scores
        for component, weight in self.weights.items():
            score = scores.get(component, 0.0)
            breakdown['component_scores'][component] = {
                'score': score,
                'weight': weight,
                'contribution': score * weight
            }
        
        # Identify strengths and weaknesses
        for component, score_info in breakdown['component_scores'].items():
            score = score_info['score']
            if score >= 0.8:
                breakdown['strengths'].append(f"Strong {component.replace('_', ' ')}")
            elif score <= 0.5:
                breakdown['weaknesses'].append(f"Needs improvement in {component.replace('_', ' ')}")
        
        return breakdown
