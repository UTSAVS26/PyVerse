"""
Feedback generation functionality for pronunciation improvement.
"""

from typing import Dict, List, Any
import random


class FeedbackGenerator:
    """Generates personalized feedback for pronunciation improvement."""
    
    def __init__(self):
        """Initialize the feedback generator."""
        self.feedback_templates = self._load_feedback_templates()
        
    def _load_feedback_templates(self) -> Dict[str, List[str]]:
        """Load feedback templates for different types of feedback."""
        return {
            'phoneme_errors': [
                "Practice the '{phoneme}' sound - try {suggestion}",
                "Your '{phoneme}' sounds like '{substitution}' - focus on {suggestion}",
                "Work on the '{phoneme}' pronunciation: {suggestion}"
            ],
            'pitch_feedback': [
                "Your intonation is {issue} - {suggestion}",
                "Try to {suggestion} for better intonation",
                "Your pitch pattern needs work: {suggestion}"
            ],
            'timing_feedback': [
                "Your speech timing is {issue} - {suggestion}",
                "Try to {suggestion} for better rhythm",
                "Work on your timing: {suggestion}"
            ],
            'general_encouragement': [
                "Great effort! Keep practicing to improve your pronunciation.",
                "You're making good progress. Focus on the areas mentioned above.",
                "With consistent practice, you'll see improvement in your accent."
            ],
            'strength_recognition': [
                "Excellent work on {strength}!",
                "Your {strength} is very good.",
                "Keep up the great work with {strength}."
            ]
        }
    
    def generate_comprehensive_feedback(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive feedback from analysis results.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            dict: Comprehensive feedback
        """
        feedback = {
            'overall_assessment': self._generate_overall_assessment(analysis_results),
            'specific_tips': self._generate_specific_tips(analysis_results),
            'improvement_areas': self._identify_improvement_areas(analysis_results),
            'strengths': self._identify_strengths(analysis_results),
            'practice_recommendations': self._generate_practice_recommendations(analysis_results),
            'encouragement': self._generate_encouragement(analysis_results)
        }
        
        return feedback
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> str:
        """
        Generate overall assessment of the user's pronunciation.
        
        Args:
            results: Analysis results
            
        Returns:
            str: Overall assessment
        """
        overall_score = results.get('overall_score', 0.0)
        accent_level = results.get('accent_level', 'Unknown')
        
        if overall_score >= 0.9:
            return f"Excellent pronunciation! Your accent is {accent_level.lower()}."
        elif overall_score >= 0.8:
            return f"Very good pronunciation with a {accent_level.lower()}."
        elif overall_score >= 0.7:
            return f"Good pronunciation with a {accent_level.lower()}."
        elif overall_score >= 0.6:
            return f"Fair pronunciation with a {accent_level.lower()}."
        else:
            return f"Your pronunciation shows a {accent_level.lower()}. There's room for improvement."
    
    def _generate_specific_tips(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate specific tips based on analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            list: List of specific tips
        """
        tips = []
        
        # Phoneme-specific tips
        if 'error_patterns' in results:
            for pattern in results['error_patterns'][:3]:  # Top 3 patterns
                if pattern['type'] == 'substitution':
                    tips.append(f"Practice '{pattern['pattern']}' - {pattern.get('suggestions', [''])[0]}")
                elif pattern['type'] == 'deletion':
                    tips.append(f"Don't skip the '{pattern['pattern']}' sound")
        
        # Pitch/intonation tips
        if 'pitch_analysis' in results:
            pitch_analysis = results['pitch_analysis']
            if pitch_analysis.get('intonation_type') == 'interrogative':
                tips.append("Use rising intonation for questions")
            elif pitch_analysis.get('intonation_type') == 'declarative':
                tips.append("Use falling intonation for statements")
        
        # Timing tips
        if 'timing_analysis' in results:
            timing_analysis = results['timing_analysis']
            if timing_analysis.get('speech_rate_category') == 'slow':
                tips.append("Try speaking a bit faster")
            elif timing_analysis.get('speech_rate_category') == 'fast':
                tips.append("Try speaking a bit slower")
            
            if timing_analysis.get('rhythm_type') != 'stress_timed':
                tips.append("English is stress-timed - vary syllable duration more")
        
        return tips
    
    def _identify_improvement_areas(self, results: Dict[str, Any]) -> List[str]:
        """
        Identify areas that need improvement.
        
        Args:
            results: Analysis results
            
        Returns:
            list: Areas needing improvement
        """
        areas = []
        
        # Check component scores
        component_scores = {
            'phoneme_accuracy': 'phoneme pronunciation',
            'pitch_similarity': 'intonation patterns',
            'duration_similarity': 'speech rhythm and timing',
            'stress_pattern_accuracy': 'stress patterns'
        }
        
        for component, description in component_scores.items():
            score = results.get(component, 0.0)
            if score < 0.6:
                areas.append(description)
        
        return areas
    
    def _identify_strengths(self, results: Dict[str, Any]) -> List[str]:
        """
        Identify areas of strength.
        
        Args:
            results: Analysis results
            
        Returns:
            list: Areas of strength
        """
        strengths = []
        
        # Check component scores
        component_scores = {
            'phoneme_accuracy': 'phoneme pronunciation',
            'pitch_similarity': 'intonation patterns',
            'duration_similarity': 'speech rhythm and timing',
            'stress_pattern_accuracy': 'stress patterns'
        }
        
        for component, description in component_scores.items():
            score = results.get(component, 0.0)
            if score >= 0.8:
                strengths.append(description)
        
        return strengths
    
    def _generate_practice_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate practice recommendations.
        
        Args:
            results: Analysis results
            
        Returns:
            list: Practice recommendations
        """
        recommendations = []
        
        # Phoneme practice
        if 'error_patterns' in results:
            difficult_phonemes = set()
            for pattern in results['error_patterns']:
                if pattern['type'] == 'substitution':
                    difficult_phonemes.add(pattern['pattern'].split(' -> ')[0])
            
            if difficult_phonemes:
                recommendations.append(f"Practice these difficult sounds: {', '.join(difficult_phonemes)}")
        
        # Minimal pairs practice
        if 'phoneme_errors' in results:
            recommendations.append("Practice minimal pairs to distinguish similar sounds")
        
        # Intonation practice
        if results.get('pitch_similarity', 0.0) < 0.7:
            recommendations.append("Practice intonation patterns with different sentence types")
        
        # Rhythm practice
        if results.get('duration_similarity', 0.0) < 0.7:
            recommendations.append("Practice stress-timed rhythm with native speakers")
        
        # General recommendations
        recommendations.extend([
            "Record yourself and compare with native speakers",
            "Practice with tongue twisters to improve articulation",
            "Listen to native English speakers and mimic their pronunciation"
        ])
        
        return recommendations
    
    def _generate_encouragement(self, results: Dict[str, Any]) -> str:
        """
        Generate encouraging feedback.
        
        Args:
            results: Analysis results
            
        Returns:
            str: Encouraging message
        """
        overall_score = results.get('overall_score', 0.0)
        
        if overall_score >= 0.8:
            return "You're doing excellent! Keep up the great work."
        elif overall_score >= 0.6:
            return "You're making good progress. Continue practicing regularly."
        else:
            return "Don't get discouraged! Pronunciation takes time and practice. Keep working at it."
    
    def format_feedback_report(self, feedback: Dict[str, Any]) -> str:
        """
        Format feedback into a readable report.
        
        Args:
            feedback: Feedback dictionary
            
        Returns:
            str: Formatted feedback report
        """
        report = []
        
        # Overall assessment
        report.append("🎤 Accent Strength Estimator Results")
        report.append("=" * 40)
        report.append("")
        report.append(feedback['overall_assessment'])
        report.append("")
        
        # Specific tips
        if feedback['specific_tips']:
            report.append("💡 Improvement Tips:")
            for tip in feedback['specific_tips']:
                report.append(f"- {tip}")
            report.append("")
        
        # Practice recommendations
        if feedback['practice_recommendations']:
            report.append("🎯 Recommended Practice:")
            for rec in feedback['practice_recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Strengths
        if feedback['strengths']:
            report.append("✅ Your Strengths:")
            for strength in feedback['strengths']:
                report.append(f"- Good {strength}")
            report.append("")
        
        # Encouragement
        report.append(feedback['encouragement'])
        
        return "\n".join(report)
    
    def generate_quick_feedback(self, results: Dict[str, Any]) -> str:
        """
        Generate a quick, concise feedback message.
        
        Args:
            results: Analysis results
            
        Returns:
            str: Quick feedback message
        """
        overall_score = results.get('overall_score', 0.0)
        
        if overall_score >= 0.8:
            return "Great pronunciation! Keep practicing to maintain your skills."
        elif overall_score >= 0.6:
            return "Good effort! Focus on the specific tips provided to improve further."
        else:
            return "Keep practicing! Focus on the improvement areas mentioned above."
    
    def generate_phoneme_specific_feedback(self, error_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate phoneme-specific feedback.
        
        Args:
            error_analysis: Phoneme error analysis
            
        Returns:
            list: Phoneme-specific feedback messages
        """
        feedback = []
        
        if 'substitution_errors' in error_analysis:
            for error in error_analysis['substitution_errors']:
                ref_phoneme = error['reference']
                user_phoneme = error['user']
                similarity = error['similarity']
                
                if similarity < 0.5:
                    feedback.append(f"Your '{ref_phoneme}' sounds like '{user_phoneme}' - practice the correct pronunciation")
        
        if 'deletion_errors' in error_analysis:
            deleted_phonemes = set(error_analysis['deletion_errors'])
            if deleted_phonemes:
                feedback.append(f"Don't skip these sounds: {', '.join(deleted_phonemes)}")
        
        return feedback
