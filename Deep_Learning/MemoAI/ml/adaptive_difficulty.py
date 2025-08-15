import json
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.pattern_learner import PatternLearner, DifficultyLevel


class AdaptiveDifficulty:
    """Adaptive difficulty system that adjusts game parameters based on player performance."""
    
    def __init__(self, pattern_learner: PatternLearner):
        self.pattern_learner = pattern_learner
        self.difficulty_settings = {
            DifficultyLevel.EASY: {
                "grid_size": 3,
                "time_limit": 300,  # 5 minutes
                "max_mistakes": 10,
                "card_complexity": "simple"
            },
            DifficultyLevel.MEDIUM: {
                "grid_size": 4,
                "time_limit": 240,  # 4 minutes
                "max_mistakes": 8,
                "card_complexity": "simple"
            },
            DifficultyLevel.HARD: {
                "grid_size": 5,
                "time_limit": 180,  # 3 minutes
                "max_mistakes": 6,
                "card_complexity": "medium"
            },
            DifficultyLevel.EXPERT: {
                "grid_size": 6,
                "time_limit": 120,  # 2 minutes
                "max_mistakes": 4,
                "card_complexity": "complex"
            }
        }
    
    def get_difficulty_settings(self, difficulty_level: str) -> Dict[str, Any]:
        """Get settings for a specific difficulty level."""
        return self.difficulty_settings.get(difficulty_level, self.difficulty_settings[DifficultyLevel.MEDIUM])
    
    def analyze_player_performance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player performance and provide recommendations."""
        analysis = {
            "current_performance": self._evaluate_current_performance(session_data),
            "difficulty_prediction": self.pattern_learner.predict_difficulty(session_data),
            "performance_prediction": self.pattern_learner.predict_performance(session_data),
            "recommendations": self._generate_recommendations(session_data)
        }
        
        return analysis
    
    def _evaluate_current_performance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the current session performance."""
        total_time = session_data.get('total_time', 0)
        mistakes = session_data.get('mistakes', 0)
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        completion_percentage = session_data.get('completion_percentage', 0)
        
        # Performance scoring
        time_score = max(0, 1 - (total_time / 300))  # Normalize to 5 minutes
        mistake_score = max(0, 1 - (mistakes / 10))   # Normalize to 10 mistakes
        reaction_score = max(0, 1 - (avg_reaction_time / 5))  # Normalize to 5 seconds
        
        overall_score = (time_score + mistake_score + reaction_score) / 3
        
        return {
            "overall_score": overall_score,
            "time_score": time_score,
            "mistake_score": mistake_score,
            "reaction_score": reaction_score,
            "completion_percentage": completion_percentage,
            "performance_level": self._get_performance_level(overall_score)
        }
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "average"
        else:
            return "needs_improvement"
    
    def _generate_recommendations(self, session_data: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on performance."""
        recommendations = []
        
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        mistakes = session_data.get('mistakes', 0)
        total_time = session_data.get('total_time', 0)
        
        if avg_reaction_time > 3.0:
            recommendations.append("Try to improve your reaction time by focusing on card patterns")
        
        if mistakes > 5:
            recommendations.append("Take your time to avoid mistakes - accuracy is more important than speed")
        
        if total_time > 300:
            recommendations.append("Consider using memory techniques to improve your completion time")
        
        if not recommendations:
            recommendations.append("Great performance! Try increasing the difficulty for a challenge")
        
        return recommendations
    
    def suggest_next_difficulty(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest the next difficulty level based on current performance."""
        prediction = self.pattern_learner.predict_difficulty(session_data)
        
        if "error" in prediction:
            # Fallback to rule-based difficulty adjustment
            return self._rule_based_difficulty_adjustment(session_data)
        
        recommended_difficulty = prediction["recommended_difficulty"]
        confidence = prediction.get("confidence", 0.5)
        
        # Apply confidence-based adjustments
        if confidence < 0.6:
            # Low confidence - be conservative
            if recommended_difficulty == DifficultyLevel.EXPERT:
                recommended_difficulty = DifficultyLevel.HARD
            elif recommended_difficulty == DifficultyLevel.HARD:
                recommended_difficulty = DifficultyLevel.MEDIUM
        
        settings = self.get_difficulty_settings(recommended_difficulty)
        
        return {
            "recommended_difficulty": recommended_difficulty,
            "confidence": confidence,
            "settings": settings,
            "reasoning": self._explain_difficulty_choice(session_data, recommended_difficulty)
        }
    
    def _rule_based_difficulty_adjustment(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based difficulty adjustment when ML model is not available."""
        performance = self._evaluate_current_performance(session_data)
        overall_score = performance["overall_score"]
        
        if overall_score >= 0.8:
            recommended_difficulty = DifficultyLevel.HARD
        elif overall_score >= 0.6:
            recommended_difficulty = DifficultyLevel.MEDIUM
        elif overall_score >= 0.4:
            recommended_difficulty = DifficultyLevel.EASY
        else:
            recommended_difficulty = DifficultyLevel.EASY
        
        settings = self.get_difficulty_settings(recommended_difficulty)
        
        return {
            "recommended_difficulty": recommended_difficulty,
            "confidence": 0.5,  # Lower confidence for rule-based approach
            "settings": settings,
            "reasoning": f"Rule-based adjustment based on performance score: {overall_score:.2f}"
        }
    
    def _explain_difficulty_choice(self, session_data: Dict[str, Any], difficulty: str) -> str:
        """Explain why a particular difficulty level was chosen."""
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        mistakes = session_data.get('mistakes', 0)
        total_time = session_data.get('total_time', 0)
        
        reasons = []
        
        if difficulty == DifficultyLevel.EASY:
            if avg_reaction_time > 2.5:
                reasons.append("slow reaction time")
            if mistakes > 6:
                reasons.append("high mistake rate")
            if total_time > 240:
                reasons.append("slow completion time")
        elif difficulty == DifficultyLevel.MEDIUM:
            reasons.append("balanced performance")
        elif difficulty == DifficultyLevel.HARD:
            if avg_reaction_time < 1.5:
                reasons.append("fast reaction time")
            if mistakes < 3:
                reasons.append("low mistake rate")
            if total_time < 180:
                reasons.append("quick completion")
        elif difficulty == DifficultyLevel.EXPERT:
            reasons.append("excellent performance across all metrics")
        
        if not reasons:
            reasons.append("moderate performance")
        
        return f"Recommended {difficulty} difficulty due to: {', '.join(reasons)}"
    
    def get_adaptive_hints(self, session_data: Dict[str, Any]) -> List[str]:
        """Generate adaptive hints based on player behavior."""
        hints = []
        
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        mistakes = session_data.get('mistakes', 0)
        total_moves = session_data.get('total_moves', 0)
        
        if avg_reaction_time > 3.0:
            hints.append("💡 Try to remember card positions to reduce reaction time")
        
        if mistakes > 5:
            hints.append("💡 Take a moment to think before clicking - accuracy matters")
        
        if total_moves > (session_data.get('grid_size', 4) ** 2):
            hints.append("💡 Try to find pairs more efficiently by remembering patterns")
        
        if not hints:
            hints.append("🎯 Great job! Keep up the good work")
        
        return hints
    
    def track_progress(self, all_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track player progress over multiple sessions."""
        if len(all_sessions) < 2:
            return {"error": "Need at least 2 sessions to track progress"}
        
        # Sort sessions by session_id
        sorted_sessions = sorted(all_sessions, key=lambda x: x.get('session_id', 0))
        
        # Calculate progress metrics
        early_sessions = sorted_sessions[:len(sorted_sessions)//2]
        recent_sessions = sorted_sessions[len(sorted_sessions)//2:]
        
        early_avg_reaction = sum(s.get('avg_reaction_time', 0) for s in early_sessions) / len(early_sessions)
        recent_avg_reaction = sum(s.get('avg_reaction_time', 0) for s in recent_sessions) / len(recent_sessions)
        
        early_avg_mistakes = sum(s.get('mistakes', 0) for s in early_sessions) / len(early_sessions)
        recent_avg_mistakes = sum(s.get('mistakes', 0) for s in recent_sessions) / len(recent_sessions)
        
        early_avg_time = sum(s.get('total_time', 0) for s in early_sessions) / len(early_sessions)
        recent_avg_time = sum(s.get('total_time', 0) for s in recent_sessions) / len(recent_sessions)
        
        progress = {
            "total_sessions": len(all_sessions),
            "reaction_time_improvement": early_avg_reaction - recent_avg_reaction,
            "mistake_reduction": early_avg_mistakes - recent_avg_mistakes,
            "time_improvement": early_avg_time - recent_avg_time,
            "overall_trend": self._determine_overall_trend(early_sessions, recent_sessions)
        }
        
        return progress
    
    def _determine_overall_trend(self, early_sessions: List[Dict], recent_sessions: List[Dict]) -> str:
        """Determine overall performance trend."""
        early_performance = sum(self._evaluate_current_performance(s)["overall_score"] for s in early_sessions) / len(early_sessions)
        recent_performance = sum(self._evaluate_current_performance(s)["overall_score"] for s in recent_sessions) / len(recent_sessions)
        
        improvement = recent_performance - early_performance
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_personalized_feedback(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized feedback for the player."""
        performance = self._evaluate_current_performance(session_data)
        analysis = self.analyze_player_performance(session_data)
        
        feedback = {
            "performance_summary": performance,
            "strengths": self._identify_session_strengths(session_data),
            "areas_for_improvement": self._identify_improvement_areas(session_data),
            "next_session_recommendation": self.suggest_next_difficulty(session_data),
            "motivational_message": self._generate_motivational_message(performance["overall_score"])
        }
        
        return feedback
    
    def _identify_session_strengths(self, session_data: Dict[str, Any]) -> List[str]:
        """Identify strengths in the current session."""
        strengths = []
        
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        mistakes = session_data.get('mistakes', 0)
        total_time = session_data.get('total_time', 0)
        
        if avg_reaction_time < 1.5:
            strengths.append("Excellent reaction time")
        elif avg_reaction_time < 2.5:
            strengths.append("Good reaction time")
        
        if mistakes < 3:
            strengths.append("Very few mistakes")
        elif mistakes < 6:
            strengths.append("Good accuracy")
        
        if total_time < 120:
            strengths.append("Quick completion")
        elif total_time < 240:
            strengths.append("Reasonable completion time")
        
        return strengths
    
    def _identify_improvement_areas(self, session_data: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement in the current session."""
        areas = []
        
        avg_reaction_time = session_data.get('avg_reaction_time', 0)
        mistakes = session_data.get('mistakes', 0)
        total_time = session_data.get('total_time', 0)
        
        if avg_reaction_time > 3.0:
            areas.append("Reaction time could be improved")
        
        if mistakes > 8:
            areas.append("Try to reduce the number of mistakes")
        
        if total_time > 300:
            areas.append("Completion time could be faster")
        
        return areas
    
    def _generate_motivational_message(self, performance_score: float) -> str:
        """Generate a motivational message based on performance."""
        if performance_score >= 0.8:
            return "🎉 Outstanding performance! You're mastering this game!"
        elif performance_score >= 0.6:
            return "👍 Great job! You're showing real improvement!"
        elif performance_score >= 0.4:
            return "💪 Good effort! Keep practicing to get even better!"
        else:
            return "🌟 Don't worry! Every expert was once a beginner. Keep going!" 