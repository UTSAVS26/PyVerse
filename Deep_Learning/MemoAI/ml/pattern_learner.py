import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle
import os
import sys

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timer import SessionTimer


class DifficultyLevel:
    """Enum for difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class PatternLearner:
    """ML model for analyzing player behavior and predicting optimal difficulty."""
    
    def __init__(self, data_file: str = "data/user_sessions.json"):
        self.data_file = data_file
        self.difficulty_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'avg_reaction_time', 'std_reaction_time', 'total_moves',
            'mistakes', 'total_time', 'grid_size', 'completion_percentage'
        ]
    
    def load_data(self) -> Dict[str, Any]:
        """Load session data from JSON file."""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"sessions": [], "user_profiles": {}, "model_data": {"difficulty_predictions": [], "performance_trends": []}}
    
    def save_data(self, data: Dict[str, Any]) -> None:
        """Save session data to JSON file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_session(self, session_data: Dict[str, Any]) -> None:
        """Add a new session to the data."""
        data = self.load_data()
        data["sessions"].append(session_data)
        self.save_data(data)
    
    def extract_features(self, session: Dict[str, Any]) -> List[float]:
        """Extract features from a session for ML training."""
        features = []
        
        # Basic performance metrics
        features.append(session.get('avg_reaction_time', 0.0))
        features.append(session.get('std_reaction_time', 0.0))
        features.append(session.get('total_moves', 0))
        features.append(session.get('mistakes', 0))
        features.append(session.get('total_time', 0.0))
        features.append(session.get('grid_size', 4))
        features.append(session.get('completion_percentage', 0.0))
        
        return features
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for the ML models."""
        data = self.load_data()
        sessions = data.get("sessions", [])
        
        if len(sessions) < 5:
            # Not enough data to train
            return np.array([]), np.array([]), np.array([])
        
        X = []
        y_difficulty = []
        y_performance = []
        
        for session in sessions:
            if session.get('completed', False):
                features = self.extract_features(session)
                X.append(features)
                
                # Target for difficulty prediction (based on performance)
                performance_score = self._calculate_performance_score(session)
                difficulty_level = self._determine_difficulty_level(performance_score)
                y_difficulty.append(difficulty_level)
                
                # Target for performance prediction (completion time)
                y_performance.append(session.get('total_time', 0.0))
        
        return np.array(X), np.array(y_difficulty), np.array(y_performance)
    
    def _calculate_performance_score(self, session: Dict[str, Any]) -> float:
        """Calculate a performance score based on session data."""
        # Normalize different metrics and combine them
        time_score = max(0, 1 - (session.get('total_time', 0) / 300))  # Normalize to 5 minutes
        mistake_penalty = max(0, 1 - (session.get('mistakes', 0) / 10))  # Penalty for mistakes
        reaction_score = max(0, 1 - (session.get('avg_reaction_time', 0) / 5))  # Normalize to 5 seconds
        
        # Weighted combination
        performance_score = (0.4 * time_score + 0.3 * mistake_penalty + 0.3 * reaction_score)
        return max(0, min(1, performance_score))  # Clamp between 0 and 1
    
    def _determine_difficulty_level(self, performance_score: float) -> str:
        """Determine difficulty level based on performance score."""
        if performance_score >= 0.8:
            return DifficultyLevel.EASY
        elif performance_score >= 0.6:
            return DifficultyLevel.MEDIUM
        elif performance_score >= 0.4:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT
    
    def train_models(self) -> Dict[str, Any]:
        """Train the ML models with available session data."""
        X, y_difficulty, y_performance = self.prepare_training_data()
        
        if len(X) == 0:
            return {"status": "insufficient_data", "message": "Need at least 5 completed sessions to train"}
        
        # Split data for training
        X_train, X_test, y_difficulty_train, y_difficulty_test = train_test_split(
            X, y_difficulty, test_size=0.2, random_state=42
        )
        
        X_train, X_test, y_performance_train, y_performance_test = train_test_split(
            X, y_performance, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train difficulty prediction model
        self.difficulty_model.fit(X_train_scaled, y_difficulty_train)
        difficulty_accuracy = accuracy_score(y_difficulty_test, self.difficulty_model.predict(X_test_scaled))
        
        # Train performance prediction model
        self.performance_model.fit(X_train_scaled, y_performance_train)
        performance_mse = mean_squared_error(y_performance_test, self.performance_model.predict(X_test_scaled))
        
        self.is_trained = True
        
        return {
            "status": "success",
            "difficulty_accuracy": difficulty_accuracy,
            "performance_mse": performance_mse,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    def predict_difficulty(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal difficulty level for a player."""
        if not self.is_trained:
            return {"error": "Model not trained", "recommended_difficulty": DifficultyLevel.MEDIUM}
        
        features = self.extract_features(session_data)
        features_scaled = self.scaler.transform([features])
        
        predicted_difficulty = self.difficulty_model.predict(features_scaled)[0]
        difficulty_confidence = np.max(self.difficulty_model.predict_proba(features_scaled))
        
        return {
            "recommended_difficulty": predicted_difficulty,
            "confidence": difficulty_confidence,
            "features_used": dict(zip(self.feature_names, features))
        }
    
    def predict_performance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict expected performance metrics."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        features = self.extract_features(session_data)
        features_scaled = self.scaler.transform([features])
        
        predicted_time = self.performance_model.predict(features_scaled)[0]
        
        return {
            "predicted_completion_time": predicted_time,
            "features_used": dict(zip(self.feature_names, features))
        }
    
    def get_player_insights(self, player_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about player behavior patterns."""
        if not player_sessions:
            return {"error": "No session data available"}
        
        # Calculate trends
        reaction_times = [s.get('avg_reaction_time', 0) for s in player_sessions]
        mistake_counts = [s.get('mistakes', 0) for s in player_sessions]
        completion_times = [s.get('total_time', 0) for s in player_sessions]
        
        insights = {
            "total_sessions": len(player_sessions),
            "avg_reaction_time": np.mean(reaction_times),
            "avg_mistakes": np.mean(mistake_counts),
            "avg_completion_time": np.mean(completion_times),
            "improvement_trend": self._calculate_improvement_trend(player_sessions),
            "strengths": self._identify_strengths(player_sessions),
            "weaknesses": self._identify_weaknesses(player_sessions)
        }
        
        return insights
    
    def _calculate_improvement_trend(self, sessions: List[Dict[str, Any]]) -> str:
        """Calculate if player is improving over time."""
        if len(sessions) < 3:
            return "insufficient_data"
        
        # Sort by session ID to get chronological order
        sorted_sessions = sorted(sessions, key=lambda x: x.get('session_id', 0))
        
        # Calculate improvement in reaction time
        early_reaction_times = [s.get('avg_reaction_time', 0) for s in sorted_sessions[:len(sorted_sessions)//2]]
        late_reaction_times = [s.get('avg_reaction_time', 0) for s in sorted_sessions[len(sorted_sessions)//2:]]
        
        early_avg = np.mean(early_reaction_times)
        late_avg = np.mean(late_reaction_times)
        
        if late_avg < early_avg * 0.9:  # 10% improvement
            return "improving"
        elif late_avg > early_avg * 1.1:  # 10% decline
            return "declining"
        else:
            return "stable"
    
    def _identify_strengths(self, sessions: List[Dict[str, Any]]) -> List[str]:
        """Identify player strengths."""
        strengths = []
        
        avg_reaction_time = np.mean([s.get('avg_reaction_time', 0) for s in sessions])
        avg_mistakes = np.mean([s.get('mistakes', 0) for s in sessions])
        avg_completion_time = np.mean([s.get('total_time', 0) for s in sessions])
        
        if avg_reaction_time < 1.5:
            strengths.append("Fast reaction time")
        if avg_mistakes < 3:
            strengths.append("Low mistake rate")
        if avg_completion_time < 120:
            strengths.append("Quick completion")
        
        return strengths
    
    def _identify_weaknesses(self, sessions: List[Dict[str, Any]]) -> List[str]:
        """Identify player weaknesses."""
        weaknesses = []
        
        avg_reaction_time = np.mean([s.get('avg_reaction_time', 0) for s in sessions])
        avg_mistakes = np.mean([s.get('mistakes', 0) for s in sessions])
        avg_completion_time = np.mean([s.get('total_time', 0) for s in sessions])
        
        if avg_reaction_time > 3.0:
            weaknesses.append("Slow reaction time")
        if avg_mistakes > 8:
            weaknesses.append("High mistake rate")
        if avg_completion_time > 300:
            weaknesses.append("Slow completion")
        
        return weaknesses
    
    def save_models(self, filepath: str = "ml/trained_models.pkl") -> None:
        """Save trained models to file."""
        if not self.is_trained:
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'difficulty_model': self.difficulty_model,
                'performance_model': self.performance_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
    
    def load_models(self, filepath: str = "ml/trained_models.pkl") -> bool:
        """Load trained models from file."""
        try:
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
                self.difficulty_model = models['difficulty_model']
                self.performance_model = models['performance_model']
                self.scaler = models['scaler']
                self.feature_names = models['feature_names']
                self.is_trained = True
                return True
        except FileNotFoundError:
            return False 