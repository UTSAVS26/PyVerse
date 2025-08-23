"""
Prosody classifier module for VoiceMoodMirror.
Maps prosodic features to emotional states using rule-based heuristics and ML models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import json


class EmotionClassifier:
    """Base class for emotion classification from prosodic features."""
    
    def __init__(self):
        """Initialize the emotion classifier."""
        self.emotions = ['happy', 'sad', 'angry', 'calm', 'excited', 'tired', 'neutral']
        self.confidence_threshold = 0.3
        
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion from prosodic features.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        raise NotImplementedError("Subclasses must implement classify method")
    
    def get_emotion_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over all emotions.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Dictionary mapping emotions to probabilities
        """
        emotion, confidence = self.classify(features)
        probabilities = {emotion: 0.0 for emotion in self.emotions}
        probabilities[emotion] = confidence
        
        return probabilities


class RuleBasedClassifier(EmotionClassifier):
    """Rule-based emotion classifier using prosodic feature heuristics."""
    
    def __init__(self):
        """Initialize the rule-based classifier."""
        super().__init__()
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize classification rules based on prosodic features."""
        return [
            # Happy: high pitch, high energy, fast tempo
            {
                'emotion': 'happy',
                'conditions': [
                    lambda f: f.get('pitch_mean', 0) > 180,
                    lambda f: f.get('energy_mean', 0) > 0.15,
                    lambda f: f.get('tempo', 0) > 140,
                    lambda f: f.get('pitch_variability', 0) > 0.2
                ],
                'confidence': 0.8
            },
            
            # Sad: low pitch, low energy, slow tempo
            {
                'emotion': 'sad',
                'conditions': [
                    lambda f: f.get('pitch_mean', 0) < 120,
                    lambda f: f.get('energy_mean', 0) < 0.08,
                    lambda f: f.get('tempo', 0) < 100,
                    lambda f: f.get('pitch_variability', 0) < 0.1
                ],
                'confidence': 0.8
            },
            
            # Angry: high pitch, high energy, high variability
            {
                'emotion': 'angry',
                'conditions': [
                    lambda f: f.get('pitch_mean', 0) > 200,
                    lambda f: f.get('energy_mean', 0) > 0.2,
                    lambda f: f.get('pitch_variability', 0) > 0.3,
                    lambda f: f.get('energy_variability', 0) > 0.8
                ],
                'confidence': 0.8
            },
            
            # Calm: medium pitch, low energy, low variability
            {
                'emotion': 'calm',
                'conditions': [
                    lambda f: 120 <= f.get('pitch_mean', 0) <= 180,
                    lambda f: f.get('energy_mean', 0) < 0.12,
                    lambda f: f.get('pitch_variability', 0) < 0.15,
                    lambda f: f.get('energy_variability', 0) < 0.5
                ],
                'confidence': 0.8
            },
            
            # Excited: high pitch, high energy, fast tempo, high variability
            {
                'emotion': 'excited',
                'conditions': [
                    lambda f: f.get('pitch_mean', 0) > 190,
                    lambda f: f.get('energy_mean', 0) > 0.18,
                    lambda f: f.get('tempo', 0) > 150,
                    lambda f: f.get('pitch_variability', 0) > 0.25
                ],
                'confidence': 0.8
            },
            
            # Tired: low pitch, low energy, slow tempo, low variability
            {
                'emotion': 'tired',
                'conditions': [
                    lambda f: f.get('pitch_mean', 0) < 110,
                    lambda f: f.get('energy_mean', 0) < 0.06,
                    lambda f: f.get('tempo', 0) < 90,
                    lambda f: f.get('pitch_variability', 0) < 0.08
                ],
                'confidence': 0.8
            }
        ]
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion using rule-based heuristics.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        best_match = None
        best_confidence = 0.0
        
        for rule in self.rules:
            # Check if all conditions are met
            conditions_met = sum(1 for condition in rule['conditions'] if condition(features))
            total_conditions = len(rule['conditions'])
            
            if conditions_met >= total_conditions * 0.75:  # At least 75% of conditions met
                confidence = rule['confidence'] * (conditions_met / total_conditions)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = rule['emotion']
        
        # If no rule matches, return neutral
        if best_match is None:
            return 'neutral', 0.5
        
        return best_match, best_confidence
    
    def get_emotion_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over all emotions using rule-based approach.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Dictionary mapping emotions to probabilities
        """
        probabilities = {emotion: 0.0 for emotion in self.emotions}
        
        # Calculate confidence for each rule
        rule_confidences = {}
        for rule in self.rules:
            conditions_met = sum(1 for condition in rule['conditions'] if condition(features))
            total_conditions = len(rule['conditions'])
            
            if conditions_met >= total_conditions * 0.5:  # At least 50% of conditions met
                confidence = rule['confidence'] * (conditions_met / total_conditions)
                rule_confidences[rule['emotion']] = confidence
        
        # Normalize probabilities
        total_confidence = sum(rule_confidences.values())
        
        if total_confidence > 0:
            for emotion, confidence in rule_confidences.items():
                probabilities[emotion] = confidence / total_confidence
        else:
            # Default to neutral if no rules match
            probabilities['neutral'] = 1.0
        
        return probabilities


class MLClassifier(EmotionClassifier):
    """Machine learning-based emotion classifier."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML classifier.
        
        Args:
            model_path: Path to pre-trained model file
        """
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, features_list: List[Dict[str, float]], 
              labels: List[str], model_type: str = 'random_forest'):
        """
        Train the ML classifier.
        
        Args:
            features_list: List of feature dictionaries
            labels: List of emotion labels
            model_type: Type of model ('random_forest' or 'decision_tree')
        """
        # Convert features to feature matrix
        if not features_list:
            raise ValueError("No features provided for training")
        
        # Get feature names from first sample
        self.feature_names = list(features_list[0].keys())
        
        # Create feature matrix
        X = np.array([[features.get(feature, 0.0) for feature in self.feature_names] 
                     for features in features_list])
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_scaled, labels)
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion using trained ML model.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        
        X = np.array([[features.get(feature, 0.0) for feature in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get confidence
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def get_emotion_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution over all emotions using ML model.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Dictionary mapping emotions to probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        
        X = np.array([[features.get(feature, 0.0) for feature in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map to emotion names
        emotion_probs = {}
        for i, emotion in enumerate(self.model.classes_):
            emotion_probs[emotion] = probabilities[i]
        
        # Fill in missing emotions with 0 probability
        for emotion in self.emotions:
            if emotion not in emotion_probs:
                emotion_probs[emotion] = 0.0
        
        return emotion_probs
    
    def save_model(self, model_path: str):
        """
        Save the trained model to file.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'emotions': self.emotions
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.emotions = model_data.get('emotions', self.emotions)


class HybridClassifier(EmotionClassifier):
    """Hybrid classifier combining rule-based and ML approaches."""
    
    def __init__(self, ml_model_path: Optional[str] = None):
        """
        Initialize the hybrid classifier.
        
        Args:
            ml_model_path: Path to pre-trained ML model
        """
        super().__init__()
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLClassifier(ml_model_path) if ml_model_path else None
        
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion using hybrid approach.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Tuple of (emotion, confidence)
        """
        # Get rule-based classification
        rule_emotion, rule_confidence = self.rule_classifier.classify(features)
        
        # If ML model is available, combine predictions
        if self.ml_classifier is not None:
            try:
                ml_emotion, ml_confidence = self.ml_classifier.classify(features)
                
                # Weighted combination (favor ML if confidence is high)
                if ml_confidence > 0.7:
                    return ml_emotion, ml_confidence
                elif rule_confidence > 0.7:
                    return rule_emotion, rule_confidence
                else:
                    # Average the predictions
                    if rule_emotion == ml_emotion:
                        return rule_emotion, (rule_confidence + ml_confidence) / 2
                    else:
                        # Choose the one with higher confidence
                        return (ml_emotion, ml_confidence) if ml_confidence > rule_confidence else (rule_emotion, rule_confidence)
            except Exception:
                # Fall back to rule-based if ML fails
                return rule_emotion, rule_confidence
        
        # Fall back to rule-based only
        return rule_emotion, rule_confidence
    
    def get_emotion_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution using hybrid approach.
        
        Args:
            features: Dictionary of prosodic features
            
        Returns:
            Dictionary mapping emotions to probabilities
        """
        rule_probs = self.rule_classifier.get_emotion_probabilities(features)
        
        if self.ml_classifier is not None:
            try:
                ml_probs = self.ml_classifier.get_emotion_probabilities(features)
                
                # Combine probabilities (simple average)
                combined_probs = {}
                for emotion in self.emotions:
                    combined_probs[emotion] = (rule_probs.get(emotion, 0) + ml_probs.get(emotion, 0)) / 2
                
                return combined_probs
            except Exception:
                return rule_probs
        
        return rule_probs
