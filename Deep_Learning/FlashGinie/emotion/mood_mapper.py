"""
Mood mapper module for VoiceMoodMirror.
Maps inferred emotions to visual feedback and music recommendations.
"""

from typing import Dict, List, Tuple, Optional
import colorsys


class MoodMapper:
    """Maps emotions to visual feedback and music recommendations."""
    
    def __init__(self):
        """Initialize the mood mapper."""
        self.emotion_colors = self._initialize_emotion_colors()
        self.emotion_music_tags = self._initialize_music_tags()
        self.emotion_emojis = self._initialize_emotion_emojis()
        self.emotion_descriptions = self._initialize_emotion_descriptions()
        
    def _initialize_emotion_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Initialize color mappings for emotions."""
        return {
            'happy': (255, 255, 0),      # Bright yellow
            'sad': (0, 0, 255),          # Blue
            'angry': (255, 0, 0),        # Red
            'calm': (0, 255, 255),       # Cyan
            'excited': (255, 165, 0),    # Orange
            'tired': (128, 128, 128),    # Gray
            'neutral': (255, 255, 255)   # White
        }
    
    def _initialize_music_tags(self) -> Dict[str, List[str]]:
        """Initialize music tag mappings for emotions."""
        return {
            'happy': ['upbeat', 'cheerful', 'energetic', 'positive', 'uplifting'],
            'sad': ['melancholic', 'slow', 'emotional', 'reflective', 'calm'],
            'angry': ['intense', 'aggressive', 'powerful', 'energetic', 'dramatic'],
            'calm': ['relaxing', 'peaceful', 'ambient', 'soft', 'gentle'],
            'excited': ['energetic', 'fast', 'upbeat', 'dynamic', 'powerful'],
            'tired': ['slow', 'relaxing', 'ambient', 'gentle', 'soothing'],
            'neutral': ['balanced', 'moderate', 'pleasant', 'smooth', 'easy']
        }
    
    def _initialize_emotion_emojis(self) -> Dict[str, str]:
        """Initialize emoji mappings for emotions."""
        return {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'calm': '😌',
            'excited': '🤩',
            'tired': '😴',
            'neutral': '😐'
        }
    
    def _initialize_emotion_descriptions(self) -> Dict[str, str]:
        """Initialize description mappings for emotions."""
        return {
            'happy': 'You sound cheerful and upbeat!',
            'sad': 'You seem a bit down. Maybe some uplifting music would help?',
            'angry': 'You sound frustrated. How about some calming music?',
            'calm': 'You have a peaceful, relaxed tone.',
            'excited': 'You sound enthusiastic and energetic!',
            'tired': 'You seem tired. Some gentle music might help you relax.',
            'neutral': 'You have a balanced, neutral tone.'
        }
    
    def get_emotion_color(self, emotion: str, confidence: float = 1.0) -> Tuple[int, int, int]:
        """
        Get color for an emotion with confidence-based intensity.
        
        Args:
            emotion: Emotion name
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            RGB color tuple
        """
        base_color = self.emotion_colors.get(emotion, self.emotion_colors['neutral'])
        
        # Adjust intensity based on confidence
        if confidence < 1.0:
            # Blend with neutral color
            neutral_color = self.emotion_colors['neutral']
            r = int(base_color[0] * confidence + neutral_color[0] * (1 - confidence))
            g = int(base_color[1] * confidence + neutral_color[1] * (1 - confidence))
            b = int(base_color[2] * confidence + neutral_color[2] * (1 - confidence))
            return (r, g, b)
        
        return base_color
    
    def get_emotion_gradient(self, emotion: str, steps: int = 10) -> List[Tuple[int, int, int]]:
        """
        Get color gradient for an emotion.
        
        Args:
            emotion: Emotion name
            steps: Number of gradient steps
            
        Returns:
            List of RGB color tuples
        """
        base_color = self.emotion_colors.get(emotion, self.emotion_colors['neutral'])
        neutral_color = self.emotion_colors['neutral']
        
        gradient = []
        for i in range(steps):
            t = i / (steps - 1)
            r = int(base_color[0] * t + neutral_color[0] * (1 - t))
            g = int(base_color[1] * t + neutral_color[1] * (1 - t))
            b = int(base_color[2] * t + neutral_color[2] * (1 - t))
            gradient.append((r, g, b))
        
        return gradient
    
    def get_music_tags(self, emotion: str, num_tags: int = 3) -> List[str]:
        """
        Get music tags for an emotion.
        
        Args:
            emotion: Emotion name
            num_tags: Number of tags to return
            
        Returns:
            List of music tags
        """
        tags = self.emotion_music_tags.get(emotion, self.emotion_music_tags['neutral'])
        return tags[:num_tags]
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """
        Get emoji for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Emoji string
        """
        return self.emotion_emojis.get(emotion, self.emotion_emojis['neutral'])
    
    def get_emotion_description(self, emotion: str) -> str:
        """
        Get description for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Description string
        """
        return self.emotion_descriptions.get(emotion, self.emotion_descriptions['neutral'])
    
    def get_mood_enhancement_suggestions(self, emotion: str) -> List[str]:
        """
        Get suggestions for mood enhancement based on emotion.
        
        Args:
            emotion: Current emotion
            
        Returns:
            List of suggestions
        """
        suggestions = {
            'happy': [
                "Keep up the positive energy!",
                "Share your joy with others",
                "Try some upbeat music to maintain the mood"
            ],
            'sad': [
                "Consider listening to uplifting music",
                "Take a short walk to clear your mind",
                "Talk to a friend or loved one"
            ],
            'angry': [
                "Try some calming music to help relax",
                "Take deep breaths and count to ten",
                "Consider physical exercise to release tension"
            ],
            'calm': [
                "Enjoy this peaceful state",
                "Meditation or gentle music could enhance this feeling",
                "Perfect time for reflection or creative activities"
            ],
            'excited': [
                "Channel this energy into something productive",
                "Try some energetic music to match your mood",
                "Consider physical activity to burn off energy"
            ],
            'tired': [
                "Rest is important - consider taking a break",
                "Gentle, soothing music might help you relax",
                "Stay hydrated and get some fresh air"
            ],
            'neutral': [
                "A balanced state - good for focused work",
                "Consider what might help you feel more energized or relaxed",
                "This is a good baseline for emotional stability"
            ]
        }
        
        return suggestions.get(emotion, suggestions['neutral'])
    
    def get_mood_transition_suggestions(self, current_emotion: str, target_emotion: str) -> List[str]:
        """
        Get suggestions for transitioning from one mood to another.
        
        Args:
            current_emotion: Current emotion
            target_emotion: Target emotion
            
        Returns:
            List of transition suggestions
        """
        transitions = {
            ('sad', 'happy'): [
                "Listen to upbeat, cheerful music",
                "Think about positive memories or future plans",
                "Engage in activities you enjoy"
            ],
            ('angry', 'calm'): [
                "Try deep breathing exercises",
                "Listen to calming, ambient music",
                "Take a moment to step back and reflect"
            ],
            ('tired', 'excited'): [
                "Listen to energetic, motivating music",
                "Get some physical activity or movement",
                "Consider what excites or interests you"
            ],
            ('excited', 'calm'): [
                "Try some relaxing, peaceful music",
                "Practice mindfulness or meditation",
                "Engage in quiet, focused activities"
            ]
        }
        
        key = (current_emotion, target_emotion)
        if key in transitions:
            return transitions[key]
        
        # Default suggestions
        return [
            f"Consider what might help you transition from {current_emotion} to {target_emotion}",
            "Music can be a powerful tool for mood regulation",
            "Small changes in environment or activity can make a big difference"
        ]
    
    def get_visual_feedback_config(self, emotion: str, confidence: float) -> Dict:
        """
        Get complete visual feedback configuration for an emotion.
        
        Args:
            emotion: Emotion name
            confidence: Confidence level
            
        Returns:
            Dictionary with visual feedback configuration
        """
        return {
            'color': self.get_emotion_color(emotion, confidence),
            'gradient': self.get_emotion_gradient(emotion),
            'emoji': self.get_emotion_emoji(emotion),
            'description': self.get_emotion_description(emotion),
            'confidence': confidence,
            'intensity': self._calculate_intensity(confidence),
            'animation_speed': self._get_animation_speed(emotion, confidence)
        }
    
    def _calculate_intensity(self, confidence: float) -> float:
        """Calculate intensity level based on confidence."""
        # Clamp confidence to [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        return min(confidence * 1.5, 1.0)  # Scale confidence to intensity
    
    def _get_animation_speed(self, emotion: str, confidence: float) -> float:
        """Get animation speed based on emotion and confidence."""
        base_speeds = {
            'happy': 1.2,
            'sad': 0.6,
            'angry': 1.5,
            'calm': 0.8,
            'excited': 1.4,
            'tired': 0.5,
            'neutral': 1.0
        }
        
        base_speed = base_speeds.get(emotion, 1.0)
        return base_speed * confidence
    
    def get_music_recommendation_config(self, emotion: str, 
                                      strategy: str = 'match') -> Dict:
        """
        Get music recommendation configuration.
        
        Args:
            emotion: Current emotion
            strategy: 'match' to match mood, 'modulate' to change mood
            
        Returns:
            Dictionary with music recommendation configuration
        """
        if strategy == 'modulate':
            # Suggest music to change the mood
            modulation_map = {
                'sad': 'happy',
                'angry': 'calm',
                'tired': 'excited',
                'excited': 'calm'
            }
            target_emotion = modulation_map.get(emotion, 'calm')  # Default to calm
            tags = self.get_music_tags(target_emotion)
            description = f"Music to help you feel more {target_emotion}"
        else:
            # Match the current mood
            target_emotion = emotion
            tags = self.get_music_tags(emotion)
            description = f"Music that matches your {emotion} mood"
        
        return {
            'tags': tags,
            'description': description,
            'strategy': strategy,
            'target_emotion': target_emotion
        }
