"""
Playlist builder module for VoiceMoodMirror.
Builds adaptive playlists based on mood and user preferences.
"""

import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PlaylistBuilder:
    """Builds adaptive playlists based on mood analysis and user preferences."""
    
    def __init__(self, music_selector):
        """
        Initialize the playlist builder.
        
        Args:
            music_selector: MusicSelector instance
        """
        self.music_selector = music_selector
        self.mood_history = []
        self.playlist_history = []
        
    def build_adaptive_playlist(self, current_emotion: str, 
                               target_duration: int = 30,
                               strategy: str = 'adaptive') -> List[Dict]:
        """
        Build an adaptive playlist based on current mood and history.
        
        Args:
            current_emotion: Current detected emotion
            target_duration: Target playlist duration in minutes
            strategy: Playlist strategy ('match', 'modulate', 'adaptive')
            
        Returns:
            List of songs for the playlist
        """
        if strategy == 'adaptive':
            return self._build_adaptive_playlist(current_emotion, target_duration)
        else:
            return self.music_selector.create_playlist(current_emotion, strategy, target_duration)
    
    def _build_adaptive_playlist(self, current_emotion: str, target_duration: int) -> List[Dict]:
        """
        Build an adaptive playlist that considers mood history and transitions.
        
        Args:
            current_emotion: Current detected emotion
            target_duration: Target playlist duration in minutes
            
        Returns:
            List of songs for the playlist
        """
        # Analyze mood history to determine if we need mood modulation
        mood_trend = self._analyze_mood_trend()
        
        # Determine playlist strategy based on mood trend
        if mood_trend == 'declining' and current_emotion in ['sad', 'angry', 'tired']:
            strategy = 'modulate'
        elif mood_trend == 'improving' and current_emotion in ['happy', 'excited']:
            strategy = 'match'
        else:
            strategy = 'match'
        
        # Get base playlist
        base_playlist = self.music_selector.create_playlist(current_emotion, strategy, target_duration)
        
        # Apply adaptive modifications
        adaptive_playlist = self._apply_adaptive_modifications(base_playlist, current_emotion, mood_trend)
        
        return adaptive_playlist
    
    def _analyze_mood_trend(self) -> str:
        """
        Analyze mood history to determine trend.
        
        Returns:
            Mood trend ('improving', 'declining', 'stable')
        """
        if len(self.mood_history) < 3:
            return 'stable'
        
        # Get recent mood scores
        recent_moods = self.mood_history[-5:]  # Last 5 mood entries
        
        # Convert emotions to numerical scores
        emotion_scores = {
            'happy': 5, 'excited': 4, 'calm': 3, 'neutral': 2, 'tired': 1, 'sad': 0, 'angry': 0
        }
        
        scores = [emotion_scores.get(mood['emotion'], 2) for mood in recent_moods]
        
        # Calculate trend
        if len(scores) >= 2:
            slope = (scores[-1] - scores[0]) / len(scores)
            
            if slope > 0.5:
                return 'improving'
            elif slope < -0.5:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'
    
    def _apply_adaptive_modifications(self, playlist: List[Dict], 
                                    current_emotion: str, 
                                    mood_trend: str) -> List[Dict]:
        """
        Apply adaptive modifications to the playlist.
        
        Args:
            playlist: Base playlist
            current_emotion: Current emotion
            mood_trend: Mood trend analysis
            
        Returns:
            Modified playlist
        """
        modified_playlist = playlist.copy()
        
        # Add mood transition songs if needed
        if mood_trend == 'declining' and current_emotion in ['sad', 'angry']:
            transition_songs = self._get_transition_songs(current_emotion, 'happy')
            modified_playlist = self._insert_transition_songs(modified_playlist, transition_songs)
        
        # Add calming songs for high-energy emotions if user seems stressed
        elif current_emotion in ['excited', 'angry'] and self._detect_stress_signals():
            calming_songs = self.music_selector.select_music_for_mood('calm', 'match', 2)
            modified_playlist = self._insert_calming_songs(modified_playlist, calming_songs)
        
        # Add variety based on time of day
        time_based_songs = self._get_time_based_songs()
        if time_based_songs:
            modified_playlist = self._insert_time_based_songs(modified_playlist, time_based_songs)
        
        return modified_playlist
    
    def _get_transition_songs(self, from_emotion: str, to_emotion: str) -> List[Dict]:
        """
        Get songs that help transition between emotions.
        
        Args:
            from_emotion: Starting emotion
            to_emotion: Target emotion
            
        Returns:
            List of transition songs
        """
        transition_mappings = {
            ('sad', 'happy'): ['upbeat', 'positive', 'energetic'],
            ('angry', 'calm'): ['relaxing', 'peaceful', 'soft'],
            ('tired', 'excited'): ['energetic', 'motivating', 'dynamic'],
            ('excited', 'calm'): ['relaxing', 'ambient', 'gentle']
        }
        
        key = (from_emotion, to_emotion)
        if key in transition_mappings:
            tags = transition_mappings[key]
            return self.music_selector.search_music_by_tags(tags, 3)
        
        return []
    
    def _insert_transition_songs(self, playlist: List[Dict], 
                               transition_songs: List[Dict]) -> List[Dict]:
        """
        Insert transition songs at strategic points in the playlist.
        
        Args:
            playlist: Original playlist
            transition_songs: Songs to insert
            
        Returns:
            Modified playlist
        """
        if not transition_songs:
            return playlist
        
        modified_playlist = playlist.copy()
        
        # Insert transition songs at 1/3 and 2/3 points
        insert_points = [len(playlist) // 3, 2 * len(playlist) // 3]
        
        for i, point in enumerate(insert_points):
            if i < len(transition_songs):
                modified_playlist.insert(point, transition_songs[i])
        
        return modified_playlist
    
    def _insert_calming_songs(self, playlist: List[Dict], 
                            calming_songs: List[Dict]) -> List[Dict]:
        """
        Insert calming songs into the playlist.
        
        Args:
            playlist: Original playlist
            calming_songs: Calming songs to insert
            
        Returns:
            Modified playlist
        """
        if not calming_songs:
            return playlist
        
        modified_playlist = playlist.copy()
        
        # Insert calming songs at the end
        for song in calming_songs:
            modified_playlist.append(song)
        
        return modified_playlist
    
    def _get_time_based_songs(self) -> List[Dict]:
        """
        Get songs based on time of day.
        
        Returns:
            List of time-appropriate songs
        """
        current_hour = datetime.now().hour
        
        if 6 <= current_hour < 12:
            # Morning: energetic, uplifting
            return self.music_selector.search_music_by_tags(['energetic', 'upbeat', 'positive'], 2)
        elif 12 <= current_hour < 18:
            # Afternoon: balanced, moderate
            return self.music_selector.search_music_by_tags(['balanced', 'moderate'], 2)
        elif 18 <= current_hour < 22:
            # Evening: relaxing, winding down
            return self.music_selector.search_music_by_tags(['relaxing', 'peaceful'], 2)
        else:
            # Night: very calm, ambient
            return self.music_selector.search_music_by_tags(['ambient', 'gentle', 'soothing'], 2)
    
    def _insert_time_based_songs(self, playlist: List[Dict], 
                               time_songs: List[Dict]) -> List[Dict]:
        """
        Insert time-based songs into the playlist.
        
        Args:
            playlist: Original playlist
            time_songs: Time-appropriate songs
            
        Returns:
            Modified playlist
        """
        if not time_songs:
            return playlist
        
        modified_playlist = playlist.copy()
        
        # Insert time-based songs at the beginning
        for song in reversed(time_songs):
            modified_playlist.insert(0, song)
        
        return modified_playlist
    
    def _detect_stress_signals(self) -> bool:
        """
        Detect if user shows signs of stress based on mood history.
        
        Returns:
            True if stress signals detected
        """
        if len(self.mood_history) < 3:
            return False
        
        # Check for rapid mood changes or sustained negative emotions
        recent_moods = self.mood_history[-3:]
        emotions = [mood['emotion'] for mood in recent_moods]
        
        # Stress indicators
        stress_emotions = ['angry', 'sad', 'tired']
        stress_count = sum(1 for emotion in emotions if emotion in stress_emotions)
        
        return stress_count >= 2
    
    def add_mood_entry(self, emotion: str, confidence: float, timestamp: Optional[datetime] = None):
        """
        Add a mood entry to the history.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence level
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.mood_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # Keep only last 50 entries
        if len(self.mood_history) > 50:
            self.mood_history = self.mood_history[-50:]
    
    def add_playlist_entry(self, playlist: List[Dict], emotion: str, 
                          strategy: str, timestamp: Optional[datetime] = None):
        """
        Add a playlist entry to the history.
        
        Args:
            playlist: Generated playlist
            emotion: Target emotion
            strategy: Used strategy
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.playlist_history.append({
            'playlist': playlist,
            'emotion': emotion,
            'strategy': strategy,
            'timestamp': timestamp
        })
        
        # Keep only last 20 entries
        if len(self.playlist_history) > 20:
            self.playlist_history = self.playlist_history[-20:]
    
    def get_mood_statistics(self) -> Dict:
        """
        Get statistics about mood history.
        
        Returns:
            Dictionary with mood statistics
        """
        if not self.mood_history:
            return {}
        
        emotions = [entry['emotion'] for entry in self.mood_history]
        confidences = [entry['confidence'] for entry in self.mood_history]
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # Get most common emotion
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'total_entries': len(self.mood_history),
            'emotion_counts': emotion_counts,
            'most_common_emotion': most_common_emotion,
            'average_confidence': avg_confidence,
            'mood_trend': self._analyze_mood_trend()
        }
    
    def get_personalized_playlist(self, current_emotion: str, 
                                target_duration: int = 30) -> List[Dict]:
        """
        Get a personalized playlist based on user history and preferences.
        
        Args:
            current_emotion: Current emotion
            target_duration: Target duration in minutes
            
        Returns:
            Personalized playlist
        """
        # Get personalized recommendations
        personalized_songs = self.music_selector.get_personalized_recommendations(
            current_emotion, target_duration // 3
        )
        
        # Get general recommendations
        general_songs = self.music_selector.select_music_for_mood(
            current_emotion, 'match', target_duration // 3
        )
        
        # Combine and shuffle
        combined_playlist = personalized_songs + general_songs
        random.shuffle(combined_playlist)
        
        # Apply adaptive modifications
        final_playlist = self._apply_adaptive_modifications(
            combined_playlist, current_emotion, self._analyze_mood_trend()
        )
        
        return final_playlist
    
    def clear_history(self):
        """Clear mood and playlist history."""
        self.mood_history = []
        self.playlist_history = []
