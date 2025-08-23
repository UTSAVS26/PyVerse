import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json
from datetime import datetime, timedelta

from music.playlist_builder import PlaylistBuilder


class TestPlaylistBuilder:
    """Test cases for PlaylistBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock music selector
        self.mock_music_selector = Mock()
        self.builder = PlaylistBuilder(self.mock_music_selector)
        self.sample_mood_data = {
            'happy': {'confidence': 0.8, 'timestamp': datetime.now()},
            'calm': {'confidence': 0.6, 'timestamp': datetime.now()},
            'excited': {'confidence': 0.9, 'timestamp': datetime.now()}
        }
        
    def test_initialization(self):
        """Test PlaylistBuilder initialization."""
        assert self.builder.mood_history == []
        assert self.builder.playlist_history == []
        assert self.builder.user_preferences == {}
        assert self.builder.adaptive_strategies == ['matching', 'modulating', 'enhancing']
        
    def test_build_adaptive_playlist_matching(self):
        """Test building adaptive playlist with matching strategy."""
        mood_data = {'emotion': 'happy', 'confidence': 0.8}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['song1', 'song2', 'song3']
            result = self.builder.build_adaptive_playlist(mood_data, strategy='matching')
            
            assert result == ['song1', 'song2', 'song3']
            mock_build.assert_called_once_with(mood_data, 'matching')
            
    def test_build_adaptive_playlist_modulating(self):
        """Test building adaptive playlist with modulating strategy."""
        mood_data = {'emotion': 'stressed', 'confidence': 0.7}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['calm_song1', 'calm_song2']
            result = self.builder.build_adaptive_playlist(mood_data, strategy='modulating')
            
            assert result == ['calm_song1', 'calm_song2']
            mock_build.assert_called_once_with(mood_data, 'modulating')
            
    def test_build_adaptive_playlist_invalid_strategy(self):
        """Test building adaptive playlist with invalid strategy."""
        mood_data = {'emotion': 'happy', 'confidence': 0.8}
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            self.builder.build_adaptive_playlist(mood_data, strategy='invalid')
            
    def test_build_adaptive_playlist_enhancing(self):
        """Test building adaptive playlist with enhancing strategy."""
        mood_data = {'emotion': 'excited', 'confidence': 0.9}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['energetic_song1', 'energetic_song2']
            result = self.builder.build_adaptive_playlist(mood_data, strategy='enhancing')
            
            assert result == ['energetic_song1', 'energetic_song2']
            mock_build.assert_called_once_with(mood_data, 'enhancing')
            
    def test_analyze_mood_trend(self):
        """Test mood trend analysis."""
        # Add some mood history
        self.builder.mood_history = [
            {'emotion': 'happy', 'confidence': 0.7, 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'emotion': 'excited', 'confidence': 0.8, 'timestamp': datetime.now() - timedelta(minutes=3)},
            {'emotion': 'excited', 'confidence': 0.9, 'timestamp': datetime.now() - timedelta(minutes=1)}
        ]
        
        trend = self.builder._analyze_mood_trend()
        assert 'trend' in trend
        assert 'stability' in trend
        assert 'dominant_emotion' in trend
        assert trend['trend'] in ['improving', 'declining', 'stable']
        
    def test_analyze_mood_trend_empty_history(self):
        """Test mood trend analysis with empty history."""
        trend = self.builder._analyze_mood_trend()
        assert trend['trend'] == 'stable'
        assert trend['stability'] == 1.0
        assert trend['dominant_emotion'] is None
        
    def test_apply_adaptive_modifications(self):
        """Test applying adaptive modifications to playlist."""
        base_playlist = ['song1', 'song2', 'song3']
        mood_data = {'emotion': 'stressed', 'confidence': 0.8}
        strategy = 'modulating'
        
        with patch('music.playlist_builder.PlaylistBuilder._insert_calming_songs') as mock_calm:
            mock_calm.return_value = ['calm_song1', 'song1', 'calm_song2', 'song2', 'song3']
            
            result = self.builder._apply_adaptive_modifications(base_playlist, mood_data, strategy)
            
            assert result == ['calm_song1', 'song1', 'calm_song2', 'song2', 'song3']
            mock_calm.assert_called_once_with(base_playlist, mood_data)
            
    def test_get_transition_songs(self):
        """Test getting transition songs."""
        current_mood = 'stressed'
        target_mood = 'calm'
        
        transitions = self.builder._get_transition_songs(current_mood, target_mood)
        assert isinstance(transitions, list)
        assert len(transitions) > 0
        
    def test_insert_transition_songs(self):
        """Test inserting transition songs into playlist."""
        playlist = ['song1', 'song2', 'song3']
        current_mood = 'stressed'
        target_mood = 'calm'
        
        with patch('music.playlist_builder.PlaylistBuilder._get_transition_songs') as mock_transitions:
            mock_transitions.return_value = ['transition1', 'transition2']
            
            result = self.builder._insert_transition_songs(playlist, current_mood, target_mood)
            
            assert 'transition1' in result
            assert 'transition2' in result
            assert len(result) > len(playlist)
            
    def test_insert_calming_songs(self):
        """Test inserting calming songs for stressed mood."""
        playlist = ['song1', 'song2', 'song3']
        mood_data = {'emotion': 'stressed', 'confidence': 0.8}
        
        result = self.builder._insert_calming_songs(playlist, mood_data)
        assert isinstance(result, list)
        assert len(result) >= len(playlist)
        
    def test_get_time_based_songs(self):
        """Test getting time-based songs."""
        current_time = datetime.now()
        
        songs = self.builder._get_time_based_songs(current_time)
        assert isinstance(songs, list)
        
    def test_insert_time_based_songs(self):
        """Test inserting time-based songs."""
        playlist = ['song1', 'song2', 'song3']
        current_time = datetime.now()
        
        result = self.builder._insert_time_based_songs(playlist, current_time)
        assert isinstance(result, list)
        assert len(result) >= len(playlist)
        
    def test_detect_stress_signals(self):
        """Test stress signal detection."""
        mood_history = [
            {'emotion': 'stressed', 'confidence': 0.8, 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'emotion': 'stressed', 'confidence': 0.9, 'timestamp': datetime.now() - timedelta(minutes=3)},
            {'emotion': 'stressed', 'confidence': 0.7, 'timestamp': datetime.now() - timedelta(minutes=1)}
        ]
        
        stress_level = self.builder._detect_stress_signals(mood_history)
        assert isinstance(stress_level, float)
        assert 0 <= stress_level <= 1
        
    def test_add_mood_entry(self):
        """Test adding mood entry to history."""
        mood_data = {'emotion': 'happy', 'confidence': 0.8}
        
        self.builder.add_mood_entry(mood_data)
        
        assert len(self.builder.mood_history) == 1
        assert self.builder.mood_history[0]['emotion'] == 'happy'
        assert self.builder.mood_history[0]['confidence'] == 0.8
        assert 'timestamp' in self.builder.mood_history[0]
        
    def test_add_playlist_entry(self):
        """Test adding playlist entry to history."""
        playlist_data = {
            'songs': ['song1', 'song2'],
            'strategy': 'matching',
            'mood': 'happy'
        }
        
        self.builder.add_playlist_entry(playlist_data)
        
        assert len(self.builder.playlist_history) == 1
        assert self.builder.playlist_history[0]['songs'] == ['song1', 'song2']
        assert self.builder.playlist_history[0]['strategy'] == 'matching'
        assert 'timestamp' in self.builder.playlist_history[0]
        
    def test_get_mood_statistics(self):
        """Test getting mood statistics."""
        # Add some mood history
        self.builder.mood_history = [
            {'emotion': 'happy', 'confidence': 0.7, 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'emotion': 'excited', 'confidence': 0.8, 'timestamp': datetime.now() - timedelta(minutes=3)},
            {'emotion': 'calm', 'confidence': 0.6, 'timestamp': datetime.now() - timedelta(minutes=1)}
        ]
        
        stats = self.builder.get_mood_statistics()
        
        assert 'total_entries' in stats
        assert 'emotion_distribution' in stats
        assert 'average_confidence' in stats
        assert 'mood_stability' in stats
        assert stats['total_entries'] == 3
        
    def test_get_personalized_playlist(self):
        """Test getting personalized playlist."""
        user_id = 'test_user'
        mood_data = {'emotion': 'happy', 'confidence': 0.8}
        
        with patch('music.playlist_builder.PlaylistBuilder.build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['personalized_song1', 'personalized_song2']
            
            result = self.builder.get_personalized_playlist(user_id, mood_data)
            
            assert result == ['personalized_song1', 'personalized_song2']
            mock_build.assert_called_once()
            
    def test_clear_history(self):
        """Test clearing mood and playlist history."""
        # Add some data
        self.builder.mood_history = [{'emotion': 'happy'}]
        self.builder.playlist_history = [{'songs': ['song1']}]
        
        self.builder.clear_history()
        
        assert self.builder.mood_history == []
        assert self.builder.playlist_history == []
        
    def test_build_adaptive_playlist_with_mood_history(self):
        """Test building adaptive playlist considering mood history."""
        # Add mood history
        self.builder.mood_history = [
            {'emotion': 'stressed', 'confidence': 0.8, 'timestamp': datetime.now() - timedelta(minutes=5)},
            {'emotion': 'stressed', 'confidence': 0.9, 'timestamp': datetime.now() - timedelta(minutes=3)}
        ]
        
        current_mood = {'emotion': 'stressed', 'confidence': 0.7}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['calm_song1', 'calm_song2']
            
            result = self.builder.build_adaptive_playlist(current_mood, strategy='modulating')
            
            assert result == ['calm_song1', 'calm_song2']
            
    def test_build_adaptive_playlist_edge_cases(self):
        """Test building adaptive playlist with edge cases."""
        # Test with very low confidence
        mood_data = {'emotion': 'happy', 'confidence': 0.1}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['default_song1']
            
            result = self.builder.build_adaptive_playlist(mood_data, strategy='matching')
            
            assert result == ['default_song1']
            
        # Test with very high confidence
        mood_data = {'emotion': 'excited', 'confidence': 0.99}
        
        with patch('music.playlist_builder.PlaylistBuilder._build_adaptive_playlist') as mock_build:
            mock_build.return_value = ['intense_song1', 'intense_song2']
            
            result = self.builder.build_adaptive_playlist(mood_data, strategy='enhancing')
            
            assert result == ['intense_song1', 'intense_song2']
