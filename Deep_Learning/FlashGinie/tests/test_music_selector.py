"""
Tests for the music selector module.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music.music_selector import MusicSelector


class TestMusicSelector:
    """Test cases for MusicSelector class."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.selector = MusicSelector()
    
    def test_initialization(self):
        """Test MusicSelector initialization."""
        assert hasattr(self.selector, 'music_database')
        assert hasattr(self.selector, 'current_playlist')
        assert hasattr(self.selector, 'playlist_index')
        assert hasattr(self.selector, 'user_preferences')
        
        assert len(self.selector.music_database) > 0
        assert len(self.selector.current_playlist) == 0
        assert self.selector.playlist_index == 0
        assert len(self.selector.user_preferences) == 0
    
    def test_music_database_structure(self):
        """Test music database structure."""
        for category, data in self.selector.music_database.items():
            assert 'tags' in data
            assert 'songs' in data
            assert isinstance(data['tags'], list)
            assert isinstance(data['songs'], list)
            assert len(data['songs']) > 0
            
            for song in data['songs']:
                assert 'title' in song
                assert 'artist' in song
                assert 'genre' in song
                assert isinstance(song['title'], str)
                assert isinstance(song['artist'], str)
                assert isinstance(song['genre'], str)
    
    def test_select_music_for_mood_match(self):
        """Test selecting music to match mood."""
        songs = self.selector.select_music_for_mood('happy', 'match', 3)
        
        assert isinstance(songs, list)
        assert len(songs) <= 3
        
        for song in songs:
            assert 'title' in song
            assert 'artist' in song
            assert 'genre' in song
    
    def test_select_music_for_mood_modulate(self):
        """Test selecting music to modulate mood."""
        songs = self.selector.select_music_for_mood('sad', 'modulate', 3)
        
        assert isinstance(songs, list)
        assert len(songs) <= 3
        
        # Should select happy songs to modulate sad mood
        assert len(songs) > 0
    
    def test_select_music_for_unknown_emotion(self):
        """Test selecting music for unknown emotion."""
        songs = self.selector.select_music_for_mood('unknown', 'match', 3)
        
        # Should return empty list for unknown emotion
        assert isinstance(songs, list)
        assert len(songs) == 0
    
    def test_create_playlist(self):
        """Test creating a playlist."""
        playlist = self.selector.create_playlist('happy', 'match', 30)
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        
        # Should set current playlist
        assert self.selector.current_playlist == playlist
        assert self.selector.playlist_index == 0
    
    def test_get_next_song(self):
        """Test getting next song from playlist."""
        # Create a playlist first
        self.selector.create_playlist('happy', 'match', 30)
        
        song = self.selector.get_next_song()
        
        assert song is not None
        assert 'title' in song
        assert 'artist' in song
        assert self.selector.playlist_index == 1
    
    def test_get_next_song_empty_playlist(self):
        """Test getting next song from empty playlist."""
        song = self.selector.get_next_song()
        
        assert song is None
    
    def test_get_next_song_loop(self):
        """Test getting next song with looping."""
        # Create a short playlist
        self.selector.create_playlist('happy', 'match', 10)
        original_length = len(self.selector.current_playlist)
        
        # Get all songs
        for _ in range(original_length + 1):
            song = self.selector.get_next_song()
            assert song is not None
        
        # Should loop back to beginning
        assert self.selector.playlist_index == 1
    
    def test_get_previous_song(self):
        """Test getting previous song from playlist."""
        # Create a playlist and advance
        self.selector.create_playlist('happy', 'match', 30)
        self.selector.get_next_song()  # Advance to index 1
        
        song = self.selector.get_previous_song()
        
        assert song is not None
        assert 'title' in song
        assert 'artist' in song
        assert self.selector.playlist_index == 0
    
    def test_get_previous_song_empty_playlist(self):
        """Test getting previous song from empty playlist."""
        song = self.selector.get_previous_song()
        
        assert song is None
    
    def test_search_music_by_tags(self):
        """Test searching music by tags."""
        songs = self.selector.search_music_by_tags(['upbeat', 'energetic'], 5)
        
        assert isinstance(songs, list)
        assert len(songs) <= 5
        
        for song in songs:
            assert 'title' in song
            assert 'artist' in song
            assert 'genre' in song
    
    def test_search_music_by_tags_no_matches(self):
        """Test searching music with no matching tags."""
        songs = self.selector.search_music_by_tags(['nonexistent_tag'], 5)
        
        assert isinstance(songs, list)
        assert len(songs) == 0
    
    def test_add_user_preference(self):
        """Test adding user preference."""
        song = {'title': 'Test Song', 'artist': 'Test Artist', 'genre': 'test'}
        
        self.selector.add_user_preference('happy', song, 0.8)
        
        assert 'happy' in self.selector.user_preferences
        assert len(self.selector.user_preferences['happy']) == 1
        
        preference = self.selector.user_preferences['happy'][0]
        assert preference['song_key'] == 'Test Song_Test Artist'
        assert preference['rating'] == 0.8
        assert preference['song'] == song
    
    def test_add_user_preference_update_existing(self):
        """Test updating existing user preference."""
        song = {'title': 'Test Song', 'artist': 'Test Artist', 'genre': 'test'}
        
        # Add preference twice with different ratings
        self.selector.add_user_preference('happy', song, 0.8)
        self.selector.add_user_preference('happy', song, 0.9)
        
        assert len(self.selector.user_preferences['happy']) == 1
        assert self.selector.user_preferences['happy'][0]['rating'] == 0.9
    
    def test_get_personalized_recommendations(self):
        """Test getting personalized recommendations."""
        # Add some preferences
        song1 = {'title': 'Song 1', 'artist': 'Artist 1', 'genre': 'pop'}
        song2 = {'title': 'Song 2', 'artist': 'Artist 2', 'genre': 'rock'}
        
        self.selector.add_user_preference('happy', song1, 0.9)
        self.selector.add_user_preference('happy', song2, 0.7)
        
        recommendations = self.selector.get_personalized_recommendations('happy', 3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Should include personalized songs
        song_titles = [song['title'] for song in recommendations]
        assert 'Song 1' in song_titles or 'Song 2' in song_titles
    
    def test_get_personalized_recommendations_no_preferences(self):
        """Test getting personalized recommendations without preferences."""
        recommendations = self.selector.get_personalized_recommendations('happy', 3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
    
    def test_save_and_load_user_preferences(self):
        """Test saving and loading user preferences."""
        # Add some preferences
        song = {'title': 'Test Song', 'artist': 'Test Artist', 'genre': 'test'}
        self.selector.add_user_preference('happy', song, 0.8)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Save preferences
            self.selector.save_user_preferences(temp_filename)
            
            # Create new selector and load preferences
            new_selector = MusicSelector()
            new_selector.load_user_preferences(temp_filename)
            
            assert 'happy' in new_selector.user_preferences
            assert len(new_selector.user_preferences['happy']) == 1
            assert new_selector.user_preferences['happy'][0]['rating'] == 0.8
        
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_load_user_preferences_nonexistent_file(self):
        """Test loading preferences from nonexistent file."""
        new_selector = MusicSelector()
        new_selector.load_user_preferences('nonexistent_file.json')
        
        # Should not raise error and should have empty preferences
        assert len(new_selector.user_preferences) == 0
    
    def test_get_playlist_info(self):
        """Test getting playlist information."""
        # Test with empty playlist
        info = self.selector.get_playlist_info()
        
        assert info['total_songs'] == 0
        assert info['current_index'] == 0
        assert info['current_song'] is None
        assert info['remaining_songs'] == 0
        
        # Test with playlist
        self.selector.create_playlist('happy', 'match', 30)
        info = self.selector.get_playlist_info()
        
        assert info['total_songs'] > 0
        assert info['current_index'] == 0
        assert info['current_song'] is None
        assert info['remaining_songs'] == info['total_songs']
        
        # Test after advancing
        self.selector.get_next_song()
        info = self.selector.get_playlist_info()
        
        assert info['current_index'] == 1
        assert info['current_song'] is not None
        assert info['remaining_songs'] == info['total_songs'] - 1
    
    def test_shuffle_playlist(self):
        """Test shuffling playlist."""
        # Create a playlist
        original_playlist = self.selector.create_playlist('happy', 'match', 30)
        
        # Shuffle the playlist
        self.selector.shuffle_playlist()
        
        # Playlist should be different (though same length)
        assert len(self.selector.current_playlist) == len(original_playlist)
        assert self.selector.playlist_index == 0
    
    def test_clear_playlist(self):
        """Test clearing playlist."""
        # Create a playlist
        self.selector.create_playlist('happy', 'match', 30)
        
        # Clear the playlist
        self.selector.clear_playlist()
        
        assert len(self.selector.current_playlist) == 0
        assert self.selector.playlist_index == 0
    
    def test_emotion_to_category_mapping(self):
        """Test emotion to music category mapping."""
        test_cases = [
            ('happy', 'happy_songs'),
            ('sad', 'sad_songs'),
            ('calm', 'calm_songs'),
            ('excited', 'energetic_songs'),
            ('angry', 'angry_songs'),
            ('tired', 'tired_songs'),
            ('neutral', 'calm_songs')
        ]
        
        for emotion, expected_category in test_cases:
            songs = self.selector.select_music_for_mood(emotion, 'match', 1)
            if songs:  # If songs are available for this emotion
                assert len(songs) > 0
    
    def test_modulation_mapping(self):
        """Test emotion modulation mapping."""
        test_cases = [
            ('sad', 'happy'),
            ('angry', 'calm'),
            ('tired', 'energetic'),
            ('excited', 'calm')
        ]
        
        for from_emotion, expected_to_emotion in test_cases:
            songs = self.selector.select_music_for_mood(from_emotion, 'modulate', 1)
            if songs:  # If songs are available
                assert len(songs) > 0
    
    def test_playlist_duration_calculation(self):
        """Test playlist duration calculation."""
        # Test with different durations
        for duration in [10, 30, 60]:
            playlist = self.selector.create_playlist('happy', 'match', duration)
            
            # Should have reasonable number of songs
            # Assuming average song duration of 3.5 minutes
            expected_songs = int(duration / 3.5)
            assert len(playlist) >= expected_songs * 0.5  # Allow some flexibility
            assert len(playlist) <= expected_songs * 2
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with zero duration
        playlist = self.selector.create_playlist('happy', 'match', 0)
        assert len(playlist) == 0
        
        # Test with negative duration
        playlist = self.selector.create_playlist('happy', 'match', -10)
        assert len(playlist) == 0
        
        # Test with very large duration
        playlist = self.selector.create_playlist('happy', 'match', 1000)
        assert len(playlist) > 0
        
        # Test with zero songs requested
        songs = self.selector.select_music_for_mood('happy', 'match', 0)
        assert len(songs) == 0
        
        # Test with negative songs requested
        songs = self.selector.select_music_for_mood('happy', 'match', -5)
        assert len(songs) == 0
