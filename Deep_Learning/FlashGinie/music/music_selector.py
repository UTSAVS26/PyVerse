"""
Music selector module for VoiceMoodMirror.
Selects and queues music based on mood analysis.
"""

import os
import random
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MusicSelector:
    """Selects music based on mood and user preferences."""
    
    def __init__(self, music_library_path: Optional[str] = None):
        """
        Initialize the music selector.
        
        Args:
            music_library_path: Path to music library directory
        """
        self.music_library_path = music_library_path
        self.music_database = self._initialize_music_database()
        self.current_playlist = []
        self.playlist_index = 0
        self.user_preferences = {}
        
    def _initialize_music_database(self) -> Dict[str, Dict]:
        """Initialize music database with mood tags."""
        # This is a sample database - in practice, you'd load from a file or API
        return {
            'happy_songs': {
                'tags': ['upbeat', 'cheerful', 'energetic', 'positive'],
                'songs': [
                    {'title': 'Happy', 'artist': 'Pharrell Williams', 'genre': 'pop'},
                    {'title': 'Walking on Sunshine', 'artist': 'Katrina & The Waves', 'genre': 'pop'},
                    {'title': 'Good Vibrations', 'artist': 'The Beach Boys', 'genre': 'rock'},
                    {'title': 'I Gotta Feeling', 'artist': 'The Black Eyed Peas', 'genre': 'pop'},
                    {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'genre': 'pop'}
                ]
            },
            'sad_songs': {
                'tags': ['melancholic', 'slow', 'emotional', 'reflective'],
                'songs': [
                    {'title': 'Mad World', 'artist': 'Gary Jules', 'genre': 'alternative'},
                    {'title': 'Hallelujah', 'artist': 'Jeff Buckley', 'genre': 'folk'},
                    {'title': 'Creep', 'artist': 'Radiohead', 'genre': 'alternative'},
                    {'title': 'The Scientist', 'artist': 'Coldplay', 'genre': 'alternative'},
                    {'title': 'Fix You', 'artist': 'Coldplay', 'genre': 'alternative'}
                ]
            },
            'calm_songs': {
                'tags': ['relaxing', 'peaceful', 'ambient', 'soft'],
                'songs': [
                    {'title': 'Weightless', 'artist': 'Marconi Union', 'genre': 'ambient'},
                    {'title': 'Claire de Lune', 'artist': 'Debussy', 'genre': 'classical'},
                    {'title': 'River Flows in You', 'artist': 'Yiruma', 'genre': 'piano'},
                    {'title': 'Gymnopedie No. 1', 'artist': 'Erik Satie', 'genre': 'classical'},
                    {'title': 'Spiegel im Spiegel', 'artist': 'Arvo Pärt', 'genre': 'classical'}
                ]
            },
            'energetic_songs': {
                'tags': ['energetic', 'fast', 'upbeat', 'dynamic'],
                'songs': [
                    {'title': 'Eye of the Tiger', 'artist': 'Survivor', 'genre': 'rock'},
                    {'title': 'We Will Rock You', 'artist': 'Queen', 'genre': 'rock'},
                    {'title': 'Thunderstruck', 'artist': 'AC/DC', 'genre': 'rock'},
                    {'title': 'Don\'t Stop Believin\'', 'artist': 'Journey', 'genre': 'rock'},
                    {'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses', 'genre': 'rock'}
                ]
            },
            'angry_songs': {
                'tags': ['intense', 'aggressive', 'powerful', 'dramatic'],
                'songs': [
                    {'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'genre': 'nu-metal'},
                    {'title': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'genre': 'rap-rock'},
                    {'title': 'Given Up', 'artist': 'Linkin Park', 'genre': 'nu-metal'},
                    {'title': 'Bodies', 'artist': 'Drowning Pool', 'genre': 'nu-metal'},
                    {'title': 'Du Hast', 'artist': 'Rammstein', 'genre': 'industrial'}
                ]
            },
            'tired_songs': {
                'tags': ['slow', 'relaxing', 'ambient', 'gentle'],
                'songs': [
                    {'title': 'Nocturne in C Minor', 'artist': 'Chopin', 'genre': 'classical'},
                    {'title': 'Moonlight Sonata', 'artist': 'Beethoven', 'genre': 'classical'},
                    {'title': 'Air on the G String', 'artist': 'Bach', 'genre': 'classical'},
                    {'title': 'Pavane', 'artist': 'Gabriel Fauré', 'genre': 'classical'},
                    {'title': 'Adagio for Strings', 'artist': 'Samuel Barber', 'genre': 'classical'}
                ]
            }
        }
    
    def select_music_for_mood(self, emotion: str, strategy: str = 'match', 
                            num_songs: int = 5) -> List[Dict]:
        """
        Select music based on emotion and strategy.
        
        Args:
            emotion: Current emotion
            strategy: 'match' to match mood, 'modulate' to change mood
            num_songs: Number of songs to select
            
        Returns:
            List of selected songs
        """
        if strategy == 'modulate':
            # Map emotions to target emotions for modulation
            modulation_map = {
                'sad': 'happy',
                'angry': 'calm',
                'tired': 'energetic',
                'excited': 'calm'
            }
            target_emotion = modulation_map.get(emotion, 'calm')
        else:
            target_emotion = emotion
        
        # Map emotions to music categories
        emotion_to_category = {
            'happy': 'happy_songs',
            'sad': 'sad_songs',
            'calm': 'calm_songs',
            'excited': 'energetic_songs',
            'angry': 'angry_songs',
            'tired': 'tired_songs',
            'neutral': 'calm_songs'
        }
        
        category = emotion_to_category.get(target_emotion)
        
        if category and category in self.music_database:
            available_songs = self.music_database[category]['songs']
            if num_songs <= 0:
                return []
            selected_songs = random.sample(available_songs, min(num_songs, len(available_songs)))
            return selected_songs
        
        return []
    
    def create_playlist(self, emotion: str, strategy: str = 'match', 
                       duration_minutes: int = 30) -> List[Dict]:
        """
        Create a playlist based on mood and target duration.
        
        Args:
            emotion: Current emotion
            strategy: 'match' or 'modulate'
            duration_minutes: Target playlist duration in minutes
            
        Returns:
            List of songs for the playlist
        """
        # Handle invalid duration
        if duration_minutes <= 0:
            return []
            
        # Estimate average song duration (3-4 minutes)
        avg_song_duration = 3.5
        target_songs = int(duration_minutes / avg_song_duration)
        
        # Get initial selection
        selected_songs = self.select_music_for_mood(emotion, strategy, target_songs)
        
        # Add variety by mixing in some neutral/calm songs
        if len(selected_songs) < target_songs and emotion != 'calm':
            additional_songs = self.select_music_for_mood('calm', 'match', 
                                                        target_songs - len(selected_songs))
            selected_songs.extend(additional_songs)
        
        # Shuffle the playlist
        random.shuffle(selected_songs)
        
        self.current_playlist = selected_songs
        self.playlist_index = 0
        
        return selected_songs
    
    def get_next_song(self) -> Optional[Dict]:
        """
        Get the next song from the current playlist.
        
        Args:
            Next song or None if playlist is empty
        """
        if not self.current_playlist:
            return None
        
        if self.playlist_index >= len(self.current_playlist):
            # Loop back to beginning
            self.playlist_index = 0
        
        song = self.current_playlist[self.playlist_index]
        self.playlist_index += 1
        
        return song
    
    def get_previous_song(self) -> Optional[Dict]:
        """
        Get the previous song from the current playlist.
        
        Args:
            Previous song or None if playlist is empty
        """
        if not self.current_playlist:
            return None
        
        self.playlist_index -= 1
        if self.playlist_index < 0:
            self.playlist_index = len(self.current_playlist) - 1
        
        return self.current_playlist[self.playlist_index]
    
    def search_music_by_tags(self, tags: List[str], num_results: int = 10) -> List[Dict]:
        """
        Search music by tags.
        
        Args:
            tags: List of tags to search for
            num_results: Maximum number of results
            
        Returns:
            List of matching songs
        """
        matching_songs = []
        
        for category, category_data in self.music_database.items():
            category_tags = category_data['tags']
            
            # Check if any of the search tags match category tags
            if any(tag.lower() in [ct.lower() for ct in category_tags] for tag in tags):
                matching_songs.extend(category_data['songs'])
        
        # Remove duplicates and limit results
        unique_songs = []
        seen_titles = set()
        
        for song in matching_songs:
            title_key = f"{song['title']}_{song['artist']}"
            if title_key not in seen_titles:
                unique_songs.append(song)
                seen_titles.add(title_key)
                
                if len(unique_songs) >= num_results:
                    break
        
        return unique_songs
    
    def add_user_preference(self, emotion: str, song: Dict, rating: float):
        """
        Add user preference for a song-emotion combination.
        
        Args:
            emotion: Emotion associated with the song
            song: Song dictionary
            rating: User rating (0.0 to 1.0)
        """
        if emotion not in self.user_preferences:
            self.user_preferences[emotion] = []
        
        song_key = f"{song['title']}_{song['artist']}"
        
        # Check if song already exists in preferences
        for pref in self.user_preferences[emotion]:
            if pref['song_key'] == song_key:
                pref['rating'] = rating
                return
        
        # Add new preference
        self.user_preferences[emotion].append({
            'song_key': song_key,
            'song': song,
            'rating': rating
        })
    
    def get_personalized_recommendations(self, emotion: str, num_songs: int = 5) -> List[Dict]:
        """
        Get personalized recommendations based on user preferences.
        
        Args:
            emotion: Current emotion
            num_songs: Number of songs to recommend
            
        Returns:
            List of recommended songs
        """
        if emotion not in self.user_preferences:
            return self.select_music_for_mood(emotion, 'match', num_songs)
        
        # Sort preferences by rating
        preferences = sorted(self.user_preferences[emotion], 
                           key=lambda x: x['rating'], reverse=True)
        
        # Get top-rated songs
        top_songs = [pref['song'] for pref in preferences[:num_songs]]
        
        # If we don't have enough personalized songs, fill with general recommendations
        if len(top_songs) < num_songs:
            general_songs = self.select_music_for_mood(emotion, 'match', 
                                                     num_songs - len(top_songs))
            top_songs.extend(general_songs)
        
        return top_songs
    
    def save_user_preferences(self, filepath: str):
        """
        Save user preferences to file.
        
        Args:
            filepath: Path to save preferences
        """
        with open(filepath, 'w') as f:
            json.dump(self.user_preferences, f, indent=2)
    
    def load_user_preferences(self, filepath: str):
        """
        Load user preferences from file.
        
        Args:
            filepath: Path to preferences file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.user_preferences = json.load(f)
    
    def get_playlist_info(self) -> Dict:
        """
        Get information about the current playlist.
        
        Returns:
            Dictionary with playlist information
        """
        if not self.current_playlist:
            return {
                'total_songs': 0,
                'current_index': 0,
                'current_song': None,
                'remaining_songs': 0
            }
        
        return {
            'total_songs': len(self.current_playlist),
            'current_index': self.playlist_index,
            'current_song': self.current_playlist[self.playlist_index - 1] if self.playlist_index > 0 else None,
            'remaining_songs': len(self.current_playlist) - self.playlist_index
        }
    
    def shuffle_playlist(self):
        """Shuffle the current playlist."""
        if self.current_playlist:
            random.shuffle(self.current_playlist)
            self.playlist_index = 0
    
    def clear_playlist(self):
        """Clear the current playlist."""
        self.current_playlist = []
        self.playlist_index = 0
