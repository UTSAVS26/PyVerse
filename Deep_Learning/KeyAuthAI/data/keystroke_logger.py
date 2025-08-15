"""
Keystroke Logger Module for KeyAuthAI

This module captures keystroke dynamics including:
- Key press and release timing
- Dwell time (how long keys are held)
- Flight time (time between key releases and presses)
- Key sequences and patterns
"""

import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import os


class KeystrokeLogger:
    """Captures and stores keystroke dynamics data."""
    
    def __init__(self, data_file: str = "data/user_data.json"):
        """
        Initialize the keystroke logger.
        
        Args:
            data_file: Path to store user data
        """
        self.data_file = data_file
        self.current_session = []
        self.session_start_time = None
        self.key_press_times = {}
        self.key_release_times = {}
        self.listener = None
        self.is_recording = False
        
        # Ensure data directory exists
        # Ensure data directory exists
        dir_path = os.path.dirname(data_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load existing data
        self.user_data = self._load_user_data()
    
    def _load_user_data(self) -> Dict:
        """Load existing user data from file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return {}
    
    def _save_user_data(self):
        """Save user data to file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def start_recording(self, username: str, passphrase: str = "the quick brown fox jumps over the lazy dog"):
        """
        Start recording keystroke dynamics for a user.
        
        Args:
            username: Name of the user
            passphrase: Text to type for training
        """
        if self.is_recording:
            print("Already recording. Stop current session first.")
            return
        
        self.username = username
        self.passphrase = passphrase
        self.current_session = []
        self.session_start_time = time.time()
        self.key_press_times = {}
        self.key_release_times = {}
        self.is_recording = True
        
        print(f"Recording keystroke dynamics for user: {username}")
        print(f"Please type the following passphrase: '{passphrase}'")
        print("Press Enter to stop recording")
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
    
    def stop_recording(self) -> List[Dict]:
        """
        Stop recording and return the session data.
        
        Returns:
            List of keystroke events
        """
        if not self.is_recording:
            return []
        
        if self.listener:
            self.listener.stop()
            self.listener = None
        
        self.is_recording = False
        
        # Process and store the session data
        session_data = self._process_session_data()
        
        # Store in user data
        if self.username not in self.user_data:
            self.user_data[self.username] = {
                'sessions': [],
                'passphrase': self.passphrase,
                'created_at': time.time()
            }
        
        self.user_data[self.username]['sessions'].append({
            'timestamp': self.session_start_time,
            'data': session_data
        })
        
        self._save_user_data()
        
        print(f"Session recorded with {len(session_data)} keystroke events")
        return session_data
    
    def _on_key_press(self, key):
        """Handle key press events."""
        if not self.is_recording:
            return
        
        current_time = time.time()
        key_char = self._get_key_char(key)
        
        if key_char:
            self.key_press_times[key_char] = current_time
            
            event = {
                'type': 'press',
                'key': key_char,
                'timestamp': current_time,
                'relative_time': current_time - self.session_start_time
            }
            self.current_session.append(event)
    
    def _on_key_release(self, key):
        """Handle key release events."""
        if not self.is_recording:
            return
        
        current_time = time.time()
        key_char = self._get_key_char(key)
        
        if key_char and key_char in self.key_press_times:
            press_time = self.key_press_times[key_char]
            dwell_time = current_time - press_time
            
            event = {
                'type': 'release',
                'key': key_char,
                'timestamp': current_time,
                'relative_time': current_time - self.session_start_time,
                'dwell_time': dwell_time
            }
            self.current_session.append(event)
            
            # Calculate flight time to next key press
            if len(self.current_session) > 1:
                prev_event = self.current_session[-2]
                if prev_event['type'] == 'release':
                    flight_time = event['timestamp'] - prev_event['timestamp']
                    event['flight_time'] = flight_time
    
    def _get_key_char(self, key) -> Optional[str]:
        """Extract character from key event."""
        if hasattr(key, 'char') and key.char:
            return key.char
        elif isinstance(key, KeyCode):
            return key.char if key.char else None
        return None
    
    def _process_session_data(self) -> List[Dict]:
        """Process raw session data into structured format."""
        processed_data = []
        
        for i, event in enumerate(self.current_session):
            processed_event = {
                'index': i,
                'type': event['type'],
                'key': event['key'],
                'timestamp': event['timestamp'],
                'relative_time': event['relative_time']
            }
            
            if 'dwell_time' in event:
                processed_event['dwell_time'] = event['dwell_time']
            
            if 'flight_time' in event:
                processed_event['flight_time'] = event['flight_time']
            
            processed_data.append(processed_event)
        
        return processed_data
    
    def get_user_sessions(self, username: str) -> List[Dict]:
        """Get all sessions for a specific user."""
        if username in self.user_data:
            return self.user_data[username]['sessions']
        return []
    
    def get_user_passphrase(self, username: str) -> Optional[str]:
        """Get the passphrase for a specific user."""
        if username in self.user_data:
            return self.user_data[username]['passphrase']
        return None
    
    def delete_user_data(self, username: str):
        """Delete all data for a specific user."""
        if username in self.user_data:
            del self.user_data[username]
            self._save_user_data()
            print(f"Deleted all data for user: {username}")
    
    def list_users(self) -> List[str]:
        """Get list of all registered users."""
        return list(self.user_data.keys())


def record_keystroke_session(username: str, passphrase: str = None) -> List[Dict]:
    """
    Record a single keystroke session for a user.
    
    Args:
        username: Name of the user
        passphrase: Text to type (default: standard passphrase)
    
    Returns:
        List of keystroke events
    """
    if passphrase is None:
        passphrase = "the quick brown fox jumps over the lazy dog"
    
    logger = KeystrokeLogger()
    
    try:
        logger.start_recording(username, passphrase)
        input("Press Enter when you've finished typing the passphrase...")
    except KeyboardInterrupt:
        pass
    
    return logger.stop_recording()


if __name__ == "__main__":
    # Example usage
    logger = KeystrokeLogger()
    
    print("KeyAuthAI Keystroke Logger")
    print("=" * 30)
    
    username = input("Enter username: ")
    passphrase = input("Enter passphrase to type (or press Enter for default): ").strip()
    
    if not passphrase:
        passphrase = "the quick brown fox jumps over the lazy dog"
    
    print(f"\nPlease type: '{passphrase}'")
    print("Press Enter when done...")
    
    try:
        logger.start_recording(username, passphrase)
        input()
    except KeyboardInterrupt:
        pass
    
    session_data = logger.stop_recording()
    print(f"Recorded {len(session_data)} keystroke events") 