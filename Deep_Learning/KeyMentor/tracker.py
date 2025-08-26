"""
KeyMentor - Typing Data Tracker
Captures typing data in real time including keypress timings, mistakes, and typing speed.
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sqlite3
import os


@dataclass
class TypingEvent:
    """Represents a single typing event"""
    timestamp: float
    key: str
    expected_key: str
    is_correct: bool
    reaction_time: float
    session_id: str


@dataclass
class TypingSession:
    """Represents a complete typing session"""
    session_id: str
    start_time: float
    end_time: float
    total_keys: int
    correct_keys: int
    wpm: float
    accuracy: float
    text_content: str
    mistakes: List[Tuple[str, str]]  # (expected, actual)


class TypingTracker:
    """Tracks typing data in real-time"""
    
    def __init__(self, db_path: str = "typing_data.db"):
        self.db_path = db_path
        self.current_session: Optional[TypingSession] = None
        self.events: List[TypingEvent] = []
        self.session_start_time: Optional[float] = None
        self.last_key_time: Optional[float] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing typing data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS typing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                key TEXT,
                expected_key TEXT,
                is_correct BOOLEAN,
                reaction_time REAL
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS typing_sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_keys INTEGER,
                correct_keys INTEGER,
                wpm REAL,
                accuracy REAL,
                text_content TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new typing session"""
        if session_id is None:
            # Use microsecond precision and add a random component for uniqueness
            import random
            session_id = f"session_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"
        
        self.session_start_time = time.time()
        self.last_key_time = self.session_start_time
        self.events = []
        
        self.current_session = TypingSession(
            session_id=session_id,
            start_time=self.session_start_time,
            end_time=0,
            total_keys=0,
            correct_keys=0,
            wpm=0.0,
            accuracy=0.0,
            text_content="",
            mistakes=[]
        )
        
        return session_id
    
    def record_keypress(self, key: str, expected_key: str, text_content: str = "") -> TypingEvent:
        """Record a single keypress event"""
        if self.session_start_time is None:
            raise ValueError("No active session. Call start_session() first.")
        
        current_time = time.time()
        reaction_time = current_time - self.last_key_time if self.last_key_time else 0
        is_correct = key == expected_key
        
        event = TypingEvent(
            timestamp=current_time,
            key=key,
            expected_key=expected_key,
            is_correct=is_correct,
            reaction_time=reaction_time,
            session_id=self.current_session.session_id
        )
        
        self.events.append(event)
        self.current_session.total_keys += 1
        self.current_session.text_content = text_content
        
        if is_correct:
            self.current_session.correct_keys += 1
        else:
            self.current_session.mistakes.append((expected_key, key))
        
        self.last_key_time = current_time
        return event
    
    def end_session(self) -> TypingSession:
        """End the current typing session and calculate metrics"""
        if self.current_session is None:
            raise ValueError("No active session to end.")
        
        end_time = time.time()
        self.current_session.end_time = end_time
        
        # Calculate WPM (assuming average word length of 5 characters)
        session_duration = end_time - self.session_start_time
        if session_duration > 0:
            words_typed = len(self.current_session.text_content.split())
            self.current_session.wpm = (words_typed / session_duration) * 60
        
        # Calculate accuracy
        if self.current_session.total_keys > 0:
            self.current_session.accuracy = (self.current_session.correct_keys / 
                                           self.current_session.total_keys) * 100
        
        # Save to database
        self._save_session_to_db()
        
        return self.current_session
    
    def _save_session_to_db(self):
        """Save session and events to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save session
        cursor.execute('''
            INSERT INTO typing_sessions 
            (session_id, start_time, end_time, total_keys, correct_keys, wpm, accuracy, text_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session.session_id,
            self.current_session.start_time,
            self.current_session.end_time,
            self.current_session.total_keys,
            self.current_session.correct_keys,
            self.current_session.wpm,
            self.current_session.accuracy,
            self.current_session.text_content
        ))
        
        # Save events
        for event in self.events:
            cursor.execute('''
                INSERT INTO typing_events 
                (session_id, timestamp, key, expected_key, is_correct, reaction_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event.session_id,
                event.timestamp,
                event.key,
                event.expected_key,
                event.is_correct,
                event.reaction_time
            ))
        
        conn.commit()
        conn.close()
    
    def get_session_history(self, limit: int = 10) -> List[TypingSession]:
        """Retrieve recent typing sessions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, start_time, end_time, total_keys, correct_keys, wpm, accuracy, text_content
            FROM typing_sessions
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            session = TypingSession(
                session_id=row[0],
                start_time=row[1],
                end_time=row[2],
                total_keys=row[3],
                correct_keys=row[4],
                wpm=row[5],
                accuracy=row[6],
                text_content=row[7],
                mistakes=[]
            )
            sessions.append(session)
        
        conn.close()
        return sessions
    
    def get_typing_events(self, session_id: str) -> List[TypingEvent]:
        """Retrieve typing events for a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, timestamp, key, expected_key, is_correct, reaction_time
            FROM typing_events
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        events = []
        for row in cursor.fetchall():
            event = TypingEvent(
                timestamp=row[1],
                key=row[2],
                expected_key=row[3],
                is_correct=bool(row[4]),
                reaction_time=row[5],
                session_id=row[0]
            )
            events.append(event)
        
        conn.close()
        return events


# Convenience functions for easy usage
def create_tracker(db_path: str = "typing_data.db") -> TypingTracker:
    """Create a new typing tracker instance"""
    return TypingTracker(db_path)


def simulate_typing_session(tracker: TypingTracker, text: str, typing_speed: float = 0.1):
    """Simulate a typing session for testing purposes"""
    import random
    
    session_id = tracker.start_session()
    current_text = ""
    
    for i, expected_char in enumerate(text):
        # Simulate occasional mistakes
        if random.random() < 0.05:  # 5% error rate
            wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
            tracker.record_keypress(wrong_char, expected_char, current_text)
            current_text += wrong_char
        else:
            tracker.record_keypress(expected_char, expected_char, current_text)
            current_text += expected_char
        
        time.sleep(typing_speed)
    
    session = tracker.end_session()
    return session


if __name__ == "__main__":
    # Example usage
    tracker = create_tracker()
    
    # Simulate a typing session
    sample_text = "The quick brown fox jumps over the lazy dog."
    session = simulate_typing_session(tracker, sample_text)
    
    print(f"Session ID: {session.session_id}")
    print(f"WPM: {session.wpm:.2f}")
    print(f"Accuracy: {session.accuracy:.2f}%")
    print(f"Total keys: {session.total_keys}")
    print(f"Mistakes: {len(session.mistakes)}")
