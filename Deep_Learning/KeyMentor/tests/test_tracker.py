"""
Tests for the tracker module
"""

import unittest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from tracker import TypingTracker, TypingEvent, TypingSession, create_tracker, simulate_typing_session


class TestTypingTracker(unittest.TestCase):
    """Test cases for TypingTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary database file for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = TypingTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary database file
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_init_database(self):
        """Test database initialization"""
        # Check that database file exists
        self.assertTrue(os.path.exists(self.temp_db.name))
        
        # Check that tables are created
        import sqlite3
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check typing_events table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='typing_events'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check typing_sessions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='typing_sessions'")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    def test_start_session(self):
        """Test starting a new typing session"""
        session_id = self.tracker.start_session()
        
        # Check that session_id is generated
        self.assertIsInstance(session_id, str)
        self.assertTrue(session_id.startswith("session_"))
        
        # Check that current_session is set
        self.assertIsNotNone(self.tracker.current_session)
        self.assertEqual(self.tracker.current_session.session_id, session_id)
        self.assertEqual(self.tracker.current_session.total_keys, 0)
        self.assertEqual(self.tracker.current_session.correct_keys, 0)
    
    def test_start_session_with_custom_id(self):
        """Test starting a session with custom ID"""
        custom_id = "test_session_123"
        session_id = self.tracker.start_session(custom_id)
        
        self.assertEqual(session_id, custom_id)
        self.assertEqual(self.tracker.current_session.session_id, custom_id)
    
    def test_record_keypress(self):
        """Test recording keypress events"""
        self.tracker.start_session()
        
        # Record a correct keypress
        event = self.tracker.record_keypress('a', 'a', "a")
        
        # Check event properties
        self.assertIsInstance(event, TypingEvent)
        self.assertEqual(event.key, 'a')
        self.assertEqual(event.expected_key, 'a')
        self.assertTrue(event.is_correct)
        self.assertEqual(self.tracker.current_session.total_keys, 1)
        self.assertEqual(self.tracker.current_session.correct_keys, 1)
        
        # Record an incorrect keypress
        event2 = self.tracker.record_keypress('b', 'a', "ab")
        
        self.assertEqual(event2.key, 'b')
        self.assertEqual(event2.expected_key, 'a')
        self.assertFalse(event2.is_correct)
        self.assertEqual(self.tracker.current_session.total_keys, 2)
        self.assertEqual(self.tracker.current_session.correct_keys, 1)
        self.assertEqual(len(self.tracker.current_session.mistakes), 1)
    
    def test_record_keypress_no_session(self):
        """Test recording keypress without active session"""
        with self.assertRaises(ValueError):
            self.tracker.record_keypress('a', 'a', "a")
    
    def test_end_session(self):
        """Test ending a typing session"""
        self.tracker.start_session()
        
        # Add some keypresses
        self.tracker.record_keypress('h', 'h', "h")
        self.tracker.record_keypress('e', 'e', "he")
        self.tracker.record_keypress('l', 'l', "hel")
        self.tracker.record_keypress('l', 'l', "hell")
        self.tracker.record_keypress('o', 'o', "hello")
        
        # End session
        session = self.tracker.end_session()
        
        # Check session properties
        self.assertIsInstance(session, TypingSession)
        self.assertEqual(session.total_keys, 5)
        self.assertEqual(session.correct_keys, 5)
        self.assertGreater(session.wpm, 0)
        self.assertEqual(session.accuracy, 100.0)
        self.assertGreater(session.end_time, session.start_time)
    
    def test_end_session_no_session(self):
        """Test ending session without active session"""
        with self.assertRaises(ValueError):
            self.tracker.end_session()
    
    def test_wpm_calculation(self):
        """Test WPM calculation"""
        self.tracker.start_session()
        
        # Type "hello world" (2 words)
        text = "hello world"
        for i, char in enumerate(text):
            self.tracker.record_keypress(char, char, text[:i+1])
        
        session = self.tracker.end_session()
        
        # WPM should be calculated based on time taken
        self.assertGreater(session.wpm, 0)
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        self.tracker.start_session()
        
        # Type with some mistakes
        self.tracker.record_keypress('h', 'h', "h")      # correct
        self.tracker.record_keypress('e', 'e', "he")     # correct
        self.tracker.record_keypress('x', 'l', "hex")    # mistake
        self.tracker.record_keypress('l', 'l', "hexl")   # correct
        self.tracker.record_keypress('o', 'o', "hexlo")  # correct
        
        session = self.tracker.end_session()
        
        # 4 correct out of 5 total = 80% accuracy
        self.assertEqual(session.accuracy, 80.0)
    
    def test_get_session_history(self):
        """Test retrieving session history"""
        # Create multiple sessions
        for i in range(3):
            self.tracker.start_session()
            self.tracker.record_keypress('a', 'a', "a")
            self.tracker.end_session()
        
        # Get history
        sessions = self.tracker.get_session_history(5)
        
        self.assertEqual(len(sessions), 3)
        self.assertIsInstance(sessions[0], TypingSession)
    
    def test_get_typing_events(self):
        """Test retrieving typing events"""
        session_id = self.tracker.start_session()
        
        # Record some events
        self.tracker.record_keypress('a', 'a', "a")
        self.tracker.record_keypress('b', 'b', "ab")
        
        self.tracker.end_session()
        
        # Get events
        events = self.tracker.get_typing_events(session_id)
        
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], TypingEvent)
        self.assertEqual(events[0].key, 'a')
        self.assertEqual(events[1].key, 'b')


class TestTypingEvent(unittest.TestCase):
    """Test cases for TypingEvent dataclass"""
    
    def test_typing_event_creation(self):
        """Test creating TypingEvent instances"""
        event = TypingEvent(
            timestamp=1234567890.0,
            key='a',
            expected_key='a',
            is_correct=True,
            reaction_time=0.1,
            session_id='test_session'
        )
        
        self.assertEqual(event.timestamp, 1234567890.0)
        self.assertEqual(event.key, 'a')
        self.assertEqual(event.expected_key, 'a')
        self.assertTrue(event.is_correct)
        self.assertEqual(event.reaction_time, 0.1)
        self.assertEqual(event.session_id, 'test_session')


class TestTypingSession(unittest.TestCase):
    """Test cases for TypingSession dataclass"""
    
    def test_typing_session_creation(self):
        """Test creating TypingSession instances"""
        session = TypingSession(
            session_id='test_session',
            start_time=1234567890.0,
            end_time=1234567895.0,
            total_keys=10,
            correct_keys=8,
            wpm=60.0,
            accuracy=80.0,
            text_content="hello world",
            mistakes=[('l', 'k'), ('o', 'p')]
        )
        
        self.assertEqual(session.session_id, 'test_session')
        self.assertEqual(session.total_keys, 10)
        self.assertEqual(session.correct_keys, 8)
        self.assertEqual(session.wpm, 60.0)
        self.assertEqual(session.accuracy, 80.0)
        self.assertEqual(len(session.mistakes), 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def test_create_tracker(self):
        """Test create_tracker function"""
        tracker = create_tracker()
        self.assertIsInstance(tracker, TypingTracker)
    
    @patch('tracker.time.sleep')
    def test_simulate_typing_session(self, mock_sleep):
        """Test simulate_typing_session function"""
        tracker = create_tracker()
        text = "hello"
        
        session = simulate_typing_session(tracker, text)
        
        self.assertIsInstance(session, TypingSession)
        self.assertGreater(session.total_keys, 0)
        mock_sleep.assert_called()


class TestDatabaseOperations(unittest.TestCase):
    """Test cases for database operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = TypingTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Close any open database connections
        if hasattr(self, 'tracker'):
            self.tracker = None
        
        # Wait a moment for file handles to be released
        import time
        time.sleep(0.1)
        
        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except PermissionError:
                # If we can't delete it, that's okay for tests
                pass
    
    def test_save_session_to_db(self):
        """Test saving session to database"""
        self.tracker.start_session()
        self.tracker.record_keypress('a', 'a', "a")
        self.tracker.record_keypress('b', 'b', "ab")
        
        session = self.tracker.end_session()
        
        # Check that session is saved in database
        import sqlite3
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM typing_sessions")
        session_count = cursor.fetchone()[0]
        self.assertEqual(session_count, 1)
        
        cursor.execute("SELECT COUNT(*) FROM typing_events")
        event_count = cursor.fetchone()[0]
        self.assertEqual(event_count, 2)
        
        conn.close()
    
    def test_multiple_sessions(self):
        """Test multiple sessions in database"""
        # Create multiple sessions
        for i in range(3):
            self.tracker.start_session()
            self.tracker.record_keypress('a', 'a', "a")
            self.tracker.end_session()
        
        # Check database
        import sqlite3
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM typing_sessions")
        session_count = cursor.fetchone()[0]
        self.assertEqual(session_count, 3)
        
        cursor.execute("SELECT COUNT(*) FROM typing_events")
        event_count = cursor.fetchone()[0]
        self.assertEqual(event_count, 3)
        
        conn.close()


if __name__ == '__main__':
    unittest.main()
