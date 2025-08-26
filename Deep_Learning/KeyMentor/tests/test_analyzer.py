"""
Tests for the analyzer module
"""

import unittest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from analyzer import TypingAnalyzer, WeakSpot, TypingProfile, create_analyzer
from tracker import TypingTracker


class TestTypingAnalyzer(unittest.TestCase):
    """Test cases for TypingAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary database file for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.analyzer = TypingAnalyzer(self.temp_db.name)
        self.tracker = TypingTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary database file
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.common_bigrams)
        self.assertIsNotNone(self.analyzer.common_trigrams)
        self.assertIsInstance(self.analyzer.common_bigrams, set)
        self.assertIsInstance(self.analyzer.common_trigrams, set)
    
    def test_load_common_bigrams(self):
        """Test loading common bigrams"""
        bigrams = self.analyzer._load_common_bigrams()
        self.assertIsInstance(bigrams, set)
        self.assertGreater(len(bigrams), 0)
        # Check for some common bigrams
        self.assertIn('th', bigrams)
        self.assertIn('he', bigrams)
        self.assertIn('an', bigrams)
    
    def test_load_common_trigrams(self):
        """Test loading common trigrams"""
        trigrams = self.analyzer._load_common_trigrams()
        self.assertIsInstance(trigrams, set)
        self.assertGreater(len(trigrams), 0)
        # Check for some common trigrams
        self.assertIn('the', trigrams)
        self.assertIn('and', trigrams)
    
    def test_analyze_user_typing_no_data(self):
        """Test analysis with no typing data"""
        with self.assertRaises(ValueError):
            self.analyzer.analyze_user_typing()
    
    def test_analyze_user_typing_with_data(self):
        """Test analysis with typing data"""
        # Create some typing data
        self._create_sample_typing_data()
        
        profile = self.analyzer.analyze_user_typing()
        
        # Check profile properties
        self.assertIsInstance(profile, TypingProfile)
        self.assertGreater(profile.total_sessions, 0)
        self.assertGreater(profile.avg_wpm, 0)
        self.assertGreater(profile.avg_accuracy, 0)
        self.assertIsInstance(profile.weak_spots, list)
        self.assertIsInstance(profile.common_mistakes, dict)
        self.assertIsInstance(profile.finger_weaknesses, dict)
    
    def test_identify_weak_spots(self):
        """Test weak spot identification"""
        # Create typing data with known patterns
        self._create_typing_data_with_mistakes()
        
        profile = self.analyzer.analyze_user_typing()
        
        # Should have some weak spots
        self.assertGreater(len(profile.weak_spots), 0)
        
        # Check weak spot properties
        for weak_spot in profile.weak_spots:
            self.assertIsInstance(weak_spot, WeakSpot)
            self.assertIsInstance(weak_spot.pattern, str)
            self.assertIsInstance(weak_spot.error_rate, float)
            self.assertIsInstance(weak_spot.avg_reaction_time, float)
            self.assertIsInstance(weak_spot.frequency, int)
            self.assertIsInstance(weak_spot.difficulty_score, float)
            self.assertIn(weak_spot.pattern_type, ['character', 'bigram', 'trigram'])
    
    def test_analyze_character_weak_spots(self):
        """Test character weak spot analysis"""
        # Create data with character mistakes
        self._create_character_mistakes_data()
        
        profile = self.analyzer.analyze_user_typing()
        char_weak_spots = [ws for ws in profile.weak_spots if ws.pattern_type == 'character']
        
        self.assertGreater(len(char_weak_spots), 0)
        
        # Check that difficulty scores are calculated
        for weak_spot in char_weak_spots:
            self.assertGreaterEqual(weak_spot.difficulty_score, 0)
            self.assertLessEqual(weak_spot.difficulty_score, 1)
    
    def test_analyze_bigram_weak_spots(self):
        """Test bigram weak spot analysis"""
        # Create data with bigram patterns
        self._create_bigram_data()
        
        profile = self.analyzer.analyze_user_typing()
        bigram_weak_spots = [ws for ws in profile.weak_spots if ws.pattern_type == 'bigram']
        
        # May or may not have bigram weak spots depending on data
        for weak_spot in bigram_weak_spots:
            self.assertEqual(len(weak_spot.pattern), 2)
    
    def test_analyze_trigram_weak_spots(self):
        """Test trigram weak spot analysis"""
        # Create data with trigram patterns
        self._create_trigram_data()
        
        profile = self.analyzer.analyze_user_typing()
        trigram_weak_spots = [ws for ws in profile.weak_spots if ws.pattern_type == 'trigram']
        
        # May or may not have trigram weak spots depending on data
        for weak_spot in trigram_weak_spots:
            self.assertEqual(len(weak_spot.pattern), 3)
    
    def test_analyze_common_mistakes(self):
        """Test common mistake analysis"""
        # Create data with mistakes
        self._create_typing_data_with_mistakes()
        
        profile = self.analyzer.analyze_user_typing()
        
        # Should have some common mistakes
        self.assertGreater(len(profile.common_mistakes), 0)
        
        # Check mistake format
        for expected, mistakes in profile.common_mistakes.items():
            self.assertIsInstance(expected, str)
            self.assertIsInstance(mistakes, list)
            self.assertLessEqual(len(mistakes), 3)  # Top 3 mistakes per character
    
    def test_analyze_finger_weaknesses(self):
        """Test finger weakness analysis"""
        # Create data with various characters
        self._create_finger_test_data()
        
        profile = self.analyzer.analyze_user_typing()
        
        # Should have some finger weaknesses
        self.assertGreater(len(profile.finger_weaknesses), 0)
        
        # Check finger weakness format
        for finger, error_rate in profile.finger_weaknesses.items():
            self.assertIsInstance(finger, str)
            self.assertIsInstance(error_rate, float)
            self.assertGreaterEqual(error_rate, 0)
            self.assertLessEqual(error_rate, 1)
    
    def test_get_progress_report(self):
        """Test progress report generation"""
        # Create recent typing data
        self._create_recent_typing_data()
        
        progress = self.analyzer.get_progress_report(days=7)
        
        # Check progress report properties
        self.assertIsInstance(progress, dict)
        self.assertIn('total_sessions', progress)
        self.assertIn('avg_wpm', progress)
        self.assertIn('avg_accuracy', progress)
        self.assertIn('wpm_trend', progress)
        self.assertIn('accuracy_trend', progress)
        
        self.assertGreater(progress['total_sessions'], 0)
        self.assertGreater(progress['avg_wpm'], 0)
    
    def test_get_progress_report_no_data(self):
        """Test progress report with no recent data"""
        progress = self.analyzer.get_progress_report(days=1)
        
        self.assertIn('error', progress)
        self.assertEqual(progress['error'], 'No sessions found in the specified time period')
    
    def _create_sample_typing_data(self):
        """Create sample typing data for testing"""
        # Create a few typing sessions
        for i in range(3):
            self.tracker.start_session()
            
            # Type "hello world" with some mistakes
            text = "hello world"
            for j, char in enumerate(text):
                if j == 2 and i == 0:  # Make a mistake in first session
                    self.tracker.record_keypress('x', char, text[:j] + 'x')
                else:
                    self.tracker.record_keypress(char, char, text[:j+1])
            
            self.tracker.end_session()
    
    def _create_typing_data_with_mistakes(self):
        """Create typing data with intentional mistakes"""
        self.tracker.start_session()
        
        # Type with mistakes - focus on specific characters to meet minimum sample size
        text = "the the the the the quick brown fox"
        for i, char in enumerate(text):
            if char == 'e' and i % 3 == 0:  # Make mistakes on 'e' characters
                self.tracker.record_keypress('r', char, text[:i] + 'r')
            else:
                self.tracker.record_keypress(char, char, text[:i+1])
        
        self.tracker.end_session()
    
    def _create_character_mistakes_data(self):
        """Create data with character-specific mistakes"""
        self.tracker.start_session()
        
        # Focus on specific characters - create more data to meet minimum sample size
        text = "aaaaa aaaaa aaaaa bbbb cccc dddd"
        for i, char in enumerate(text):
            if char == 'a' and i % 3 == 0:  # Mistake every third 'a'
                self.tracker.record_keypress('s', char, text[:i] + 's')
            else:
                self.tracker.record_keypress(char, char, text[:i+1])
        
        self.tracker.end_session()
    
    def _create_bigram_data(self):
        """Create data with bigram patterns"""
        self.tracker.start_session()
        
        # Use common bigrams - create more data to meet minimum sample size
        text = "the the the the the and and and and and"
        for i, char in enumerate(text):
            self.tracker.record_keypress(char, char, text[:i+1])
        
        self.tracker.end_session()
    
    def _create_trigram_data(self):
        """Create data with trigram patterns"""
        self.tracker.start_session()
        
        # Use common trigrams
        text = "the the and and for for"
        for i, char in enumerate(text):
            self.tracker.record_keypress(char, char, text[:i+1])
        
        self.tracker.end_session()
    
    def _create_finger_test_data(self):
        """Create data for finger weakness testing"""
        self.tracker.start_session()
        
        # Use characters mapped to different fingers - create more data
        text = "asdf jkl; qwer uiop asdf jkl; qwer uiop asdf jkl;"
        for i, char in enumerate(text):
            self.tracker.record_keypress(char, char, text[:i+1])
        
        self.tracker.end_session()
    
    def _create_recent_typing_data(self):
        """Create recent typing data for progress report testing"""
        # Create sessions with recent timestamps
        for i in range(3):
            self.tracker.start_session()
            self.tracker.record_keypress('a', 'a', "a")
            self.tracker.record_keypress('b', 'b', "ab")
            self.tracker.end_session()


class TestWeakSpot(unittest.TestCase):
    """Test cases for WeakSpot dataclass"""
    
    def test_weak_spot_creation(self):
        """Test creating WeakSpot instances"""
        weak_spot = WeakSpot(
            pattern="th",
            error_rate=0.15,
            avg_reaction_time=0.2,
            frequency=20,
            difficulty_score=0.75,
            pattern_type="bigram"
        )
        
        self.assertEqual(weak_spot.pattern, "th")
        self.assertEqual(weak_spot.error_rate, 0.15)
        self.assertEqual(weak_spot.avg_reaction_time, 0.2)
        self.assertEqual(weak_spot.frequency, 20)
        self.assertEqual(weak_spot.difficulty_score, 0.75)
        self.assertEqual(weak_spot.pattern_type, "bigram")


class TestTypingProfile(unittest.TestCase):
    """Test cases for TypingProfile dataclass"""
    
    def test_typing_profile_creation(self):
        """Test creating TypingProfile instances"""
        from datetime import datetime
        
        weak_spots = [
            WeakSpot("a", 0.1, 0.15, 10, 0.5, "character"),
            WeakSpot("th", 0.2, 0.25, 5, 0.7, "bigram")
        ]
        
        profile = TypingProfile(
            user_id="test_user",
            total_sessions=5,
            avg_wpm=60.0,
            avg_accuracy=85.0,
            weak_spots=weak_spots,
            common_mistakes={"a": ["s", "q"], "e": ["r"]},
            finger_weaknesses={"left_index": 0.1, "right_index": 0.05},
            generated_at=datetime.now()
        )
        
        self.assertEqual(profile.user_id, "test_user")
        self.assertEqual(profile.total_sessions, 5)
        self.assertEqual(profile.avg_wpm, 60.0)
        self.assertEqual(profile.avg_accuracy, 85.0)
        self.assertEqual(len(profile.weak_spots), 2)
        self.assertEqual(len(profile.common_mistakes), 2)
        self.assertEqual(len(profile.finger_weaknesses), 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def test_create_analyzer(self):
        """Test create_analyzer function"""
        analyzer = create_analyzer()
        self.assertIsInstance(analyzer, TypingAnalyzer)


class TestDatabaseOperations(unittest.TestCase):
    """Test cases for database operations in analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.analyzer = TypingAnalyzer(self.temp_db.name)
        self.tracker = TypingTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_get_recent_sessions(self):
        """Test retrieving recent sessions"""
        # Create some sessions
        for i in range(3):
            self.tracker.start_session()
            self.tracker.record_keypress('a', 'a', "a")
            self.tracker.end_session()
        
        sessions = self.analyzer._get_recent_sessions(5)
        
        self.assertEqual(len(sessions), 3)
        self.assertIsInstance(sessions[0], dict)
        self.assertIn('session_id', sessions[0])
        self.assertIn('wpm', sessions[0])
        self.assertIn('accuracy', sessions[0])
    
    def test_get_all_typing_events(self):
        """Test retrieving all typing events"""
        # Create a session with events
        session_id = self.tracker.start_session()
        self.tracker.record_keypress('a', 'a', "a")
        self.tracker.record_keypress('b', 'b', "ab")
        self.tracker.end_session()
        
        sessions = self.analyzer._get_recent_sessions(1)
        events = self.analyzer._get_all_typing_events(sessions)
        
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], dict)
        self.assertIn('key', events[0])
        self.assertIn('expected_key', events[0])
        self.assertIn('is_correct', events[0])


if __name__ == '__main__':
    unittest.main()
