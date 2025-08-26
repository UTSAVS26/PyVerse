"""
KeyMentor - Typing Analysis Engine
Learns weak spots from typing logs and identifies areas for improvement.
"""

import sqlite3
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import statistics
from dataclasses import dataclass
import json
from datetime import datetime, timedelta


@dataclass
class WeakSpot:
    """Represents a typing weak spot"""
    pattern: str  # Could be a single character, bigram, trigram, etc.
    error_rate: float
    avg_reaction_time: float
    frequency: int
    difficulty_score: float
    pattern_type: str  # 'character', 'bigram', 'trigram', 'word'


@dataclass
class TypingProfile:
    """User's typing profile with identified weak spots"""
    user_id: str
    total_sessions: int
    avg_wpm: float
    avg_accuracy: float
    weak_spots: List[WeakSpot]
    common_mistakes: Dict[str, List[str]]  # expected -> [actual mistakes]
    finger_weaknesses: Dict[str, float]  # finger -> error rate
    generated_at: datetime


class TypingAnalyzer:
    """Analyzes typing data to identify weak spots and improvement areas"""
    
    def __init__(self, db_path: str = "typing_data.db"):
        self.db_path = db_path
        self.common_bigrams = self._load_common_bigrams()
        self.common_trigrams = self._load_common_trigrams()
    
    def _load_common_bigrams(self) -> Set[str]:
        """Load common English bigrams for analysis"""
        return {
            'th', 'he', 'an', 'in', 'er', 're', 'on', 'at', 'en', 'nd', 'ti', 'es', 'or', 'te',
            'of', 'ed', 'is', 'it', 'al', 'ar', 'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou',
            'io', 'le', 've', 'co', 'me', 'de', 'hi', 'ri', 'ro', 'ic', 'ne', 'ea', 'ch', 'll',
            'be', 'ma', 'si', 'om', 'ur', 'ca', 'el', 'ta', 'la', 'ns', 'di', 'fo', 'ho', 'fe',
            'pa', 'we', 're', 'mo', 'no', 'li', 'wa', 'sa', 'se', 'ne', 'us', 'na', 'ni', 'ka'
        }
    
    def _load_common_trigrams(self) -> Set[str]:
        """Load common English trigrams for analysis"""
        return {
            'the', 'and', 'tha', 'ent', 'ing', 'ion', 'tio', 'for', 'nde', 'has', 'nce',
            'edt', 'tis', 'oft', 'sth', 'men', 'ere', 'con', 'res', 'ver', 'ter', 'com',
            'ess', 'ate', 'his', 'ill', 'sse', 'nce', 'ect', 'are', 'ain', 'sto', 'her',
            'ere', 'est', 'ons', 'nti', 'int', 'rea', 'era', 'nct', 'thi', 'wit', 'din',
            'ver', 'se', 'pro', 'thi', 'wit', 'but', 'hav', 'thi', 'his', 'not', 'thi'
        }
    
    def analyze_user_typing(self, user_id: str = "default", 
                          sessions_limit: int = 50) -> TypingProfile:
        """Analyze user's typing patterns and identify weak spots"""
        
        # Get recent sessions
        sessions = self._get_recent_sessions(sessions_limit)
        if not sessions:
            raise ValueError("No typing sessions found for analysis")
        
        # Get all typing events
        all_events = self._get_all_typing_events(sessions)
        
        # Calculate basic metrics
        avg_wpm = statistics.mean([s['wpm'] for s in sessions])
        avg_accuracy = statistics.mean([s['accuracy'] for s in sessions])
        
        # Analyze weak spots
        weak_spots = self._identify_weak_spots(all_events)
        
        # Analyze common mistakes
        common_mistakes = self._analyze_common_mistakes(all_events)
        
        # Analyze finger weaknesses (simplified mapping)
        finger_weaknesses = self._analyze_finger_weaknesses(all_events)
        
        return TypingProfile(
            user_id=user_id,
            total_sessions=len(sessions),
            avg_wpm=avg_wpm,
            avg_accuracy=avg_accuracy,
            weak_spots=weak_spots,
            common_mistakes=common_mistakes,
            finger_weaknesses=finger_weaknesses,
            generated_at=datetime.now()
        )
    
    def _get_recent_sessions(self, limit: int) -> List[Dict]:
        """Get recent typing sessions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, start_time, end_time, total_keys, correct_keys, wpm, accuracy
            FROM typing_sessions
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'total_keys': row[3],
                'correct_keys': row[4],
                'wpm': row[5],
                'accuracy': row[6]
            })
        
        conn.close()
        return sessions
    
    def _get_all_typing_events(self, sessions: List[Dict]) -> List[Dict]:
        """Get all typing events for the given sessions"""
        if not sessions:
            return []
        
        session_ids = [s['session_id'] for s in sessions]
        placeholders = ','.join(['?' for _ in session_ids])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT session_id, timestamp, key, expected_key, is_correct, reaction_time
            FROM typing_events
            WHERE session_id IN ({placeholders})
            ORDER BY timestamp
        ''', session_ids)
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'session_id': row[0],
                'timestamp': row[1],
                'key': row[2],
                'expected_key': row[3],
                'is_correct': bool(row[4]),
                'reaction_time': row[5]
            })
        
        conn.close()
        return events
    
    def _identify_weak_spots(self, events: List[Dict]) -> List[WeakSpot]:
        """Identify weak spots from typing events"""
        weak_spots = []
        
        # Analyze individual characters
        char_weak_spots = self._analyze_character_weak_spots(events)
        weak_spots.extend(char_weak_spots)
        
        # Analyze bigrams
        bigram_weak_spots = self._analyze_bigram_weak_spots(events)
        weak_spots.extend(bigram_weak_spots)
        
        # Analyze trigrams
        trigram_weak_spots = self._analyze_trigram_weak_spots(events)
        weak_spots.extend(trigram_weak_spots)
        
        # Sort by difficulty score (higher = more difficult)
        weak_spots.sort(key=lambda x: x.difficulty_score, reverse=True)
        
        return weak_spots[:20]  # Return top 20 weak spots
    
    def _analyze_character_weak_spots(self, events: List[Dict]) -> List[WeakSpot]:
        """Analyze weak spots for individual characters"""
        char_stats = defaultdict(lambda: {'errors': 0, 'total': 0, 'reaction_times': []})
        
        for event in events:
            char = event['expected_key'].lower()
            if len(char) == 1 and char.isalpha():
                char_stats[char]['total'] += 1
                char_stats[char]['reaction_times'].append(event['reaction_time'])
                
                if not event['is_correct']:
                    char_stats[char]['errors'] += 1
        
        weak_spots = []
        for char, stats in char_stats.items():
            if stats['total'] >= 3:  # Minimum sample size (reduced for testing)
                error_rate = stats['errors'] / stats['total']
                avg_reaction_time = statistics.mean(stats['reaction_times'])
                
                # Calculate difficulty score (higher = more difficult)
                difficulty_score = (error_rate * 0.6) + (avg_reaction_time * 0.4)
                
                weak_spots.append(WeakSpot(
                    pattern=char,
                    error_rate=error_rate,
                    avg_reaction_time=avg_reaction_time,
                    frequency=stats['total'],
                    difficulty_score=difficulty_score,
                    pattern_type='character'
                ))
        
        return weak_spots
    
    def _analyze_bigram_weak_spots(self, events: List[Dict]) -> List[WeakSpot]:
        """Analyze weak spots for bigrams"""
        bigram_stats = defaultdict(lambda: {'errors': 0, 'total': 0, 'reaction_times': []})
        
        # Extract bigrams from events
        for i in range(len(events) - 1):
            if events[i]['is_correct'] and events[i+1]['is_correct']:
                bigram = (events[i]['expected_key'] + events[i+1]['expected_key']).lower()
                if bigram in self.common_bigrams:
                    bigram_stats[bigram]['total'] += 1
                    bigram_stats[bigram]['reaction_times'].append(
                        events[i]['reaction_time'] + events[i+1]['reaction_time']
                    )
        
        # Check for errors in bigrams
        for i in range(len(events) - 1):
            bigram = (events[i]['expected_key'] + events[i+1]['expected_key']).lower()
            if bigram in self.common_bigrams:
                if not events[i]['is_correct'] or not events[i+1]['is_correct']:
                    bigram_stats[bigram]['errors'] += 1
        
        weak_spots = []
        for bigram, stats in bigram_stats.items():
            if stats['total'] >= 2:  # Minimum sample size for bigrams (reduced for testing)
                error_rate = stats['errors'] / stats['total']
                avg_reaction_time = statistics.mean(stats['reaction_times'])
                
                difficulty_score = (error_rate * 0.7) + (avg_reaction_time * 0.3)
                
                weak_spots.append(WeakSpot(
                    pattern=bigram,
                    error_rate=error_rate,
                    avg_reaction_time=avg_reaction_time,
                    frequency=stats['total'],
                    difficulty_score=difficulty_score,
                    pattern_type='bigram'
                ))
        
        return weak_spots
    
    def _analyze_trigram_weak_spots(self, events: List[Dict]) -> List[WeakSpot]:
        """Analyze weak spots for trigrams"""
        trigram_stats = defaultdict(lambda: {'errors': 0, 'total': 0, 'reaction_times': []})
        
        # Extract trigrams from events
        for i in range(len(events) - 2):
            if (events[i]['is_correct'] and events[i+1]['is_correct'] and 
                events[i+2]['is_correct']):
                trigram = (events[i]['expected_key'] + events[i+1]['expected_key'] + 
                          events[i+2]['expected_key']).lower()
                if trigram in self.common_trigrams:
                    trigram_stats[trigram]['total'] += 1
                    trigram_stats[trigram]['reaction_times'].append(
                        events[i]['reaction_time'] + events[i+1]['reaction_time'] + 
                        events[i+2]['reaction_time']
                    )
        
        # Check for errors in trigrams
        for i in range(len(events) - 2):
            trigram = (events[i]['expected_key'] + events[i+1]['expected_key'] + 
                      events[i+2]['expected_key']).lower()
            if trigram in self.common_trigrams:
                if (not events[i]['is_correct'] or not events[i+1]['is_correct'] or 
                    not events[i+2]['is_correct']):
                    trigram_stats[trigram]['errors'] += 1
        
        weak_spots = []
        for trigram, stats in trigram_stats.items():
            if stats['total'] >= 2:  # Minimum sample size for trigrams
                error_rate = stats['errors'] / stats['total']
                avg_reaction_time = statistics.mean(stats['reaction_times'])
                
                difficulty_score = (error_rate * 0.8) + (avg_reaction_time * 0.2)
                
                weak_spots.append(WeakSpot(
                    pattern=trigram,
                    error_rate=error_rate,
                    avg_reaction_time=avg_reaction_time,
                    frequency=stats['total'],
                    difficulty_score=difficulty_score,
                    pattern_type='trigram'
                ))
        
        return weak_spots
    
    def _analyze_common_mistakes(self, events: List[Dict]) -> Dict[str, List[str]]:
        """Analyze common typing mistakes"""
        mistake_patterns = defaultdict(Counter)
        
        for event in events:
            if not event['is_correct']:
                expected = event['expected_key'].lower()
                actual = event['key'].lower()
                if expected.isalpha() and actual.isalpha():
                    mistake_patterns[expected][actual] += 1
        
        # Convert to list format and keep top 3 mistakes per character
        common_mistakes = {}
        for expected, mistakes in mistake_patterns.items():
            common_mistakes[expected] = [char for char, count in mistakes.most_common(3)]
        
        return common_mistakes
    
    def _analyze_finger_weaknesses(self, events: List[Dict]) -> Dict[str, float]:
        """Analyze weaknesses by finger (simplified mapping)"""
        # Simplified finger mapping (QWERTY layout)
        finger_mapping = {
            'a': 'left_pinky', 'q': 'left_pinky', 'z': 'left_pinky',
            's': 'left_ring', 'w': 'left_ring', 'x': 'left_ring',
            'd': 'left_middle', 'e': 'left_middle', 'c': 'left_middle',
            'f': 'left_index', 'r': 'left_index', 'v': 'left_index',
            'g': 'left_index', 't': 'left_index', 'b': 'left_index',
            'h': 'right_index', 'y': 'right_index', 'n': 'right_index',
            'j': 'right_index', 'u': 'right_index', 'm': 'right_index',
            'k': 'right_middle', 'i': 'right_middle', ',': 'right_middle',
            'l': 'right_ring', 'o': 'right_ring', '.': 'right_ring',
            ';': 'right_pinky', 'p': 'right_pinky', '/': 'right_pinky',
            "'": 'right_pinky', '[': 'right_pinky', ']': 'right_pinky'
        }
        
        finger_stats = defaultdict(lambda: {'errors': 0, 'total': 0})
        
        for event in events:
            char = event['expected_key'].lower()
            if char in finger_mapping:
                finger = finger_mapping[char]
                finger_stats[finger]['total'] += 1
                if not event['is_correct']:
                    finger_stats[finger]['errors'] += 1
        
        finger_weaknesses = {}
        for finger, stats in finger_stats.items():
            if stats['total'] >= 5:  # Minimum sample size
                error_rate = stats['errors'] / stats['total']
                finger_weaknesses[finger] = error_rate
        
        return finger_weaknesses
    
    def get_progress_report(self, user_id: str = "default", 
                          days: int = 7) -> Dict:
        """Generate a progress report for the specified time period"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT wpm, accuracy, total_keys, correct_keys
            FROM typing_sessions
            WHERE start_time > ?
            ORDER BY start_time
        ''', (cutoff_time.timestamp(),))
        
        recent_sessions = cursor.fetchall()
        conn.close()
        
        if not recent_sessions:
            return {"error": "No sessions found in the specified time period"}
        
        # Calculate progress metrics
        wpm_values = [row[0] for row in recent_sessions]
        accuracy_values = [row[1] for row in recent_sessions]
        total_keys = sum(row[2] for row in recent_sessions)
        correct_keys = sum(row[3] for row in recent_sessions)
        
        overall_accuracy = (correct_keys / total_keys * 100) if total_keys > 0 else 0
        
        return {
            "period_days": days,
            "total_sessions": len(recent_sessions),
            "avg_wpm": statistics.mean(wpm_values),
            "max_wpm": max(wpm_values),
            "min_wpm": min(wpm_values),
            "avg_accuracy": statistics.mean(accuracy_values),
            "overall_accuracy": overall_accuracy,
            "wpm_trend": "improving" if len(wpm_values) > 1 and wpm_values[-1] > wpm_values[0] else "stable",
            "accuracy_trend": "improving" if len(accuracy_values) > 1 and accuracy_values[-1] > accuracy_values[0] else "stable"
        }


def create_analyzer(db_path: str = "typing_data.db") -> TypingAnalyzer:
    """Create a new typing analyzer instance"""
    return TypingAnalyzer(db_path)


if __name__ == "__main__":
    # Example usage
    analyzer = create_analyzer()
    
    try:
        profile = analyzer.analyze_user_typing()
        print(f"User Profile Analysis:")
        print(f"Total Sessions: {profile.total_sessions}")
        print(f"Average WPM: {profile.avg_wpm:.2f}")
        print(f"Average Accuracy: {profile.avg_accuracy:.2f}%")
        print(f"\nTop Weak Spots:")
        for i, weak_spot in enumerate(profile.weak_spots[:5], 1):
            print(f"{i}. {weak_spot.pattern} ({weak_spot.pattern_type}) - "
                  f"Error Rate: {weak_spot.error_rate:.2%}, "
                  f"Difficulty: {weak_spot.difficulty_score:.3f}")
        
        print(f"\nCommon Mistakes:")
        for expected, mistakes in list(profile.common_mistakes.items())[:5]:
            print(f"{expected} -> {mistakes}")
        
        print(f"\nFinger Weaknesses:")
        for finger, error_rate in profile.finger_weaknesses.items():
            print(f"{finger}: {error_rate:.2%}")
        
        # Generate progress report
        progress = analyzer.get_progress_report(days=7)
        print(f"\n7-Day Progress Report:")
        print(f"Average WPM: {progress['avg_wpm']:.2f}")
        print(f"WPM Trend: {progress['wpm_trend']}")
        print(f"Accuracy Trend: {progress['accuracy_trend']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("No typing data available for analysis. Run some typing sessions first.")
