"""
Database management for habit tracking data.
"""

import sqlite3
import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pathlib import Path

from .habit_model import HabitEntry, HabitTracker


class DatabaseManager:
    """Manages SQLite database operations for habit tracking."""
    
    def __init__(self, db_path: str = "data/habits.db"):
        """Initialize database manager with path to SQLite database."""
        self.db_path = db_path
        self._ensure_data_directory()
        self._create_tables()
    
    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _create_tables(self) -> None:
        """Create the necessary database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS habit_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    sleep_hours REAL NOT NULL,
                    exercise_minutes INTEGER NOT NULL,
                    screen_time_hours REAL NOT NULL,
                    water_glasses INTEGER NOT NULL,
                    work_hours REAL NOT NULL,
                    mood_rating INTEGER NOT NULL,
                    productivity_rating INTEGER NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_entry(self, entry: HabitEntry) -> bool:
        """Save a habit entry to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO habit_entries 
                    (date, sleep_hours, exercise_minutes, screen_time_hours, 
                     water_glasses, work_hours, mood_rating, productivity_rating, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    entry.date.isoformat(),
                    entry.sleep_hours,
                    entry.exercise_minutes,
                    entry.screen_time_hours,
                    entry.water_glasses,
                    entry.work_hours,
                    entry.mood_rating,
                    entry.productivity_rating,
                    entry.notes
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving entry: {e}")
            return False
    
    def get_entry(self, entry_date: date) -> Optional[HabitEntry]:
        """Retrieve a habit entry for a specific date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT date, sleep_hours, exercise_minutes, screen_time_hours,
                           water_glasses, work_hours, mood_rating, productivity_rating, notes
                    FROM habit_entries
                    WHERE date = ?
                ''', (entry_date.isoformat(),))
                
                row = cursor.fetchone()
                if row:
                    return HabitEntry(
                        date=datetime.fromisoformat(row[0]).date(),
                        sleep_hours=row[1],
                        exercise_minutes=row[2],
                        screen_time_hours=row[3],
                        water_glasses=row[4],
                        work_hours=row[5],
                        mood_rating=row[6],
                        productivity_rating=row[7],
                        notes=row[8]
                    )
                return None
        except Exception as e:
            print(f"Error retrieving entry: {e}")
            return None
    
    def get_all_entries(self) -> List[HabitEntry]:
        """Retrieve all habit entries from the database."""
        entries = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT date, sleep_hours, exercise_minutes, screen_time_hours,
                           water_glasses, work_hours, mood_rating, productivity_rating, notes
                    FROM habit_entries
                    ORDER BY date
                ''')
                
                for row in cursor.fetchall():
                    entry = HabitEntry(
                        date=datetime.fromisoformat(row[0]).date(),
                        sleep_hours=row[1],
                        exercise_minutes=row[2],
                        screen_time_hours=row[3],
                        water_glasses=row[4],
                        work_hours=row[5],
                        mood_rating=row[6],
                        productivity_rating=row[7],
                        notes=row[8]
                    )
                    entries.append(entry)
        except Exception as e:
            print(f"Error retrieving entries: {e}")
        
        return entries
    
    def get_entries_in_range(self, start_date: date, end_date: date) -> List[HabitEntry]:
        """Retrieve habit entries within a date range."""
        entries = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT date, sleep_hours, exercise_minutes, screen_time_hours,
                           water_glasses, work_hours, mood_rating, productivity_rating, notes
                    FROM habit_entries
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                for row in cursor.fetchall():
                    entry = HabitEntry(
                        date=datetime.fromisoformat(row[0]).date(),
                        sleep_hours=row[1],
                        exercise_minutes=row[2],
                        screen_time_hours=row[3],
                        water_glasses=row[4],
                        work_hours=row[5],
                        mood_rating=row[6],
                        productivity_rating=row[7],
                        notes=row[8]
                    )
                    entries.append(entry)
        except Exception as e:
            print(f"Error retrieving entries in range: {e}")
        
        return entries
    
    def delete_entry(self, entry_date: date) -> bool:
        """Delete a habit entry for a specific date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM habit_entries
                    WHERE date = ?
                ''', (entry_date.isoformat(),))
                
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting entry: {e}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total entries
                cursor.execute('SELECT COUNT(*) FROM habit_entries')
                total_entries = cursor.fetchone()[0]
                
                if total_entries == 0:
                    return {'total_entries': 0}
                
                # Get date range
                cursor.execute('SELECT MIN(date), MAX(date) FROM habit_entries')
                min_date, max_date = cursor.fetchone()
                
                # Get averages
                cursor.execute('''
                    SELECT AVG(sleep_hours), AVG(exercise_minutes), AVG(screen_time_hours),
                           AVG(water_glasses), AVG(work_hours), AVG(mood_rating), AVG(productivity_rating)
                    FROM habit_entries
                ''')
                averages = cursor.fetchone()
                
                # Get best days
                cursor.execute('''
                    SELECT date, sleep_hours, exercise_minutes, screen_time_hours,
                           water_glasses, work_hours, mood_rating, productivity_rating, notes
                    FROM habit_entries
                    WHERE mood_rating = (SELECT MAX(mood_rating) FROM habit_entries)
                    LIMIT 1
                ''')
                best_mood = cursor.fetchone()
                
                cursor.execute('''
                    SELECT date, sleep_hours, exercise_minutes, screen_time_hours,
                           water_glasses, work_hours, mood_rating, productivity_rating, notes
                    FROM habit_entries
                    WHERE productivity_rating = (SELECT MAX(productivity_rating) FROM habit_entries)
                    LIMIT 1
                ''')
                best_productivity = cursor.fetchone()
                
                return {
                    'total_entries': total_entries,
                    'date_range': {
                        'start': datetime.fromisoformat(min_date).date(),
                        'end': datetime.fromisoformat(max_date).date()
                    },
                    'averages': {
                        'sleep_hours': averages[0],
                        'exercise_minutes': averages[1],
                        'screen_time_hours': averages[2],
                        'water_glasses': averages[3],
                        'work_hours': averages[4],
                        'mood_rating': averages[5],
                        'productivity_rating': averages[6]
                    },
                    'best_days': {
                        'highest_mood': {
                            'date': datetime.fromisoformat(best_mood[0]).date(),
                            'sleep_hours': best_mood[1],
                            'exercise_minutes': best_mood[2],
                            'screen_time_hours': best_mood[3],
                            'water_glasses': best_mood[4],
                            'work_hours': best_mood[5],
                            'mood_rating': best_mood[6],
                            'productivity_rating': best_mood[7],
                            'notes': best_mood[8]
                        } if best_mood else None,
                        'highest_productivity': {
                            'date': datetime.fromisoformat(best_productivity[0]).date(),
                            'sleep_hours': best_productivity[1],
                            'exercise_minutes': best_productivity[2],
                            'screen_time_hours': best_productivity[3],
                            'water_glasses': best_productivity[4],
                            'work_hours': best_productivity[5],
                            'mood_rating': best_productivity[6],
                            'productivity_rating': best_productivity[7],
                            'notes': best_productivity[8]
                        } if best_productivity else None
                    }
                }
        except Exception as e:
            print(f"Error getting summary stats: {e}")
            return {}
    
    def load_tracker(self) -> HabitTracker:
        """Load all entries into a HabitTracker object."""
        tracker = HabitTracker()
        entries = self.get_all_entries()
        for entry in entries:
            tracker.add_entry(entry)
        return tracker
    
    def clear_all_data(self) -> bool:
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM habit_entries')
                conn.commit()
                return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
