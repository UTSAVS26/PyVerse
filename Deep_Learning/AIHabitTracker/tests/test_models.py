"""
Tests for habit models and database functionality.
"""

import pytest
import tempfile
import os
from datetime import date, datetime
from unittest.mock import patch

from src.models.habit_model import HabitEntry, HabitTracker
from src.models.database import DatabaseManager


class TestHabitEntry:
    """Test HabitEntry class."""
    
    def test_valid_habit_entry(self):
        """Test creating a valid habit entry."""
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4,
            notes="Great day!"
        )
        
        assert entry.date == date(2025, 1, 1)
        assert entry.sleep_hours == 7.5
        assert entry.exercise_minutes == 30
        assert entry.screen_time_hours == 4.0
        assert entry.water_glasses == 8
        assert entry.work_hours == 8.0
        assert entry.mood_rating == 4
        assert entry.productivity_rating == 4
        assert entry.notes == "Great day!"
    
    def test_invalid_mood_rating(self):
        """Test invalid mood rating validation."""
        with pytest.raises(ValueError, match="Mood rating must be between 1 and 5"):
            HabitEntry(
                date=date(2025, 1, 1),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=6,  # Invalid
                productivity_rating=4
            )
    
    def test_invalid_productivity_rating(self):
        """Test invalid productivity rating validation."""
        with pytest.raises(ValueError, match="Productivity rating must be between 1 and 5"):
            HabitEntry(
                date=date(2025, 1, 1),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=0  # Invalid
            )
    
    def test_invalid_sleep_hours(self):
        """Test invalid sleep hours validation."""
        with pytest.raises(ValueError, match="Sleep hours must be between 0 and 24"):
            HabitEntry(
                date=date(2025, 1, 1),
                sleep_hours=25.0,  # Invalid
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
    
    def test_invalid_exercise_minutes(self):
        """Test invalid exercise minutes validation."""
        with pytest.raises(ValueError, match="Exercise minutes cannot be negative"):
            HabitEntry(
                date=date(2025, 1, 1),
                sleep_hours=7.5,
                exercise_minutes=-10,  # Invalid
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
    
    def test_to_dict(self):
        """Test converting habit entry to dictionary."""
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4,
            notes="Test notes"
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict['date'] == '2025-01-01'
        assert entry_dict['sleep_hours'] == 7.5
        assert entry_dict['exercise_minutes'] == 30
        assert entry_dict['screen_time_hours'] == 4.0
        assert entry_dict['water_glasses'] == 8
        assert entry_dict['work_hours'] == 8.0
        assert entry_dict['mood_rating'] == 4
        assert entry_dict['productivity_rating'] == 4
        assert entry_dict['notes'] == "Test notes"
    
    def test_from_dict(self):
        """Test creating habit entry from dictionary."""
        entry_dict = {
            'date': '2025-01-01',
            'sleep_hours': 7.5,
            'exercise_minutes': 30,
            'screen_time_hours': 4.0,
            'water_glasses': 8,
            'work_hours': 8.0,
            'mood_rating': 4,
            'productivity_rating': 4,
            'notes': 'Test notes'
        }
        
        entry = HabitEntry.from_dict(entry_dict)
        
        assert entry.date == date(2025, 1, 1)
        assert entry.sleep_hours == 7.5
        assert entry.exercise_minutes == 30
        assert entry.screen_time_hours == 4.0
        assert entry.water_glasses == 8
        assert entry.work_hours == 8.0
        assert entry.mood_rating == 4
        assert entry.productivity_rating == 4
        assert entry.notes == "Test notes"


class TestHabitTracker:
    """Test HabitTracker class."""
    
    def test_empty_tracker(self):
        """Test empty habit tracker."""
        tracker = HabitTracker()
        
        assert len(tracker.entries) == 0
        assert tracker.get_entry(date(2025, 1, 1)) is None
        assert tracker.to_dataframe().empty
    
    def test_add_entry(self):
        """Test adding entries to tracker."""
        tracker = HabitTracker()
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        
        tracker.add_entry(entry)
        
        assert len(tracker.entries) == 1
        assert tracker.get_entry(date(2025, 1, 1)) == entry
    
    def test_update_entry(self):
        """Test updating existing entry."""
        tracker = HabitTracker()
        entry1 = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        entry2 = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=8.0,
            exercise_minutes=45,
            screen_time_hours=3.0,
            water_glasses=10,
            work_hours=7.0,
            mood_rating=5,
            productivity_rating=5
        )
        
        tracker.add_entry(entry1)
        tracker.update_entry(entry2)
        
        assert len(tracker.entries) == 1
        assert tracker.get_entry(date(2025, 1, 1)) == entry2
    
    def test_update_nonexistent_entry(self):
        """Test updating non-existent entry."""
        tracker = HabitTracker()
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        
        with pytest.raises(ValueError, match="No entry found for date"):
            tracker.update_entry(entry)
    
    def test_delete_entry(self):
        """Test deleting entry."""
        tracker = HabitTracker()
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        
        tracker.add_entry(entry)
        assert len(tracker.entries) == 1
        
        tracker.delete_entry(date(2025, 1, 1))
        assert len(tracker.entries) == 0
        assert tracker.get_entry(date(2025, 1, 1)) is None
    
    def test_get_entries_in_range(self):
        """Test getting entries in date range."""
        tracker = HabitTracker()
        
        # Add entries for different dates
        for i in range(1, 6):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            tracker.add_entry(entry)
        
        # Get entries in range
        entries = tracker.get_entries_in_range(date(2025, 1, 2), date(2025, 1, 4))
        
        assert len(entries) == 3
        assert all(entry.date >= date(2025, 1, 2) and entry.date <= date(2025, 1, 4) 
                  for entry in entries)
    
    def test_to_dataframe(self):
        """Test converting tracker to DataFrame."""
        tracker = HabitTracker()
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        
        tracker.add_entry(entry)
        df = tracker.to_dataframe()
        
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]['sleep_hours'] == 7.5
        assert df.iloc[0]['exercise_minutes'] == 30
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        tracker = HabitTracker()
        
        # Add multiple entries
        for i in range(1, 4):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.0 + i * 0.5,
                exercise_minutes=30 + i * 10,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            tracker.add_entry(entry)
        
        stats = tracker.get_summary_stats()
        
        assert stats['total_entries'] == 3
        assert stats['date_range']['start'] == date(2025, 1, 1)
        assert stats['date_range']['end'] == date(2025, 1, 3)
        assert stats['averages']['sleep_hours'] == 8.0  # (7.5 + 8.0 + 8.5) / 3
        assert stats['averages']['exercise_minutes'] == 50  # (40 + 50 + 60) / 3


class TestDatabaseManager:
    """Test DatabaseManager class."""
    
    @pytest.fixture(scope="function")
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        yield db_path
        
        # Cleanup - handle Windows file locking
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            # File might still be in use, skip cleanup
            pass
    
    def test_create_database(self, temp_db):
        """Test database creation."""
        db_manager = DatabaseManager(temp_db)
        
        # Check if database file exists
        assert os.path.exists(temp_db)
        
        # Check if table exists by trying to get summary stats
        stats = db_manager.get_summary_stats()
        assert stats == {'total_entries': 0}
    
    def test_save_and_retrieve_entry(self, temp_db):
        """Test saving and retrieving entries."""
        db_manager = DatabaseManager(temp_db)
        
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4,
            notes="Test entry"
        )
        
        # Save entry
        assert db_manager.save_entry(entry) is True
        
        # Retrieve entry
        retrieved_entry = db_manager.get_entry(date(2025, 1, 1))
        
        assert retrieved_entry is not None
        assert retrieved_entry.date == entry.date
        assert retrieved_entry.sleep_hours == entry.sleep_hours
        assert retrieved_entry.exercise_minutes == entry.exercise_minutes
        assert retrieved_entry.notes == entry.notes
    
    def test_get_all_entries(self, temp_db):
        """Test retrieving all entries."""
        db_manager = DatabaseManager(temp_db)
        
        # Add multiple entries
        for i in range(1, 4):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            db_manager.save_entry(entry)
        
        entries = db_manager.get_all_entries()
        
        assert len(entries) == 3
        assert all(isinstance(entry, HabitEntry) for entry in entries)
    
    def test_get_entries_in_range(self, temp_db):
        """Test retrieving entries in date range."""
        db_manager = DatabaseManager(temp_db)
        
        # Add entries for different dates
        for i in range(1, 6):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            db_manager.save_entry(entry)
        
        entries = db_manager.get_entries_in_range(date(2025, 1, 2), date(2025, 1, 4))
        
        assert len(entries) == 3
        assert all(entry.date >= date(2025, 1, 2) and entry.date <= date(2025, 1, 4) 
                  for entry in entries)
    
    def test_delete_entry(self, temp_db):
        """Test deleting entries."""
        db_manager = DatabaseManager(temp_db)
        
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        
        db_manager.save_entry(entry)
        assert db_manager.get_entry(date(2025, 1, 1)) is not None
        
        assert db_manager.delete_entry(date(2025, 1, 1)) is True
        assert db_manager.get_entry(date(2025, 1, 1)) is None
    
    def test_get_summary_stats(self, temp_db):
        """Test getting summary statistics from database."""
        db_manager = DatabaseManager(temp_db)
        
        # Add multiple entries
        for i in range(1, 4):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.0 + i * 0.5,
                exercise_minutes=30 + i * 10,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            db_manager.save_entry(entry)
        
        stats = db_manager.get_summary_stats()
        
        assert stats['total_entries'] == 3
        assert stats['date_range']['start'] == date(2025, 1, 1)
        assert stats['date_range']['end'] == date(2025, 1, 3)
        assert stats['averages']['sleep_hours'] == 8.0
        assert stats['averages']['exercise_minutes'] == 50
    
    def test_load_tracker(self, temp_db):
        """Test loading tracker from database."""
        db_manager = DatabaseManager(temp_db)
        
        # Add entries
        for i in range(1, 4):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.5,
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
            db_manager.save_entry(entry)
        
        tracker = db_manager.load_tracker()
        
        assert len(tracker.entries) == 3
        assert all(isinstance(entry, HabitEntry) for entry in tracker.entries.values())
    
    def test_clear_all_data(self, temp_db):
        """Test clearing all data."""
        db_manager = DatabaseManager(temp_db)
        
        # Add some entries
        entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
        )
        db_manager.save_entry(entry)
        
        assert db_manager.get_summary_stats()['total_entries'] == 1
        
        assert db_manager.clear_all_data() is True
        assert db_manager.get_summary_stats()['total_entries'] == 0
