"""
Integration tests for the complete AI Habit Tracker application.
"""

import pytest
import tempfile
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from src.models.database import DatabaseManager
from src.models.habit_model import HabitEntry, HabitTracker
from src.analysis.pattern_detector import PatternDetector
from src.analysis.visualizer import HabitVisualizer


class TestCompleteWorkflow:
    """Test the complete application workflow."""
    
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
    
    def test_complete_user_workflow(self, temp_db):
        """Test a complete user workflow from data entry to insights."""
        # Initialize database
        db_manager = DatabaseManager(temp_db)
        
        # Step 1: Add habit entries over time
        entries_data = [
            # Date, Sleep, Exercise, Screen, Water, Work, Mood, Productivity, Notes
            (date(2025, 1, 1), 7.5, 30, 4.0, 8, 8.0, 4, 4, "Good day"),
            (date(2025, 1, 2), 6.0, 0, 8.0, 4, 6.0, 2, 2, "Tired"),
            (date(2025, 1, 3), 8.0, 45, 3.0, 10, 7.0, 5, 5, "Great day!"),
            (date(2025, 1, 4), 7.0, 20, 5.0, 6, 8.0, 3, 3, "Average"),
            (date(2025, 1, 5), 7.5, 60, 2.0, 12, 6.0, 5, 4, "Very productive"),
            (date(2025, 1, 6), 9.0, 0, 6.0, 8, 0.0, 4, 3, "Weekend"),
            (date(2025, 1, 7), 8.5, 30, 4.0, 10, 0.0, 4, 4, "Rest day"),
            (date(2025, 1, 8), 6.5, 0, 7.0, 5, 9.0, 2, 2, "Busy work day"),
            (date(2025, 1, 9), 7.0, 40, 3.5, 9, 8.0, 4, 4, "Balanced day"),
            (date(2025, 1, 10), 8.0, 50, 2.5, 11, 7.0, 5, 5, "Excellent day"),
        ]
        
        for entry_data in entries_data:
            entry = HabitEntry(
                date=entry_data[0],
                sleep_hours=entry_data[1],
                exercise_minutes=entry_data[2],
                screen_time_hours=entry_data[3],
                water_glasses=entry_data[4],
                work_hours=entry_data[5],
                mood_rating=entry_data[6],
                productivity_rating=entry_data[7],
                notes=entry_data[8]
            )
            assert db_manager.save_entry(entry) is True
        
        # Step 2: Load tracker and verify data
        tracker = db_manager.load_tracker()
        assert len(tracker.entries) == 10
        
        # Step 3: Get summary statistics
        stats = db_manager.get_summary_stats()
        assert stats['total_entries'] == 10
        assert stats['date_range']['start'] == date(2025, 1, 1)
        assert stats['date_range']['end'] == date(2025, 1, 10)
        
        # Step 4: Analyze patterns
        pattern_detector = PatternDetector(tracker)
        
        # Test correlations
        correlations = pattern_detector.analyze_correlations()
        assert 'mood_correlations' in correlations
        assert 'productivity_correlations' in correlations
        
        # Test pattern detection
        patterns = pattern_detector.detect_patterns()
        assert 'sleep' in patterns
        assert 'exercise' in patterns
        assert 'screen_time' in patterns
        assert 'weekly' in patterns
        assert 'streaks' in patterns
        
        # Test insights generation
        insights = pattern_detector.generate_insights()
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Test recommendations
        recommendations = pattern_detector.get_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Step 5: Test visualizations
        visualizer = HabitVisualizer(tracker)
        
        # Test dashboard creation
        dashboard = visualizer.create_dashboard()
        assert dashboard is not None
        assert hasattr(dashboard, 'data')
        assert hasattr(dashboard, 'layout')
        
        # Test correlation heatmap
        heatmap = visualizer.create_correlation_heatmap()
        assert heatmap is not None
        
        # Test trend analysis
        trends = visualizer.create_trend_analysis()
        assert trends is not None
        
        # Test weekly summary
        weekly = visualizer.create_weekly_summary()
        assert weekly is not None
        
        # Step 6: Test data export
        df = tracker.to_dataframe()
        assert not df.empty
        assert len(df) == 10
        
        # Step 7: Test data retrieval by date range
        recent_entries = db_manager.get_entries_in_range(
            date(2025, 1, 5), date(2025, 1, 10)
        )
        assert len(recent_entries) == 6
        
        # Step 8: Test individual entry retrieval
        entry = db_manager.get_entry(date(2025, 1, 3))
        assert entry is not None
        assert entry.sleep_hours == 8.0
        assert entry.exercise_minutes == 45
        assert entry.notes == "Great day!"
        
        # Step 9: Test entry update
        updated_entry = HabitEntry(
            date=date(2025, 1, 3),
            sleep_hours=8.5,
            exercise_minutes=50,
            screen_time_hours=2.5,
            water_glasses=11,
            work_hours=7.5,
            mood_rating=5,
            productivity_rating=5,
            notes="Updated great day!"
        )
        assert db_manager.save_entry(updated_entry) is True
        
        # Verify update
        retrieved_entry = db_manager.get_entry(date(2025, 1, 3))
        assert retrieved_entry.sleep_hours == 8.5
        assert retrieved_entry.notes == "Updated great day!"
    
    def test_data_persistence(self, temp_db):
        """Test that data persists across application restarts."""
        # First session: Add data
        db_manager1 = DatabaseManager(temp_db)
        
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
        db_manager1.save_entry(entry)
        
        # Simulate application restart
        del db_manager1
        
        # Second session: Verify data is still there
        db_manager2 = DatabaseManager(temp_db)
        tracker = db_manager2.load_tracker()
        
        assert len(tracker.entries) == 1
        retrieved_entry = db_manager2.get_entry(date(2025, 1, 1))
        assert retrieved_entry is not None
        assert retrieved_entry.sleep_hours == 7.5
        assert retrieved_entry.notes == "Test entry"
    
    def test_error_handling(self, temp_db):
        """Test error handling in the application."""
        db_manager = DatabaseManager(temp_db)
        
        # Test invalid entry creation
        with pytest.raises(ValueError):
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
        
        # Test invalid mood rating
        with pytest.raises(ValueError):
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
        
        # Test retrieving non-existent entry
        entry = db_manager.get_entry(date(2025, 1, 1))
        assert entry is None
    
    def test_large_dataset_performance(self, temp_db):
        """Test performance with larger datasets."""
        db_manager = DatabaseManager(temp_db)
        
        # Add 100 entries (using valid dates)
        start_date = date(2025, 1, 1)
        for i in range(100):
            entry_date = start_date + timedelta(days=i)
            entry = HabitEntry(
                date=entry_date,
                sleep_hours=7.0 + (i % 3) * 0.5,
                exercise_minutes=30 if i % 2 == 0 else 0,
                screen_time_hours=4.0 + (i % 2) * 2.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=3 + (i % 3),
                productivity_rating=3 + (i % 3)
            )
            db_manager.save_entry(entry)
        
        # Test loading all entries
        tracker = db_manager.load_tracker()
        assert len(tracker.entries) == 100
        
        # Test pattern detection with large dataset
        pattern_detector = PatternDetector(tracker)
        patterns = pattern_detector.detect_patterns()
        assert 'sleep' in patterns
        assert 'exercise' in patterns
        
        # Test visualizations with large dataset
        visualizer = HabitVisualizer(tracker)
        dashboard = visualizer.create_dashboard()
        assert dashboard is not None
        
        # Test date range queries
        entries = db_manager.get_entries_in_range(
            date(2025, 2, 15), date(2025, 2, 20)
        )
        assert len(entries) == 6  # 15 to 20 inclusive


class TestApplicationInterfaces:
    """Test different application interfaces."""
    
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
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        tracker = HabitTracker()
        
        for i in range(1, 8):
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.0 + (i % 3) * 0.5,
                exercise_minutes=30 if i % 2 == 0 else 0,
                screen_time_hours=4.0 + (i % 2) * 2.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=3 + (i % 3),
                productivity_rating=3 + (i % 3)
            )
            tracker.add_entry(entry)
        
        return tracker
    
    def test_cli_interface_simulation(self, sample_data, temp_db):
        """Simulate CLI interface operations."""
        db_manager = DatabaseManager(temp_db)
        
        # Simulate adding entries via CLI
        for entry in sample_data.entries.values():
            assert db_manager.save_entry(entry) is True
        
        # Simulate viewing entries
        entries = db_manager.get_all_entries()
        assert len(entries) == 7
        
        # Simulate getting insights
        tracker = db_manager.load_tracker()
        pattern_detector = PatternDetector(tracker)
        insights = pattern_detector.generate_insights()
        assert len(insights) > 0
        
        # Simulate getting statistics
        stats = db_manager.get_summary_stats()
        assert stats['total_entries'] == 7
    
    @patch('streamlit.write')
    @patch('streamlit.success')
    def test_streamlit_interface_simulation(self, mock_success, mock_write, sample_data, temp_db):
        """Simulate Streamlit interface operations."""
        db_manager = DatabaseManager(temp_db)
        
        # Simulate form submission
        entry = HabitEntry(
            date=date(2025, 1, 15),
            sleep_hours=8.0,
            exercise_minutes=45,
            screen_time_hours=3.0,
            water_glasses=10,
            work_hours=7.0,
            mood_rating=5,
            productivity_rating=5,
            notes="Streamlit test entry"
        )
        
        success = db_manager.save_entry(entry)
        if success:
            mock_success("Entry saved successfully!")
        
        # Simulate displaying insights
        tracker = db_manager.load_tracker()
        pattern_detector = PatternDetector(tracker)
        insights = pattern_detector.generate_insights()
        
        for insight in insights:
            mock_write(f"• {insight}")
        
        # Simulate displaying charts
        visualizer = HabitVisualizer(tracker)
        dashboard = visualizer.create_dashboard()
        assert dashboard is not None
    
    @patch('tkinter.messagebox.showinfo')
    @patch('tkinter.messagebox.showerror')
    def test_gui_interface_simulation(self, mock_error, mock_info, sample_data, temp_db):
        """Simulate GUI interface operations."""
        db_manager = DatabaseManager(temp_db)
        
        # Simulate form submission
        entry = HabitEntry(
            date=date(2025, 1, 20),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4,
            notes="GUI test entry"
        )
        
        try:
            success = db_manager.save_entry(entry)
            if success:
                mock_info("Entry saved successfully!")
            else:
                mock_error("Failed to save entry!")
        except Exception as e:
            mock_error(f"Error: {str(e)}")
        
        # Simulate loading entry
        try:
            loaded_entry = db_manager.get_entry(date(2025, 1, 20))
            if loaded_entry:
                mock_info("Entry loaded successfully!")
            else:
                mock_info("No entry found for this date.")
        except Exception as e:
            mock_error(f"Error: {str(e)}")


class TestDataValidation:
    """Test data validation and edge cases."""
    
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
    
    def test_boundary_values(self, temp_db):
        """Test boundary values for habit entries."""
        db_manager = DatabaseManager(temp_db)
        
        # Test minimum valid values
        min_entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=0.0,
            exercise_minutes=0,
            screen_time_hours=0.0,
            water_glasses=0,
            work_hours=0.0,
            mood_rating=1,
            productivity_rating=1
        )
        assert db_manager.save_entry(min_entry) is True
        
        # Test maximum valid values
        max_entry = HabitEntry(
            date=date(2025, 1, 2),
            sleep_hours=24.0,
            exercise_minutes=480,  # 8 hours
            screen_time_hours=24.0,
            water_glasses=50,
            work_hours=24.0,
            mood_rating=5,
            productivity_rating=5
        )
        assert db_manager.save_entry(max_entry) is True
        
        # Test invalid values
        with pytest.raises(ValueError):
            HabitEntry(
                date=date(2025, 1, 3),
                sleep_hours=25.0,  # Too high
                exercise_minutes=30,
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
        
        with pytest.raises(ValueError):
            HabitEntry(
                date=date(2025, 1, 3),
                sleep_hours=7.5,
                exercise_minutes=-10,  # Negative
                screen_time_hours=4.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=4,
                productivity_rating=4
            )
    
    def test_duplicate_dates(self, temp_db):
        """Test handling of duplicate dates."""
        db_manager = DatabaseManager(temp_db)
        
        # Add first entry
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
        assert db_manager.save_entry(entry1) is True
        
        # Add entry with same date (should update)
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
        assert db_manager.save_entry(entry2) is True
        
        # Verify only one entry exists for that date
        tracker = db_manager.load_tracker()
        assert len(tracker.entries) == 1
        
        # Verify the updated entry
        retrieved_entry = db_manager.get_entry(date(2025, 1, 1))
        assert retrieved_entry.sleep_hours == 8.0
        assert retrieved_entry.exercise_minutes == 45
        assert retrieved_entry.mood_rating == 5
    
    def test_missing_data_handling(self, temp_db):
        """Test handling of missing or incomplete data."""
        db_manager = DatabaseManager(temp_db)
        
        # Test with minimal required data
        minimal_entry = HabitEntry(
            date=date(2025, 1, 1),
            sleep_hours=7.5,
            exercise_minutes=30,
            screen_time_hours=4.0,
            water_glasses=8,
            work_hours=8.0,
            mood_rating=4,
            productivity_rating=4
            # No notes
        )
        assert db_manager.save_entry(minimal_entry) is True
        
        # Test pattern detection with minimal data
        tracker = db_manager.load_tracker()
        pattern_detector = PatternDetector(tracker)
        
        # Should still work with minimal data
        correlations = pattern_detector.analyze_correlations()
        assert correlations != {}
        
        insights = pattern_detector.generate_insights()
        assert len(insights) > 0


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
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
    
    def test_memory_usage(self, temp_db):
        """Test memory usage with large datasets."""
        db_manager = DatabaseManager(temp_db)
        
        # Add 1000 entries (using valid dates)
        start_date = date(2025, 1, 1)
        for i in range(1000):
            entry_date = start_date + timedelta(days=i)
            entry = HabitEntry(
                date=entry_date,
                sleep_hours=7.0 + (i % 3) * 0.5,
                exercise_minutes=30 if i % 2 == 0 else 0,
                screen_time_hours=4.0 + (i % 2) * 2.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=3 + (i % 3),
                productivity_rating=3 + (i % 3)
            )
            db_manager.save_entry(entry)
        
        # Test loading all data
        tracker = db_manager.load_tracker()
        assert len(tracker.entries) == 1000
        
        # Test pattern detection with large dataset
        pattern_detector = PatternDetector(tracker)
        patterns = pattern_detector.detect_patterns()
        assert 'sleep' in patterns
        
        # Test visualizations with large dataset
        visualizer = HabitVisualizer(tracker)
        dashboard = visualizer.create_dashboard()
        assert dashboard is not None
    
    def test_query_performance(self, temp_db):
        """Test query performance with different date ranges."""
        db_manager = DatabaseManager(temp_db)
        
        # Add entries over a year (including leap year)
        start_date = date(2024, 1, 1)
        for i in range(366):  # 2024 is a leap year
            entry_date = start_date + timedelta(days=i)
            entry = HabitEntry(
                date=entry_date,
                sleep_hours=7.0 + (i % 3) * 0.5,
                exercise_minutes=30 if i % 2 == 0 else 0,
                screen_time_hours=4.0 + (i % 2) * 2.0,
                water_glasses=8,
                work_hours=8.0,
                mood_rating=3 + (i % 3),
                productivity_rating=3 + (i % 3)
            )
            db_manager.save_entry(entry)
        
        # Test different date range queries
        # Recent week
        recent_entries = db_manager.get_entries_in_range(
            date(2024, 12, 25), date(2024, 12, 31)
        )
        assert len(recent_entries) == 7  # 25, 26, 27, 28, 29, 30, 31
        
        # Recent month
        recent_month = db_manager.get_entries_in_range(
            date(2024, 12, 1), date(2024, 12, 31)
        )
        assert len(recent_month) == 31
        
        # Full year
        full_year = db_manager.get_entries_in_range(
            date(2024, 1, 1), date(2024, 12, 31)
        )
        assert len(full_year) == 366  # 2024 is a leap year
