"""
Tests for pattern detection and analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from src.models.habit_model import HabitEntry, HabitTracker
from src.analysis.pattern_detector import PatternDetector
from src.analysis.visualizer import HabitVisualizer


class TestPatternDetector:
    """Test PatternDetector class."""
    
    @pytest.fixture
    def sample_tracker(self):
        """Create a sample tracker with test data."""
        tracker = HabitTracker()
        
        # Add sample data for testing
        for i in range(1, 15):  # 14 days of data
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=7.0 + (i % 3) * 0.5,  # Varying sleep
                exercise_minutes=30 if i % 2 == 0 else 0,  # Every other day
                screen_time_hours=4.0 + (i % 2) * 2.0,  # Varying screen time
                water_glasses=8,
                work_hours=8.0,
                mood_rating=3 + (i % 3),  # Varying mood
                productivity_rating=3 + (i % 3)  # Varying productivity
            )
            tracker.add_entry(entry)
        
        return tracker
    
    def test_empty_tracker(self):
        """Test pattern detector with empty tracker."""
        tracker = HabitTracker()
        detector = PatternDetector(tracker)
        
        # Test correlations with empty data
        correlations = detector.analyze_correlations()
        assert correlations == {}
        
        # Test patterns with empty data
        patterns = detector.detect_patterns()
        assert patterns == {'message': 'Need at least 7 days of data for pattern detection'}
        
        # Test insights with empty data
        insights = detector.generate_insights()
        assert insights == ["Start logging your habits to get personalized insights!"]
        
        # Test recommendations with empty data
        recommendations = detector.get_recommendations()
        assert recommendations == ["Start by logging your daily habits for at least a week to get personalized recommendations."]
    
    def test_analyze_correlations(self, sample_tracker):
        """Test correlation analysis."""
        detector = PatternDetector(sample_tracker)
        correlations = detector.analyze_correlations()
        
        assert 'mood_correlations' in correlations
        assert 'productivity_correlations' in correlations
        assert 'correlation_matrix' in correlations
        
        # Check that correlations are calculated
        mood_corr = correlations['mood_correlations']
        assert 'sleep_hours' in mood_corr
        assert 'exercise_minutes' in mood_corr
        assert 'screen_time_hours' in mood_corr
        
        # Check that correlation matrix is stored
        assert detector.correlation_matrix is not None
    
    def test_detect_patterns(self, sample_tracker):
        """Test pattern detection."""
        detector = PatternDetector(sample_tracker)
        patterns = detector.detect_patterns()
        
        # Check that all pattern types are present
        assert 'sleep' in patterns
        assert 'exercise' in patterns
        assert 'screen_time' in patterns
        assert 'weekly' in patterns
        assert 'streaks' in patterns
        
        # Check sleep patterns
        sleep_patterns = patterns['sleep']
        assert 'avg_sleep' in sleep_patterns
        assert 'sleep_consistency' in sleep_patterns
        
        # Check exercise patterns
        exercise_patterns = patterns['exercise']
        assert 'avg_exercise_minutes' in exercise_patterns
        assert 'exercise_frequency' in exercise_patterns
        
        # Check streaks
        streaks = patterns['streaks']
        assert 'exercise_streaks' in streaks
        assert 'good_sleep_streaks' in streaks
        assert 'high_productivity_streaks' in streaks
    
    def test_analyze_sleep_patterns(self, sample_tracker):
        """Test sleep pattern analysis."""
        detector = PatternDetector(sample_tracker)
        sleep_patterns = detector._analyze_sleep_patterns()
        
        assert 'avg_sleep' in sleep_patterns
        assert 'sleep_consistency' in sleep_patterns
        assert 'optimal_sleep_ranges' in sleep_patterns
        
        # Check that sleep data is reasonable
        assert 6.0 <= sleep_patterns['avg_sleep'] <= 9.0
        assert sleep_patterns['sleep_consistency'] >= 0
    
    def test_analyze_exercise_patterns(self, sample_tracker):
        """Test exercise pattern analysis."""
        detector = PatternDetector(sample_tracker)
        exercise_patterns = detector._analyze_exercise_patterns()
        
        assert 'avg_exercise_minutes' in exercise_patterns
        assert 'exercise_frequency' in exercise_patterns
        assert 'exercise_consistency' in exercise_patterns
        
        # Check exercise impact if both exercise and no-exercise days exist
        if 'exercise_impact' in exercise_patterns:
            impact = exercise_patterns['exercise_impact']
            assert 'mood_improvement' in impact
            assert 'productivity_improvement' in impact
    
    def test_analyze_screen_time_patterns(self, sample_tracker):
        """Test screen time pattern analysis."""
        detector = PatternDetector(sample_tracker)
        screen_patterns = detector._analyze_screen_time_patterns()
        
        assert 'avg_screen_time' in screen_patterns
        assert 'screen_time_consistency' in screen_patterns
        
        # Check screen impact if both high and low screen time days exist
        if 'screen_impact' in screen_patterns and screen_patterns['screen_impact']:
            impact = screen_patterns['screen_impact']
            assert 'mood_difference' in impact
            assert 'productivity_difference' in impact
    
    def test_analyze_weekly_patterns(self, sample_tracker):
        """Test weekly pattern analysis."""
        detector = PatternDetector(sample_tracker)
        weekly_patterns = detector._analyze_weekly_patterns()
        
        assert 'weekly_averages' in weekly_patterns
        assert 'best_mood_day' in weekly_patterns
        assert 'best_productivity_day' in weekly_patterns
        assert 'worst_mood_day' in weekly_patterns
        assert 'worst_productivity_day' in weekly_patterns
    
    def test_analyze_streaks(self, sample_tracker):
        """Test streak analysis."""
        detector = PatternDetector(sample_tracker)
        streaks = detector._analyze_streaks()
        
        assert 'exercise_streaks' in streaks
        assert 'good_sleep_streaks' in streaks
        assert 'high_productivity_streaks' in streaks
        
        # Check streak structure
        for streak_type in streaks.values():
            assert 'max_streak' in streak_type
            assert 'avg_streak' in streak_type
            assert 'total_streaks' in streak_type
    
    def test_calculate_streaks(self, sample_tracker):
        """Test streak calculation."""
        detector = PatternDetector(sample_tracker)
        
        # Test with alternating True/False
        condition = pd.Series([True, False, True, False, True])
        streaks = detector._calculate_streaks(condition)
        
        assert streaks['max_streak'] == 1
        assert streaks['avg_streak'] == 1.0
        assert streaks['total_streaks'] == 3
        
        # Test with consecutive True values
        condition = pd.Series([True, True, False, True, True, True])
        streaks = detector._calculate_streaks(condition)
        
        assert streaks['max_streak'] == 3
        assert streaks['total_streaks'] == 2
    
    def test_predict_productivity(self, sample_tracker):
        """Test productivity prediction."""
        detector = PatternDetector(sample_tracker)
        predictions = detector.predict_productivity()
        
        # Should have enough data for predictions
        assert 'predicted_productivity' in predictions
        assert 'confidence' in predictions
        assert 'feature_importance' in predictions
        assert 'recent_averages' in predictions
        
        # Check prediction values are reasonable
        assert 1.0 <= predictions['predicted_productivity'] <= 5.0
        assert 0.0 <= predictions['confidence'] <= 1.0
        
        # Check feature importance
        feature_importance = predictions['feature_importance']
        assert 'sleep_hours' in feature_importance
        assert 'exercise_minutes' in feature_importance
        assert 'screen_time_hours' in feature_importance
        
        # Check that importance values sum to approximately 1
        total_importance = sum(feature_importance.values())
        assert 0.9 <= total_importance <= 1.1
    
    def test_predict_productivity_insufficient_data(self):
        """Test productivity prediction with insufficient data."""
        tracker = HabitTracker()
        
        # Add only 5 days of data (less than required 14)
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
        
        detector = PatternDetector(tracker)
        predictions = detector.predict_productivity()
        
        assert 'message' in predictions
        assert 'Need at least 14 days of data' in predictions['message']
    
    def test_generate_insights(self, sample_tracker):
        """Test insight generation."""
        detector = PatternDetector(sample_tracker)
        insights = detector.generate_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check that insights are strings
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0
    
    def test_get_recommendations(self, sample_tracker):
        """Test recommendation generation."""
        detector = PatternDetector(sample_tracker)
        recommendations = detector.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations are strings
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0


class TestHabitVisualizer:
    """Test HabitVisualizer class."""
    
    @pytest.fixture
    def sample_tracker(self):
        """Create a sample tracker with test data."""
        tracker = HabitTracker()
        
        # Add sample data for testing
        for i in range(1, 8):  # 7 days of data
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
    
    def test_empty_tracker(self):
        """Test visualizer with empty tracker."""
        tracker = HabitTracker()
        visualizer = HabitVisualizer(tracker)
        
        # Test dashboard with empty data
        fig = visualizer.create_dashboard()
        assert fig is not None
        
        # Test correlation heatmap with empty data
        fig = visualizer.create_correlation_heatmap()
        assert fig is not None
        
        # Test trend analysis with empty data
        fig = visualizer.create_trend_analysis()
        assert fig is not None
        
        # Test weekly summary with empty data
        fig = visualizer.create_weekly_summary()
        assert fig is not None
    
    def test_create_dashboard(self, sample_tracker):
        """Test dashboard creation."""
        visualizer = HabitVisualizer(sample_tracker)
        fig = visualizer.create_dashboard()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Check that dashboard has the expected layout
        layout = fig.layout
        assert layout.title.text == "AI Habit Tracker Dashboard"
    
    def test_create_correlation_heatmap(self, sample_tracker):
        """Test correlation heatmap creation."""
        visualizer = HabitVisualizer(sample_tracker)
        fig = visualizer.create_correlation_heatmap()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Check that heatmap has the expected layout
        layout = fig.layout
        assert layout.title.text == "Habit Correlations"
    
    def test_create_trend_analysis(self, sample_tracker):
        """Test trend analysis creation."""
        visualizer = HabitVisualizer(sample_tracker)
        fig = visualizer.create_trend_analysis()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Check that trend analysis has the expected layout
        layout = fig.layout
        assert layout.title.text == "Habit Trends Over Time"
    
    def test_create_weekly_summary(self, sample_tracker):
        """Test weekly summary creation."""
        visualizer = HabitVisualizer(sample_tracker)
        fig = visualizer.create_weekly_summary()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Check that weekly summary has the expected layout
        layout = fig.layout
        assert layout.title.text == "Weekly Habit Patterns"
    
    def test_save_all_charts(self, sample_tracker, tmp_path):
        """Test saving all charts."""
        visualizer = HabitVisualizer(sample_tracker)
        
        # Create output directory
        output_dir = tmp_path / "charts"
        output_dir.mkdir()
        
        # Save all charts
        saved_files = visualizer.save_all_charts(str(output_dir))
        
        assert isinstance(saved_files, dict)
        assert 'dashboard' in saved_files
        assert 'correlation_heatmap' in saved_files
        assert 'trend_analysis' in saved_files
        assert 'weekly_summary' in saved_files
        
        # Check that files were actually created
        for file_path in saved_files.values():
            assert os.path.exists(file_path)
            assert file_path.endswith('.html')
    
    def test_create_empty_plot(self, sample_tracker):
        """Test creating empty plot with message."""
        visualizer = HabitVisualizer(sample_tracker)
        message = "Test message"
        fig = visualizer._create_empty_plot(message)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Check that the message is in the figure
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert message in str(annotations[0].text)
    
    @patch('os.makedirs')
    @patch('os.path.join')
    def test_save_all_charts_directory_creation(self, mock_join, mock_makedirs, sample_tracker):
        """Test that save_all_charts creates directories."""
        visualizer = HabitVisualizer(sample_tracker)
        
        # Mock the file operations
        mock_join.return_value = "test_path"
        
        # Mock the write_html method to avoid file system issues
        with patch.object(visualizer, 'create_dashboard') as mock_dashboard, \
             patch.object(visualizer, 'create_correlation_heatmap') as mock_heatmap, \
             patch.object(visualizer, 'create_trend_analysis') as mock_trends, \
             patch.object(visualizer, 'create_weekly_summary') as mock_weekly:
            
            mock_fig = MagicMock()
            mock_dashboard.return_value = mock_fig
            mock_heatmap.return_value = mock_fig
            mock_trends.return_value = mock_fig
            mock_weekly.return_value = mock_fig
            
            # Call save_all_charts
            visualizer.save_all_charts("test_output_dir")
        
        # Check that makedirs was called
        mock_makedirs.assert_called_once_with("test_output_dir", exist_ok=True)


# Additional integration tests
class TestAnalysisIntegration:
    """Integration tests for analysis functionality."""
    
    def test_full_analysis_workflow(self):
        """Test the complete analysis workflow."""
        # Create tracker with realistic data
        tracker = HabitTracker()
        
        # Add 30 days of realistic data
        for i in range(1, 31):
            # Simulate realistic patterns
            day_of_week = (date(2025, 1, i).weekday())
            
            # Sleep: 7-8 hours on weekdays, 8-9 on weekends
            sleep_hours = 7.5 + (day_of_week >= 5) * 0.5 + np.random.normal(0, 0.5)
            sleep_hours = max(6.0, min(10.0, sleep_hours))
            
            # Exercise: 30-60 min on weekdays, 0-30 on weekends
            exercise_minutes = 45 if day_of_week < 5 else 15
            exercise_minutes += np.random.normal(0, 15)
            exercise_minutes = max(0, min(120, exercise_minutes))
            
            # Screen time: 4-6 hours on weekdays, 6-8 on weekends
            screen_time_hours = 5.0 + (day_of_week >= 5) * 1.5 + np.random.normal(0, 1.0)
            screen_time_hours = max(2.0, min(10.0, screen_time_hours))
            
            # Mood and productivity correlate with exercise and sleep
            base_mood = 3.0
            if exercise_minutes > 30:
                base_mood += 0.5
            if sleep_hours >= 7.5:
                base_mood += 0.3
            if screen_time_hours < 6.0:
                base_mood += 0.2
            
            mood_rating = max(1, min(5, int(base_mood + np.random.normal(0, 0.5))))
            productivity_rating = max(1, min(5, mood_rating + np.random.normal(0, 0.3)))
            
            entry = HabitEntry(
                date=date(2025, 1, i),
                sleep_hours=sleep_hours,
                exercise_minutes=int(exercise_minutes),
                screen_time_hours=screen_time_hours,
                water_glasses=8,
                work_hours=8.0 if day_of_week < 5 else 0.0,
                mood_rating=mood_rating,
                productivity_rating=productivity_rating
            )
            tracker.add_entry(entry)
        
        # Test pattern detector
        detector = PatternDetector(tracker)
        
        # Test correlations
        correlations = detector.analyze_correlations()
        assert correlations != {}
        assert 'mood_correlations' in correlations
        assert 'productivity_correlations' in correlations
        
        # Test patterns
        patterns = detector.detect_patterns()
        assert 'sleep' in patterns
        assert 'exercise' in patterns
        assert 'weekly' in patterns
        
        # Test predictions
        predictions = detector.predict_productivity()
        assert 'predicted_productivity' in predictions
        assert 'confidence' in predictions
        
        # Test insights
        insights = detector.generate_insights()
        assert len(insights) > 0
        
        # Test recommendations
        recommendations = detector.get_recommendations()
        assert len(recommendations) > 0
        
        # Test visualizer
        visualizer = HabitVisualizer(tracker)
        
        # Test dashboard
        fig = visualizer.create_dashboard()
        assert fig is not None
        
        # Test correlation heatmap
        fig = visualizer.create_correlation_heatmap()
        assert fig is not None
        
        # Test trend analysis
        fig = visualizer.create_trend_analysis()
        assert fig is not None
        
        # Test weekly summary
        fig = visualizer.create_weekly_summary()
        assert fig is not None
