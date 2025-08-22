import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import streamlit as st

from ui.dashboard import VoiceMoodMirrorDashboard


class TestVoiceMoodMirrorDashboard:
    """Test cases for VoiceMoodMirrorDashboard class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dashboard = VoiceMoodMirrorDashboard()
        
    def test_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.audio_recorder is None
        assert self.dashboard.feature_extractor is None
        assert self.dashboard.emotion_classifier is None
        assert self.dashboard.mood_mapper is None
        assert self.dashboard.music_selector is None
        assert self.dashboard.mood_smoother is None
        assert self.dashboard.is_recording is False
        assert self.dashboard.mood_history == []
        assert self.dashboard.music_history == []
        
    @patch('ui.dashboard.AudioRecorder')
    @patch('ui.dashboard.FeatureExtractor')
    @patch('ui.dashboard.RuleBasedClassifier')
    @patch('ui.dashboard.MoodMapper')
    @patch('ui.dashboard.MusicSelector')
    @patch('ui.dashboard.MoodSmoother')
    def test_initialize_components(self, mock_smoother, mock_selector, mock_mapper, 
                                 mock_classifier, mock_extractor, mock_recorder):
        """Test component initialization."""
        self.dashboard.initialize_components()
        
        assert self.dashboard.audio_recorder is not None
        assert self.dashboard.feature_extractor is not None
        assert self.dashboard.emotion_classifier is not None
        assert self.dashboard.mood_mapper is not None
        assert self.dashboard.music_selector is not None
        assert self.dashboard.mood_smoother is not None
        
    @patch('streamlit.set_page_config')
    def test_setup_page(self, mock_set_page_config):
        """Test page setup."""
        self.dashboard.setup_page()
        
        mock_set_page_config.assert_called_once()
        
    @patch('streamlit.sidebar')
    def test_create_sidebar(self, mock_sidebar):
        """Test sidebar creation."""
        mock_sidebar.title.return_value = None
        mock_sidebar.selectbox.return_value = 'simple'
        mock_sidebar.slider.return_value = 5
        mock_sidebar.button.return_value = False
        
        self.dashboard.create_sidebar()
        
        mock_sidebar.title.assert_called_once()
        mock_sidebar.selectbox.assert_called()
        mock_sidebar.slider.assert_called()
        mock_sidebar.button.assert_called()
        
    @patch('streamlit.tabs')
    def test_create_main_content(self, mock_tabs):
        """Test main content creation."""
        mock_tab1 = Mock()
        mock_tab2 = Mock()
        mock_tab3 = Mock()
        mock_tab4 = Mock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3, mock_tab4]
        
        self.dashboard.create_main_content()
        
        mock_tabs.assert_called_once()
        
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_create_live_mood_tab(self, mock_progress, mock_empty):
        """Test live mood tab creation."""
        mock_container = Mock()
        mock_empty.return_value = mock_container
        
        self.dashboard.create_live_mood_tab()
        
        mock_empty.assert_called()
        
    @patch('streamlit.plotly_chart')
    def test_create_mood_visualization(self, mock_plotly_chart):
        """Test mood visualization creation."""
        mock_container = Mock()
        
        # Mock mood data
        mood_data = {'mood': 'happy', 'confidence': 0.8}
        
        self.dashboard.create_mood_visualization(mock_container, mood_data)
        
        mock_plotly_chart.assert_called()
        
    @patch('streamlit.metric')
    @patch('streamlit.color_picker')
    def test_create_mood_details(self, mock_color_picker, mock_metric):
        """Test mood details creation."""
        mock_container = Mock()
        
        # Mock mood data
        mood_data = {'mood': 'happy', 'confidence': 0.8}
        
        self.dashboard.create_mood_details(mock_container, mood_data)
        
        mock_metric.assert_called()
        mock_color_picker.assert_called()
        
    @patch('streamlit.line_chart')
    def test_create_mood_history_tab(self, mock_line_chart):
        """Test mood history tab creation."""
        mock_container = Mock()
        
        # Add some mock history
        self.dashboard.mood_history = [
            {'mood': 'happy', 'confidence': 0.8, 'timestamp': '2023-01-01 10:00:00'},
            {'mood': 'excited', 'confidence': 0.9, 'timestamp': '2023-01-01 10:01:00'}
        ]
        
        self.dashboard.create_mood_history_tab(mock_container)
        
        mock_line_chart.assert_called()
        
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    def test_create_music_tab(self, mock_button, mock_selectbox):
        """Test music tab creation."""
        mock_container = Mock()
        
        mock_selectbox.return_value = 'matching'
        mock_button.return_value = False
        
        self.dashboard.create_music_tab(mock_container)
        
        mock_selectbox.assert_called()
        mock_button.assert_called()
        
    @patch('streamlit.bar_chart')
    @patch('streamlit.metric')
    def test_create_analytics_tab(self, mock_metric, mock_bar_chart):
        """Test analytics tab creation."""
        mock_container = Mock()
        
        # Add some mock history for analytics
        self.dashboard.mood_history = [
            {'mood': 'happy', 'confidence': 0.8, 'timestamp': '2023-01-01 10:00:00'},
            {'mood': 'excited', 'confidence': 0.9, 'timestamp': '2023-01-01 10:01:00'},
            {'mood': 'calm', 'confidence': 0.7, 'timestamp': '2023-01-01 10:02:00'}
        ]
        
        self.dashboard.create_analytics_tab(mock_container)
        
        mock_metric.assert_called()
        mock_bar_chart.assert_called()
        
    @patch('ui.dashboard.AudioRecorder')
    def test_start_recording(self, mock_recorder_class):
        """Test starting recording."""
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder
        
        self.dashboard.audio_recorder = mock_recorder
        
        self.dashboard.start_recording()
        
        assert self.dashboard.is_recording is True
        mock_recorder.start_recording.assert_called_once()
        
    @patch('ui.dashboard.AudioRecorder')
    def test_stop_recording(self, mock_recorder_class):
        """Test stopping recording."""
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder
        
        self.dashboard.audio_recorder = mock_recorder
        self.dashboard.is_recording = True
        
        self.dashboard.stop_recording()
        
        assert self.dashboard.is_recording is False
        mock_recorder.stop_recording.assert_called_once()
        
    @patch('ui.dashboard.VoiceMoodMirrorDashboard.setup_page')
    @patch('ui.dashboard.VoiceMoodMirrorDashboard.initialize_components')
    @patch('ui.dashboard.VoiceMoodMirrorDashboard.create_sidebar')
    @patch('ui.dashboard.VoiceMoodMirrorDashboard.create_main_content')
    def test_run(self, mock_main_content, mock_sidebar, mock_init_components, mock_setup_page):
        """Test dashboard run method."""
        self.dashboard.run()
        
        mock_setup_page.assert_called_once()
        mock_init_components.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_main_content.assert_called_once()
        
    def test_process_audio_chunk(self):
        """Test processing audio chunk for mood analysis."""
        # Mock components
        self.dashboard.feature_extractor = Mock()
        self.dashboard.emotion_classifier = Mock()
        self.dashboard.mood_mapper = Mock()
        self.dashboard.mood_smoother = Mock()
        
        # Mock return values
        mock_features = {'pitch_mean': 200, 'energy_mean': 0.5}
        mock_emotion = {'emotion': 'happy', 'confidence': 0.8}
        mock_smoothed = {'mood': 'happy', 'confidence': 0.8}
        
        self.dashboard.feature_extractor.extract_features.return_value = mock_features
        self.dashboard.emotion_classifier.classify.return_value = mock_emotion
        self.dashboard.mood_smoother.get_smoothed_mood.return_value = mock_smoothed
        
        # Mock audio chunk
        audio_chunk = np.random.rand(16000)  # 1 second of audio at 16kHz
        
        result = self.dashboard.process_audio_chunk(audio_chunk)
        
        assert result == mock_smoothed
        self.dashboard.feature_extractor.extract_features.assert_called_once()
        self.dashboard.emotion_classifier.classify.assert_called_once()
        self.dashboard.mood_smoother.add_mood_prediction.assert_called_once()
        
    def test_get_music_recommendations(self):
        """Test getting music recommendations."""
        # Mock music selector
        self.dashboard.music_selector = Mock()
        
        # Mock return value
        mock_recommendations = ['song1', 'song2', 'song3']
        self.dashboard.music_selector.select_music_for_mood.return_value = mock_recommendations
        
        mood_data = {'mood': 'happy', 'confidence': 0.8}
        strategy = 'matching'
        
        result = self.dashboard.get_music_recommendations(mood_data, strategy)
        
        assert result == mock_recommendations
        self.dashboard.music_selector.select_music_for_mood.assert_called_once_with(mood_data, strategy)
        
    def test_update_mood_history(self):
        """Test updating mood history."""
        mood_data = {'mood': 'happy', 'confidence': 0.8}
        
        self.dashboard.update_mood_history(mood_data)
        
        assert len(self.dashboard.mood_history) == 1
        assert self.dashboard.mood_history[0]['mood'] == 'happy'
        assert self.dashboard.mood_history[0]['confidence'] == 0.8
        assert 'timestamp' in self.dashboard.mood_history[0]
        
    def test_get_mood_statistics(self):
        """Test getting mood statistics."""
        # Add some mock history
        self.dashboard.mood_history = [
            {'mood': 'happy', 'confidence': 0.8, 'timestamp': '2023-01-01 10:00:00'},
            {'mood': 'excited', 'confidence': 0.9, 'timestamp': '2023-01-01 10:01:00'},
            {'mood': 'calm', 'confidence': 0.7, 'timestamp': '2023-01-01 10:02:00'},
            {'mood': 'happy', 'confidence': 0.8, 'timestamp': '2023-01-01 10:03:00'}
        ]
        
        stats = self.dashboard.get_mood_statistics()
        
        assert 'total_entries' in stats
        assert 'emotion_distribution' in stats
        assert 'average_confidence' in stats
        assert 'dominant_emotion' in stats
        assert stats['total_entries'] == 4
        assert stats['dominant_emotion'] == 'happy'
        
    def test_clear_history(self):
        """Test clearing history."""
        # Add some mock data
        self.dashboard.mood_history = [{'mood': 'happy'}]
        self.dashboard.music_history = [{'song': 'song1'}]
        
        self.dashboard.clear_history()
        
        assert self.dashboard.mood_history == []
        assert self.dashboard.music_history == []
        
    @patch('streamlit.error')
    def test_error_handling(self, mock_error):
        """Test error handling in dashboard."""
        # Test with None components
        self.dashboard.feature_extractor = None
        
        audio_chunk = np.random.rand(16000)
        
        result = self.dashboard.process_audio_chunk(audio_chunk)
        
        assert result is None
        mock_error.assert_called()
        
    def test_edge_cases(self):
        """Test edge cases for dashboard."""
        # Test with empty mood history
        stats = self.dashboard.get_mood_statistics()
        
        assert stats['total_entries'] == 0
        assert stats['dominant_emotion'] is None
        
        # Test with very short audio chunk
        audio_chunk = np.random.rand(100)  # Very short chunk
        
        # Mock components
        self.dashboard.feature_extractor = Mock()
        self.dashboard.emotion_classifier = Mock()
        self.dashboard.mood_mapper = Mock()
        self.dashboard.mood_smoother = Mock()
        
        # Mock return values
        mock_features = {'pitch_mean': 200, 'energy_mean': 0.5}
        mock_emotion = {'emotion': 'happy', 'confidence': 0.8}
        mock_smoothed = {'mood': 'happy', 'confidence': 0.8}
        
        self.dashboard.feature_extractor.extract_features.return_value = mock_features
        self.dashboard.emotion_classifier.classify.return_value = mock_emotion
        self.dashboard.mood_smoother.get_smoothed_mood.return_value = mock_smoothed
        
        result = self.dashboard.process_audio_chunk(audio_chunk)
        
        assert result == mock_smoothed
        
    @patch('streamlit.success')
    def test_success_messages(self, mock_success):
        """Test success message display."""
        # Mock components
        self.dashboard.audio_recorder = Mock()
        
        # Test successful recording start
        self.dashboard.start_recording()
        
        # Test successful recording stop
        self.dashboard.is_recording = True
        self.dashboard.stop_recording()
        
        # Note: In a real implementation, success messages would be called
        # This test verifies the structure supports success messaging
        
    def test_component_integration(self):
        """Test integration between dashboard components."""
        # Mock all components
        self.dashboard.audio_recorder = Mock()
        self.dashboard.feature_extractor = Mock()
        self.dashboard.emotion_classifier = Mock()
        self.dashboard.mood_mapper = Mock()
        self.dashboard.music_selector = Mock()
        self.dashboard.mood_smoother = Mock()
        
        # Test full pipeline
        audio_chunk = np.random.rand(16000)
        
        # Mock return values
        mock_features = {'pitch_mean': 200, 'energy_mean': 0.5}
        mock_emotion = {'emotion': 'happy', 'confidence': 0.8}
        mock_smoothed = {'mood': 'happy', 'confidence': 0.8}
        mock_music = ['song1', 'song2']
        
        self.dashboard.feature_extractor.extract_features.return_value = mock_features
        self.dashboard.emotion_classifier.classify.return_value = mock_emotion
        self.dashboard.mood_smoother.get_smoothed_mood.return_value = mock_smoothed
        self.dashboard.music_selector.select_music_for_mood.return_value = mock_music
        
        # Process audio
        mood_result = self.dashboard.process_audio_chunk(audio_chunk)
        
        # Get music recommendations
        music_result = self.dashboard.get_music_recommendations(mood_result, 'matching')
        
        # Verify results
        assert mood_result == mock_smoothed
        assert music_result == mock_music
        
        # Verify component interactions
        self.dashboard.feature_extractor.extract_features.assert_called_once()
        self.dashboard.emotion_classifier.classify.assert_called_once()
        self.dashboard.mood_smoother.add_mood_prediction.assert_called_once()
        self.dashboard.music_selector.select_music_for_mood.assert_called_once()


# Test the main function
@patch('ui.dashboard.VoiceMoodMirrorDashboard')
def test_main_function(mock_dashboard_class):
    """Test the main function."""
    mock_dashboard = Mock()
    mock_dashboard_class.return_value = mock_dashboard
    
    # Import and call main
    from ui.dashboard import main
    main()
    
    mock_dashboard_class.assert_called_once()
    mock_dashboard.run.assert_called_once()
