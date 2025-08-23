"""
Streamlit dashboard for VoiceMoodMirror.
Provides real-time mood visualization and music recommendations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.recorder import AudioRecorder, AudioBuffer
from audio.feature_extractor import FeatureExtractor
from emotion.prosody_classifier import RuleBasedClassifier
from emotion.mood_mapper import MoodMapper
from music.music_selector import MusicSelector
from music.playlist_builder import PlaylistBuilder
from utils.smoothing import AdaptiveSmoother


class VoiceMoodMirrorDashboard:
    """Main dashboard class for VoiceMoodMirror."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.audio_recorder = AudioRecorder()
        self.feature_extractor = FeatureExtractor()
        self.emotion_classifier = RuleBasedClassifier()
        self.mood_mapper = MoodMapper()
        self.music_selector = MusicSelector()
        self.playlist_builder = PlaylistBuilder(self.music_selector)
        self.mood_smoother = AdaptiveSmoother()
        
        # Audio buffer for real-time processing
        self.audio_buffer = AudioBuffer(max_duration=5.0)  # 5 seconds buffer
        
        # State variables
        self.is_recording = False
        self.mood_history = []
        self.current_playlist = []
        
        # Initialize session state
        if 'mood_data' not in st.session_state:
            st.session_state.mood_data = []
        if 'playlist_data' not in st.session_state:
            st.session_state.playlist_data = []
    
    def setup_page(self):
        """Setup the Streamlit page configuration."""
        st.set_page_config(
            page_title="VoiceMoodMirror",
            page_icon="🎧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎧 VoiceMoodMirror")
        st.markdown("### Real-Time Voice Emotion Analyzer & Feedback")
    
    def create_sidebar(self):
        """Create the sidebar with controls and settings."""
        st.sidebar.header("🎛️ Controls")
        
        # Recording controls
        st.sidebar.subheader("🎤 Recording")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🎙️ Start Recording", key="start_rec"):
                self.start_recording()
        
        with col2:
            if st.button("⏹️ Stop Recording", key="stop_rec"):
                self.stop_recording()
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        
        smoothing_method = st.sidebar.selectbox(
            "Smoothing Method",
            ["exponential", "weighted", "simple"],
            index=0
        )
        
        window_size = st.sidebar.slider(
            "Smoothing Window Size",
            min_value=3,
            max_value=15,
            value=5
        )
        
        # Music strategy
        music_strategy = st.sidebar.selectbox(
            "Music Strategy",
            ["match", "modulate", "adaptive"],
            index=2
        )
        
        playlist_duration = st.sidebar.slider(
            "Playlist Duration (minutes)",
            min_value=10,
            max_value=60,
            value=30
        )
        
        # Display current status
        st.sidebar.subheader("📊 Status")
        
        if self.is_recording:
            st.sidebar.success("🎙️ Recording Active")
        else:
            st.sidebar.info("⏸️ Not Recording")
        
        # Show current mood if available
        if st.session_state.mood_data:
            latest_mood = st.session_state.mood_data[-1]
            st.sidebar.metric(
                "Current Mood",
                f"{latest_mood['emotion'].title()}",
                f"{latest_mood['confidence']:.1%}"
            )
        
        return {
            'smoothing_method': smoothing_method,
            'window_size': window_size,
            'music_strategy': music_strategy,
            'playlist_duration': playlist_duration
        }
    
    def create_main_content(self, settings):
        """Create the main content area."""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎭 Live Mood", "📈 Mood History", "🎵 Music", "📊 Analytics"])
        
        with tab1:
            self.create_live_mood_tab()
        
        with tab2:
            self.create_mood_history_tab()
        
        with tab3:
            self.create_music_tab(settings)
        
        with tab4:
            self.create_analytics_tab()
    
    def create_live_mood_tab(self):
        """Create the live mood visualization tab."""
        st.header("🎭 Live Mood Analysis")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mood visualization
            if st.session_state.mood_data:
                self.create_mood_visualization()
            else:
                st.info("Start recording to see live mood analysis")
        
        with col2:
            # Current mood details
            if st.session_state.mood_data:
                self.create_mood_details()
            else:
                st.info("No mood data available")
        
        # Real-time updates
        if self.is_recording:
            st.empty()
            time.sleep(0.1)
            st.rerun()
    
    def create_mood_visualization(self):
        """Create the mood visualization."""
        if not st.session_state.mood_data:
            return
        
        latest_mood = st.session_state.mood_data[-1]
        
        # Create mood meter
        fig = go.Figure()
        
        # Emotion color mapping
        emotion_colors = {
            'happy': '#FFD700',
            'excited': '#FFA500',
            'calm': '#00CED1',
            'neutral': '#FFFFFF',
            'tired': '#808080',
            'sad': '#0000FF',
            'angry': '#FF0000'
        }
        
        emotion = latest_mood['emotion']
        confidence = latest_mood['confidence']
        color = emotion_colors.get(emotion, '#FFFFFF')
        
        # Create gauge chart
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Mood: {emotion.title()}"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "gray"},
                    {'range': [40, 60], 'color': "darkgray"},
                    {'range': [60, 80], 'color': "darkslategray"},
                    {'range': [80, 100], 'color': "black"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_mood_details(self):
        """Create detailed mood information."""
        if not st.session_state.mood_data:
            return
        
        latest_mood = st.session_state.mood_data[-1]
        
        st.subheader("🎯 Current Mood Details")
        
        # Emotion emoji and description
        emoji = self.mood_mapper.get_emotion_emoji(latest_mood['emotion'])
        description = self.mood_mapper.get_emotion_description(latest_mood['emotion'])
        
        st.markdown(f"### {emoji} {latest_mood['emotion'].title()}")
        st.write(description)
        
        # Confidence meter
        st.progress(latest_mood['confidence'])
        st.caption(f"Confidence: {latest_mood['confidence']:.1%}")
        
        # Mood suggestions
        st.subheader("💡 Suggestions")
        suggestions = self.mood_mapper.get_mood_enhancement_suggestions(latest_mood['emotion'])
        
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
        
        # Smoothing stats
        if hasattr(self.mood_smoother, 'get_adaptive_stats'):
            stats = self.mood_smoother.get_adaptive_stats()
            st.subheader("📊 Smoothing Stats")
            st.write(f"Window Size: {stats['current_size']}/{stats['window_size']}")
            st.write(f"Stability: {stats['stability']:.1%}")
            st.write(f"Trend: {stats['trend']}")
    
    def create_mood_history_tab(self):
        """Create the mood history tab."""
        st.header("📈 Mood History")
        
        if not st.session_state.mood_data:
            st.info("No mood history available. Start recording to collect data.")
            return
        
        # Create DataFrame for plotting
        df = pd.DataFrame(st.session_state.mood_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Mood over time chart
        fig = px.line(df, x='timestamp', y='confidence', color='emotion',
                     title='Mood Confidence Over Time',
                     labels={'confidence': 'Confidence', 'timestamp': 'Time'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Emotion distribution
        col1, col2 = st.columns(2)
        
        with col1:
            emotion_counts = df['emotion'].value_counts()
            fig_pie = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                           title='Emotion Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Recent mood entries
            st.subheader("Recent Moods")
            recent_data = df.tail(10)[['timestamp', 'emotion', 'confidence']]
            recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(recent_data, use_container_width=True)
    
    def create_music_tab(self, settings):
        """Create the music recommendations tab."""
        st.header("🎵 Music Recommendations")
        
        if not st.session_state.mood_data:
            st.info("No mood data available. Start recording to get music recommendations.")
            return
        
        latest_mood = st.session_state.mood_data[-1]
        
        # Current mood info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Current Mood")
            emoji = self.mood_mapper.get_emotion_emoji(latest_mood['emotion'])
            st.markdown(f"### {emoji} {latest_mood['emotion'].title()}")
        
        with col2:
            st.subheader("🎵 Music Strategy")
            strategy = settings['music_strategy']
            st.write(f"Strategy: {strategy.title()}")
            
            if strategy == 'modulate':
                modulation_map = {
                    'sad': 'happy',
                    'angry': 'calm',
                    'tired': 'excited'
                }
                target_emotion = modulation_map.get(latest_mood['emotion'], 'calm')
                st.write(f"Target: {target_emotion.title()}")
        
        # Generate playlist
        if st.button("🎵 Generate Playlist"):
            with st.spinner("Generating playlist..."):
                if strategy == 'adaptive':
                    playlist = self.playlist_builder.build_adaptive_playlist(
                        latest_mood['emotion'], 
                        settings['playlist_duration']
                    )
                else:
                    playlist = self.music_selector.create_playlist(
                        latest_mood['emotion'],
                        strategy,
                        settings['playlist_duration']
                    )
                
                st.session_state.playlist_data = playlist
                st.success(f"Generated {len(playlist)} songs!")
        
        # Display playlist
        if st.session_state.playlist_data:
            st.subheader("📋 Recommended Playlist")
            
            for i, song in enumerate(st.session_state.playlist_data, 1):
                with st.expander(f"{i}. {song['title']} - {song['artist']}"):
                    st.write(f"**Genre:** {song['genre']}")
                    
                    # Rating system
                    rating = st.slider(f"Rate this song for {latest_mood['emotion']} mood", 
                                     0.0, 1.0, 0.5, key=f"rating_{i}")
                    
                    if st.button(f"Save Rating", key=f"save_{i}"):
                        self.music_selector.add_user_preference(
                            latest_mood['emotion'], song, rating
                        )
                        st.success("Rating saved!")
    
    def create_analytics_tab(self):
        """Create the analytics tab."""
        st.header("📊 Analytics")
        
        if not st.session_state.mood_data:
            st.info("No data available for analytics. Start recording to collect data.")
            return
        
        # Mood statistics
        df = pd.DataFrame(st.session_state.mood_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Mood Entries", len(df))
        
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            most_common = df['emotion'].mode().iloc[0] if not df['emotion'].mode().empty else "None"
            st.metric("Most Common Mood", most_common.title())
        
        # Mood trends
        st.subheader("📈 Mood Trends")
        
        # Daily mood summary
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_mood = df.groupby('date')['emotion'].agg(['count', lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutral']).reset_index()
        daily_mood.columns = ['Date', 'Count', 'Dominant_Mood']
        
        st.write("Daily Mood Summary")
        st.dataframe(daily_mood, use_container_width=True)
        
        # Mood transition matrix
        st.subheader("🔄 Mood Transitions")
        
        if len(df) > 1:
            transitions = []
            for i in range(len(df) - 1):
                transitions.append((df.iloc[i]['emotion'], df.iloc[i + 1]['emotion']))
            
            transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
            transition_matrix = pd.crosstab(transition_df['From'], transition_df['To'])
            
            st.write("Mood Transition Matrix")
            st.dataframe(transition_matrix, use_container_width=True)
    
    def start_recording(self):
        """Start audio recording and processing."""
        if not self.is_recording:
            self.is_recording = True
            
            def audio_callback(audio_data):
                """Callback for audio processing."""
                # Add to buffer
                self.audio_buffer.add_audio(audio_data)
                
                # Process if we have enough audio
                if self.audio_buffer.is_full or len(self.audio_buffer.buffer) > 22050:  # 1 second
                    # Get latest audio
                    audio_chunk = self.audio_buffer.get_latest_audio(3.0)  # 3 seconds
                    
                    if len(audio_chunk) > 0:
                        # Extract features
                        features = self.feature_extractor.extract_features(audio_chunk)
                        
                        # Classify emotion
                        emotion, confidence = self.emotion_classifier.classify(features)
                        
                        # Add to smoother
                        self.mood_smoother.add_mood_prediction(emotion, confidence)
                        
                        # Get smoothed result
                        smoothed_emotion, smoothed_confidence = self.mood_smoother.get_smoothed_mood()
                        
                        # Add to history
                        mood_entry = {
                            'emotion': smoothed_emotion,
                            'confidence': smoothed_confidence,
                            'timestamp': datetime.now().isoformat(),
                            'raw_emotion': emotion,
                            'raw_confidence': confidence
                        }
                        
                        st.session_state.mood_data.append(mood_entry)
            
            # Start recording
            self.audio_recorder.start_recording(audio_callback)
    
    def stop_recording(self):
        """Stop audio recording."""
        if self.is_recording:
            self.is_recording = False
            self.audio_recorder.stop_recording()
    
    def run(self):
        """Run the dashboard."""
        self.setup_page()
        
        # Create sidebar
        settings = self.create_sidebar()
        
        # Create main content
        self.create_main_content(settings)


def main():
    """Main function to run the dashboard."""
    dashboard = VoiceMoodMirrorDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
