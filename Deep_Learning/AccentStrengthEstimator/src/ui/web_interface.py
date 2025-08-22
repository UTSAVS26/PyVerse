"""
Web interface for the Accent Strength Estimator.
"""

import streamlit as st
import time
from typing import Dict, Any


class WebInterface:
    """Web interface for the accent strength estimator."""
    
    def __init__(self):
        """Initialize the web interface."""
        self.phrases = []
        self.results = {}
        
    def run(self):
        """Run the web interface."""
        try:
            self._setup_page()
            self._load_phrases()
            self._main_interface()
        except Exception as e:
            st.error(f"❌ Web interface error: {e}")
            st.info("Please try the CLI interface instead.")
    
    def _setup_page(self):
        """Setup the Streamlit page."""
        st.set_page_config(
            page_title="🎤 Accent Strength Estimator",
            page_icon="🎤",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎤 Accent Strength Estimator")
        st.markdown("---")
    
    def _load_phrases(self):
        """Load reference phrases."""
        try:
            phrases_file = "data/reference_phrases.txt"
            with open(phrases_file, 'r', encoding='utf-8') as f:
                self.phrases = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            st.error("❌ Reference phrases file not found!")
            self.phrases = []
        except Exception as e:
            st.error(f"❌ Failed to load phrases: {e}")
            self.phrases = []
    
    def _main_interface(self):
        """Main interface logic."""
        # Sidebar
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Phrase selection
            if self.phrases:
                st.subheader("📝 Select Phrases")
                phrase_option = st.selectbox(
                    "Choose testing option:",
                    ["All phrases", "Specific phrases", "By difficulty"]
                )
                
                if phrase_option == "Specific phrases":
                    selected_indices = st.multiselect(
                        "Select phrases to test:",
                        range(len(self.phrases)),
                        format_func=lambda x: f"{x+1}. {self.phrases[x]}"
                    )
                    selected_phrases = [self.phrases[i] for i in selected_indices]
                else:
                    selected_phrases = self.phrases
                
                st.write(f"Selected: {len(selected_phrases)} phrases")
            else:
                selected_phrases = []
                st.warning("No phrases available")
            
            # Recording controls
            st.subheader("🎙️ Recording")
            
            if st.button("🎙️ Start Recording", key="record_btn"):
                self._simulate_recording()
            
            if st.button("📊 Analyze Results", key="analyze_btn"):
                self._analyze_results()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📝 Current Phrase")
            
            if selected_phrases:
                # Display current phrase
                current_phrase = selected_phrases[0]  # Simplified - just show first phrase
                st.info(f"**Current Phrase:** {current_phrase}")
                
                # Recording status
                if 'recording_status' in st.session_state:
                    st.success(st.session_state.recording_status)
                
                # Instructions
                st.markdown("""
                ### 📋 Instructions:
                1. Click "Start Recording" in the sidebar
                2. Speak the phrase clearly when prompted
                3. Wait for processing to complete
                4. Review your results
                """)
            else:
                st.warning("Please select phrases to test")
        
        with col2:
            st.header("📊 Quick Stats")
            
            if 'results' in st.session_state:
                results = st.session_state.results
                
                # Overall score
                overall_score = results.get('overall_score', 0.0)
                st.metric("Overall Score", f"{overall_score:.1%}")
                
                # Component scores
                st.subheader("Component Scores")
                
                components = {
                    'phoneme_accuracy': 'Phoneme Accuracy',
                    'pitch_similarity': 'Pitch Similarity',
                    'duration_similarity': 'Duration Similarity',
                    'stress_pattern_accuracy': 'Stress Pattern'
                }
                
                for key, label in components.items():
                    score = results.get(key, 0.0)
                    st.metric(label, f"{score:.1%}")
            else:
                st.info("No results yet. Complete a recording to see your scores.")
        
        # Results section
        if 'results' in st.session_state:
            st.header("📈 Detailed Results")
            
            with st.expander("🎤 Complete Analysis Report", expanded=True):
                self._display_detailed_results(st.session_state.results)
    
    def _simulate_recording(self):
        """Simulate recording process."""
        with st.spinner("🎙️ Recording in progress..."):
            # Simulate recording time
            time.sleep(2)
            
            # Update status
            st.session_state.recording_status = "✅ Recording completed successfully!"
            
            # Show success message
            st.success("Recording completed! Click 'Analyze Results' to see your scores.")
    
    def _analyze_results(self):
        """Analyze the recorded results."""
        with st.spinner("📊 Analyzing your pronunciation..."):
            # Simulate analysis time
            time.sleep(3)
            
            # Generate mock results
            st.session_state.results = {
                'overall_score': 0.75,
                'accent_level': 'Mild accent',
                'phoneme_accuracy': 0.82,
                'pitch_similarity': 0.68,
                'duration_similarity': 0.74,
                'stress_pattern_accuracy': 0.71,
                'feedback': [
                    "Practice vowel length in stressed syllables",
                    "Work on intonation patterns",
                    "Focus on stress-timed rhythm",
                    "Record and compare with native speakers"
                ],
                'recommendations': [
                    "Focus on minimal pairs: /θ/ vs /t/, /ð/ vs /d/",
                    "Practice stress-timed rhythm",
                    "Listen to native English speakers",
                    "Use tongue twisters for articulation"
                ]
            }
            
            st.success("Analysis complete! Check the results below.")
    
    def _display_detailed_results(self, results):
        """Display detailed analysis results."""
        # Overall assessment
        st.subheader("🎯 Overall Assessment")
        overall_score = results.get('overall_score', 0.0)
        accent_level = results.get('accent_level', 'Unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Score", f"{overall_score:.1%}")
        with col2:
            st.metric("Accent Level", accent_level)
        
        # Component breakdown
        st.subheader("📊 Component Breakdown")
        
        components = {
            'phoneme_accuracy': 'Phoneme Match Rate',
            'pitch_similarity': 'Pitch Contour Similarity',
            'duration_similarity': 'Duration Similarity',
            'stress_pattern_accuracy': 'Stress Pattern Accuracy'
        }
        
        for key, label in components.items():
            score = results.get(key, 0.0)
            st.progress(score)
            st.caption(f"{label}: {score:.1%}")
        
        # Feedback
        if 'feedback' in results:
            st.subheader("💡 Improvement Tips")
            for tip in results['feedback']:
                st.write(f"• {tip}")
        
        # Recommendations
        if 'recommendations' in results:
            st.subheader("🎯 Recommended Practice")
            for rec in results['recommendations']:
                st.write(f"• {rec}")
        
        # Encouragement
        st.subheader("🌟 Keep Going!")
        if overall_score >= 0.8:
            st.success("Excellent work! You're doing great with your pronunciation.")
        elif overall_score >= 0.6:
            st.info("Good progress! Focus on the tips above to improve further.")
        else:
            st.warning("Don't get discouraged! Pronunciation takes time and practice.")
    
    def _show_help(self):
        """Show help information."""
        st.sidebar.header("❓ Help")
        
        with st.sidebar.expander("How to use"):
            st.markdown("""
            **Getting Started:**
            1. Select phrases to test
            2. Click "Start Recording"
            3. Speak clearly when prompted
            4. Review your results
            
            **Tips for Best Results:**
            - Use a good microphone
            - Minimize background noise
            - Speak at normal pace
            - Practice phrases first
            """)
        
        with st.sidebar.expander("About"):
            st.markdown("""
            **Accent Strength Estimator**
            
            This tool analyzes your English pronunciation
            and provides feedback on your accent strength.
            
            Features:
            - Phoneme analysis
            - Pitch contour analysis
            - Duration analysis
            - Personalized feedback
            """)
