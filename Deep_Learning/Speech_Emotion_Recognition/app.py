import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile
from audio_recorder_streamlit import audio_recorder

# Must be the first Streamlit command in the script
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'Emotion_Voice_Detection_Model.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# FIXED: Use only librosa for audio processing
def extract_feature(file_path):
    try:
        # Use librosa.load for all audio loading - much more reliable
        X, sample_rate = librosa.load(file_path, sr=22050, duration=30.0)  # Force 22kHz, max 30 seconds
        
        # Check if audio is too short
        min_duration_sec = 2.0
        if len(X) < int(min_duration_sec * sample_rate):
            st.error(f"Audio file is too short. Please use audio with at least {min_duration_sec:.0f}–3 seconds duration.")
            return None
        
        # Ensure audio is valid (not all zeros/silence)
        if np.max(np.abs(X)) < 1e-6:
            st.error("Audio appears to be silent. Please upload an audio file with actual sound.")
            return None
        
        result = np.array([])
        
        # MFCC features (40 features)
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40, hop_length=512
        ).T, axis=0)
        result = np.hstack((result, mfccs))
        
        # Chroma features (12 features)
        chroma = np.mean(librosa.feature.chroma_stft(
            y=X, sr=sample_rate, hop_length=512
        ).T, axis=0)
        result = np.hstack((result, chroma))
        
        # Mel spectrogram features (128 features)
        mel = np.mean(librosa.feature.melspectrogram(
            y=X, sr=sample_rate, hop_length=512, n_mels=128
        ).T, axis=0)
        result = np.hstack((result, mel))
        
        # Ensure we have exactly 180 features (40 + 12 + 128)
        expected_length = 180
        if len(result) != expected_length:
            st.error(f"Feature extraction error: Got {len(result)} features, expected {expected_length}")
            return None
        
        return result.reshape(1, -1)
        
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        st.info("💡 Try using a WAV file or convert your audio to WAV format.")
        return None

# Emotion mapping
emotions_map = {
    'calm': '😌 Calm',
    'happy': '😊 Happy', 
    'fearful': '😨 Fearful',
    'disgust': '🤢 Disgust'
}

def main():
    
    st.title("🎤 Speech Emotion Recognition")
    st.markdown("*Detect emotions from speech using RAVDESS dataset trained MLP model*")
    
    # Load model
    model = load_model()
    if not model:
        st.error("❌ Model not found! Please ensure 'Emotion_Voice_Detection_Model.pkl' exists.")
        st.stop()
    
    # Show model info
    st.success(f"✅ Model loaded successfully! (Features: {model.n_features_in_}, Classes: {len(model.classes_)})")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])
    
    with tab1:
        st.header("📁 Upload Audio File")
        st.info("💡 **Tip**: Use WAV files for best compatibility. MP3 and other formats also supported.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
            help="Upload audio file (WAV recommended, max 30 seconds)"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # Size in MB
            st.info(f"📄 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Save uploaded file temporarily
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file)
            
            if st.button("🔍 Analyze Emotion", key="upload_analyze", type="primary"):
                with st.spinner("🧠 Analyzing emotion..."):
                    try:
                        # Extract features
                        features = extract_feature(temp_path)
                        
                        if features is not None:
                            # Validate feature dimensions
                            expected_features = model.n_features_in_
                            actual_features = features.shape[1]
                            
                            if actual_features != expected_features:
                                st.error(f"❌ Feature mismatch: Got {actual_features}, expected {expected_features}")
                                st.info("Please ensure your audio file is at least 3 seconds long and contains actual speech.")
                            else:
                                # Make prediction
                                prediction = model.predict(features)[0]
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 Analysis Results")
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    # Main result
                                    emotion_display = emotions_map.get(prediction, prediction)
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                               padding: 2rem; border-radius: 15px; text-align: center; 
                                               color: white; margin: 1rem 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                                        <h2>🎭 Detected Emotion</h2>
                                        <h1 style="font-size: 3rem; margin: 1rem 0;">
                                            {emotion_display}
                                        </h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Confidence scores
                                    if hasattr(model, 'predict_proba'):
                                        probabilities = model.predict_proba(features)[0]
                                        
                                        st.markdown("**🎯 Confidence Scores:**")
                                        
                                        # Sort by confidence
                                        emotion_probs = list(zip(model.classes_, probabilities))
                                        emotion_probs.sort(key=lambda x: x[1], reverse=True)
                                        
                                        for emotion, prob in emotion_probs:
                                            emotion_display = emotions_map.get(emotion, emotion)
                                            st.progress(prob, text=f"{emotion_display}: {prob:.1%}")
                                            
                                # Show feature info
                                st.success(f"✅ Successfully extracted {actual_features} audio features")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try using a different audio file or convert to WAV format.")
                    
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
    
    with tab2:
        st.header("🎙️ Record Audio")
        st.info("💡 **Tip**: Record for 3-5 seconds and speak clearly for best results.")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="🎤 Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔍 Analyze Emotion", key="record_analyze", type="primary"):
                with st.spinner("🧠 Analyzing emotion..."):
                    try:
                        # Save recorded audio temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_bytes)
                            temp_path = tmp_file.name
                        
                        # Extract features
                        features = extract_feature(temp_path)
                        
                        if features is not None:
                            # Validate feature dimensions
                            expected_features = model.n_features_in_
                            actual_features = features.shape[1]
                            
                            if actual_features != expected_features:
                                st.error(f"❌ Feature mismatch: Got {actual_features}, expected {expected_features}")
                                st.info("Try recording for a longer duration (3-5 seconds) and speak more clearly.")
                            else:
                                # Make prediction
                                prediction = model.predict(features)[0]
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 Analysis Results")
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    # Main result
                                    emotion_display = emotions_map.get(prediction, prediction)
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                               padding: 2rem; border-radius: 15px; text-align: center; 
                                               color: white; margin: 1rem 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                                        <h2>🎭 Detected Emotion</h2>
                                        <h1 style="font-size: 3rem; margin: 1rem 0;">
                                            {emotion_display}
                                        </h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Confidence scores
                                    if hasattr(model, 'predict_proba'):
                                        probabilities = model.predict_proba(features)[0]
                                        
                                        st.markdown("**🎯 Confidence Scores:**")
                                        
                                        # Sort by confidence
                                        emotion_probs = list(zip(model.classes_, probabilities))
                                        emotion_probs.sort(key=lambda x: x[1], reverse=True)
                                        
                                        for emotion, prob in emotion_probs:
                                            emotion_display = emotions_map.get(emotion, emotion)
                                            st.progress(prob, text=f"{emotion_display}: {prob:.1%}")
                    
                    except Exception as e:
                        st.error(f"❌ Recording analysis failed: {str(e)}")
                        st.info("💡 Try recording again with clearer speech.")
                    
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        This app uses a **Multi-Layer Perceptron (MLP)** model trained on the **RAVDESS dataset** to recognize emotions in speech.
        
        **🎭 Supported Emotions:**
        - 😌 **Calm** - Peaceful, relaxed speech
        - 😊 **Happy** - Joyful, upbeat speech  
        - 😨 **Fearful** - Anxious, worried speech
        - 🤢 **Disgust** - Repulsed, disgusted speech
        
        **🔧 Technical Details:**
        - **MFCC**: 40 features (speech characteristics)
        - **Chroma**: 12 features (pitch information)  
        - **Mel Spectrogram**: 128 features (frequency content)
        - **Total**: 180 audio features analyzed
        """)
        
        st.header("🎯 Usage Tips")
        st.markdown("""
        **For Best Results:**
        - 🎤 Speak clearly and naturally
        - 🔇 Record in a quiet environment
        - ⏱️ Use 3-5 second audio clips
        - 🎭 Express emotions distinctly
        - 📁 Prefer WAV format files
        - 📏 Keep files under 30 seconds
        """)
        
        st.header("🔧 Troubleshooting")
        st.markdown("""
        **Common Issues:**
        - **Silent audio**: Ensure microphone works
        - **Short files**: Record for at least 3 seconds
        - **Format errors**: Convert to WAV if possible
        - **Feature errors**: Check audio quality
        """)

if __name__ == "__main__":
    main()