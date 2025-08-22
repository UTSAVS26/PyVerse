# 🎧 VoiceMoodMirror: Real-Time Voice Emotion Analyzer & Feedback

## 📌 Project Overview

**VoiceMoodMirror** captures a user's speech (a few sentences), analyzes **prosodic features** like pitch, tempo, energy, and spectral characteristics to infer their current mood or emotional state, then provides **visual feedback** (e.g., color/animation dashboard) and/or **suggests or plays music** that matches or modulates that mood.

## 💡 Use Cases

* Self-awareness / mood journaling tools
* Wellness apps (e.g., calming music when stressed)
* Interactive installations or smart mirrors
* Accessibility/emotion-aware assistants

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voicemoodmirror

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the Streamlit dashboard
streamlit run ui/dashboard.py

# Or run the demo notebook
jupyter notebook examples/demo_notebook.ipynb
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_audio_recorder.py
```

## 🧠 Core Components

### 1. Audio Capture & Feature Extraction
- **`audio/recorder.py`**: Microphone capture, buffering, preprocessing
- **`audio/feature_extractor.py`**: Extract pitch, tempo, energy, MFCCs, etc. (librosa)

### 2. Emotion Analysis
- **`emotion/prosody_classifier.py`**: Rule-based or ML model mapping prosodic features to mood
- **`emotion/mood_mapper.py`**: Maps inferred mood to visuals and music tags

### 3. Music Selection
- **`music/music_selector.py`**: Selects/queues music based on mood (local library or public API)
- **`music/playlist_builder.py`**: Builds adaptive playlists (e.g., calming if stressed)

### 4. User Interface
- **`ui/dashboard.py`**: Visual feedback (e.g., real-time mood meter, color gradients)

### 5. Utilities
- **`utils/smoothing.py`**: Temporal smoothing of noisy mood predictions

## 🔧 Features

* 🎤 Live or recorded voice input
* 📊 Real-time visualization of emotional state
* 🎶 Adaptive music recommendation/player
* 🔁 Temporal smoothing to avoid jittery mood flicker
* ⚙️ Configurable mood mappings (user can choose whether to reflect mood or counteract it)
* 🧠 Optional "mood history" log for self-reflection

## 🧪 Possible Enhancements

* Add **speech-to-text** and combine **semantic sentiment** with prosody for richer inference
* Personalized calibration per user (baseline pitch/tempo)
* Support for multilingual voice input
* Mobile/web version using Web Audio API and TensorFlow.js
* Emotion change detection and notifications (e.g., "You seem more stressed than 10 minutes ago")
* Integrate with smart home to adjust lighting/music based on mood

## 📁 Project Structure

```
voicemoodmirror/
│
├── audio/
│   ├── recorder.py              # Microphone capture, buffering, preprocessing
│   └── feature_extractor.py     # Extract pitch, tempo, energy, MFCCs, etc. (librosa)
│
├── emotion/
│   ├── prosody_classifier.py    # Rule-based or ML model mapping prosodic features to mood
│   ├── model_training.py        # (Optional) Train a lightweight model on synthetic / annotated data
│   └── mood_mapper.py           # Maps inferred mood to visuals and music tags
│
├── music/
│   ├── music_selector.py        # Selects/queues music based on mood (local library or public API)
│   └── playlist_builder.py      # Builds adaptive playlists (e.g., calming if stressed)
│
├── ui/
│   └── dashboard.py             # Visual feedback (e.g., real-time mood meter, color gradients)
│
├── utils/
│   └── smoothing.py             # Temporal smoothing of noisy mood predictions
│
├── tests/
│   ├── test_audio_recorder.py
│   ├── test_feature_extractor.py
│   ├── test_prosody_classifier.py
│   ├── test_mood_mapper.py
│   ├── test_music_selector.py
│   └── test_smoothing.py
│
├── examples/
│   └── demo_notebook.ipynb      # Interactive demo with recorded audio samples
│
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
