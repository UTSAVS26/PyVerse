# 🎤 Accent Strength Estimator

A Python application that analyzes speech recordings to estimate accent strength by comparing user pronunciation to native English reference models.

## 📌 Features

- **Real-time Speech Recording**: Capture audio using microphone
- **Phoneme Analysis**: Extract and compare phoneme sequences
- **Pitch Contour Analysis**: Analyze intonation patterns
- **Duration Analysis**: Compare speech rhythm and timing
- **Accent Scoring**: Generate 0-100% accent strength scores
- **Detailed Feedback**: Provide specific pronunciation tips
- **Multiple UI Options**: CLI, Tkinter GUI, and Streamlit web interface
- **Offline Operation**: No internet required, runs entirely locally

## 🛠 Tech Stack

- **Audio Processing**: `librosa`, `sounddevice`, `pydub`
- **Phoneme Analysis**: `phonemizer`, `pocketsphinx`, `parselmouth`
- **Signal Processing**: `numpy`, `scipy`
- **UI Frameworks**: `tkinter`, `streamlit`
- **Testing**: `pytest`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AccentStrengthEstimator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (if needed):
   - **Windows**: No additional dependencies required
   - **Linux**: `sudo apt-get install espeak-ng`
   - **macOS**: `brew install espeak`

## 🚀 Usage

### Command Line Interface
```bash
python main.py --mode cli
```

### Tkinter GUI
```bash
python main.py --mode gui
```

### Streamlit Web Interface
```bash
python main.py --mode web
```

## 📊 Example Output

```
🎤 Accent Strength Estimator Results
====================================

Overall Score: 72/100 (Moderate accent)

📈 Detailed Analysis:
- Phoneme Match Rate: 85%
- Pitch Contour Similarity: 68%
- Duration Similarity: 74%
- Stress Pattern Accuracy: 71%

💡 Improvement Tips:
- Improve vowel length in stressed syllables
- Practice 'th' as in 'think' — yours sounds like 't'
- Emphasize key syllables more clearly
- Work on intonation patterns in questions

🎯 Recommended Practice:
- Focus on minimal pairs: /θ/ vs /t/, /ð/ vs /d/
- Practice stress-timed rhythm
- Record and compare with native speakers
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

Run specific test categories:
```bash
pytest tests/test_audio_processing.py -v
pytest tests/test_phoneme_analysis.py -v
pytest tests/test_accent_scoring.py -v
```

## 📁 Project Structure

```
AccentStrengthEstimator/
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py
│   │   ├── processor.py
│   │   └── reference_generator.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── phoneme_analyzer.py
│   │   ├── pitch_analyzer.py
│   │   └── duration_analyzer.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── accent_scorer.py
│   │   └── feedback_generator.py
│   └── ui/
│       ├── __init__.py
│       ├── cli_interface.py
│       ├── gui_interface.py
│       └── web_interface.py
├── tests/
│   ├── __init__.py
│   ├── test_audio_processing.py
│   ├── test_phoneme_analysis.py
│   ├── test_accent_scoring.py
│   └── test_ui_components.py
├── data/
│   ├── reference_phrases.txt
│   └── sample_audio/
├── main.py
├── requirements.txt
└── README.md
```

## 🔧 Configuration

The application uses several configuration options:

- **Sample Rate**: 22050 Hz (configurable)
- **Recording Duration**: 5 seconds per phrase (configurable)
- **Reference Phrases**: 10 standard English sentences
- **Scoring Weights**: Configurable weights for different analysis components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CMU Pronouncing Dictionary for phoneme data
- Praat for speech analysis algorithms
- Librosa for audio processing capabilities
