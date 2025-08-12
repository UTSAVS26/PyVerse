# Speech Emotion Recognition

A web-based application for detecting emotions in speech using machine learning.

## 🎯 Features

- **Real-time emotion detection** from speech audio
- **File upload support** for WAV audio files
- **Live recording** capability through web browser
- **Visual feedback** with confidence scores and waveform display
- **4 emotion classes**: Calm, Happy, Fearful, Disgust

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Speech_Emotion_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model file exists:
- Place `Emotion_Voice_Detection_Model.pkl` in the project directory
- Train the model using `training_p.ipynb` if not available

4. Run the application:
```bash
streamlit run app.py
```

## 📊 Model Information

- **Algorithm**: Multi-Layer Perceptron (MLP)
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Features**: 180 audio features (MFCC + Chroma + Mel Spectrogram)
- **Accuracy**: ~85% on test data

## 🎵 Audio Features

| Feature Type | Count | Description |
|--------------|-------|-------------|
| MFCC | 40 | Mel-frequency cepstral coefficients |
| Chroma | 12 | Pitch class profiles |
| Mel Spectrogram | 128 | Mel-scale frequency representation |

## 📁 Project Structure

```
Speech_Emotion_Recognition/
├── app.py                              # Main Streamlit application
├── training_p.ipynb                    # Model training notebook
├── requirements.txt                    # Python dependencies
├── Emotion_Voice_Detection_Model.pkl   # Trained model (generated)
└── README.md                          # This file
```

## 🎭 Supported Emotions

- 😌 **Calm** - Peaceful, relaxed speech
- 😊 **Happy** - Joyful, upbeat speech
- 😨 **Fearful** - Anxious, worried speech
- 🤢 **Disgust** - Repulsed, disgusted speech

## 🔧 Usage Tips

- Use WAV files for best compatibility
- Record for 3-5 seconds for optimal results
- Speak clearly and naturally
- Ensure good audio quality (minimal background noise)

## 📋 Requirements

See `requirements.txt` for full list. Key dependencies:
- streamlit>=1.28.0
- librosa>=0.10.0
- scikit-learn>=1.0.0
- numpy>=1.21.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- RAVDESS dataset creators
- Librosa library maintainers
- Streamlit community