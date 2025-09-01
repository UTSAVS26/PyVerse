# 🧠 DeepTrace - Real-time DeepFake Detection (Chrome Extension + AI Backend)

<div align="center">

![DeepTrace Logo](extension/icons/icon128.png)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gattigoppula--Sindhu-blue?logo=linkedin)](https://www.linkedin.com/in/gattigoppula-sindhu)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/sindhugattigoppula/deepfake-detector-extension)
[![Hugging Face](https://img.shields.io/badge/Models-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/SindhuGattigoppula)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

**Detect deepfakes in images, audio, and videos — directly in your browser.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Development](#-development) • [Model-Info](#-model-information) • [Contributing](#-contributing)

</div>

---

## 📌 GOAL

With the rise of manipulated media, **DeepFakes** have become a major threat — spreading misinformation through fake celebrity videos, morphed audios, and altered images.  
This project provides a **browser-based deepfake detector** powered by AI, integrated seamlessly via a Chrome Extension and FastAPI backend.

---

## 🚀 Features

- 🌐 **Chrome Extension**  
  - Detect fake media directly while browsing any webpage  
  - Right-click → "Detect DeepFake"  
- 🤖 **AI-Powered Backend (FastAPI)**  
  - Handles image, video, and audio classification  
  - Integrates with HuggingFace models  
- 📸 **Media Support**  
  - ✅ Image DeepFake Detection  
  - ✅ Video DeepFake Detection  
  - ✅ Audio DeepFake Detection  
- 📊 **Performance**  
  - Images: >90% accuracy (~500ms inference)  
  - Audio: ~85% accuracy (~2s inference)  
  - Video: ~88% accuracy (~10s, 30 frames)  

---

## 🧪 Tech Stack

| Area        | Technology                        |
|-------------|-----------------------------------|
| Backend     | Python, FastAPI                   |
| Models      | TensorFlow, Keras, EfficientNet, OpenCV |
| Frontend    | HTML, CSS, JavaScript             |
| Extension   | Chrome Extension API              |
| Deployment  | HuggingFace, GitHub, Docker       |

---

## 📁 Project Structure

````text
deeptrace-extension/
├── backend/         # FastAPI backend and ML models
│   ├── main.py      # API entrypoint
│   ├── requirements.txt
├── extension/       # Chrome extension files
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── styles.css
├── models/          # Model configs / HuggingFace integration
├── tests/           # Unit tests
├── docs/            # Documentation
└── [README.md](http://_vscodecontentref_/1)
`````

🧠 How It Works

User clicks Detect in the extension or right-clicks on media.
Media is sent to the FastAPI backend.
Backend identifies type → image, video, or audio.
Model classifies as Real or Fake with confidence.
Result is shown inside Chrome extension popup.

⚙️ Installation
# Clone repo
git clone https://github.com/UTSAVS26/PyVerse.git
cd PyVerse/Cybersecurity_Tools/DEEPFAKE/extension/backend

# Setup virtual environment
python -m venv venv
# For Linux/Mac
source venv/bin/activate
# For Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

API runs at → http://localhost:8000

🌐 Load Chrome Extension
Open Chrome → chrome://extensions/
Enable Developer Mode
Click Load Unpacked
Select PyVerse/Cybersecurity_Tools/DEEPFAKE/extension folder
Click the extension icon → Start detecting!
🖱️ Popup Interface
Click the extension icon → upload or detect page media.
🖱️ Context Menu
Right-click on any image/audio/video → Detect DeepFake.

🛠️ API Endpoints
POST /detect-image → analyze image
POST /detect-audio → analyze audio
POST /detect-video → analyze video
GET /health → server health
📊 Model Information
Images → Fine-tuned EfficientNetV2 (>90% accuracy)
Audio → Custom audio deepfake detection model (~85%)
Video → Phase-based video detection (~88%)
📦 Models are hosted on Hugging Face Hub and auto-downloaded on first run.

🛠️ Development
# Backend tests
cd backend
pytest tests/

# Extension tests
cd extension
npm install
npm test

🧵 Dataset

Images: Real And Fake Face dataset on kaggle.

Videos: DFDC review dataset.

Models are fine-tuned and hosted on Hugging Face.

🚀 Models Implemented

EfficientNetV2 → Image DeepFake Detection (>90% accuracy).


EfficientNetV2→ Video DeepFake Detection (~88% accuracy).

👉 Chosen because:

EfficientNet is lightweight yet powerful for visual tasks.


📈 Performance of Models

Image Model (EfficientNetV2): >90% accuracy

Video Model (EfficientNetV2): ~88% accuracy

<div align="center">
✨ Made with ❤️ by [Sindhu Gattigoppula]

</div> ````