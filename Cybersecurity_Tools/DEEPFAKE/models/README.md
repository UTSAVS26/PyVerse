# ⚡ DeepTrace Backend

FastAPI server powering **DeepFake detection models** for images, audio, and videos.  
This backend integrates with the **DeepTrace Chrome Extension** and Hugging Face models.

---

## 📌 API Endpoints

### 🔹 `POST /detect-image`
Analyze an image for deepfake manipulation.  
**Request Body:**
```json
{
  "image": "base64_encoded_image_data"
}
Response:

{
  "result": "Real|Fake",
  "confidence": 0.95,
  "error": null
}

🔹 POST /detect-audio

Analyze an audio file for deepfake manipulation.
Request Body:

{
  "audio": "base64_encoded_audio_data"
}


Response:

{
  "result": "Real|Fake",
  "confidence": 0.87,
  "error": null
}

🔹 POST /detect-video

Analyze a video for deepfake manipulation.
Request Body:

{
  "video": "base64_encoded_video_data"
}


Response:

{
  "result": "Real|Fake",
  "confidence": 0.92,
  "frames_analyzed": 25,
  "error": null
}

🔹 GET /health

Check server status and model availability.
Response:

{
  "status": "healthy",
  "models_loaded": true,
  "audio_model": true,
  "image_model": true,
  "video_model": true
}

🛠️ Development Setup
1. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

2. Install Dependencies
pip install -r requirements.txt

3. (Optional) Install Dev Tools
pip install -r requirements-dev.txt

4. Run the Server
python main.py


Server runs at → http://localhost:8000

📖 API Documentation

FastAPI provides interactive docs:

Swagger UI → http://localhost:8000/docs

ReDoc → http://localhost:8000/redoc

🧠 Model Management

Models are downloaded from Hugging Face Hub on first run.

Cached locally for later use.

Force re-download models:

Delete Hugging Face cache directory, or

Set HF_HOME environment variable.