from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cv2
import tempfile
import os
from huggingface_hub import hf_hub_download
import logging
from typing import Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepTrace API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # add your chrome-extension://<id> here
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
audio_model = None
image_model = None
video_model = None
models_loaded = False

# Request models
class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")
    
class AudioRequest(BaseModel):
    audio: str = Field(..., description="Base64 encoded audio data")
    
class VideoRequest(BaseModel):
    video: str = Field(..., description="Base64 encoded video data")

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    audio_model: bool
    image_model: bool
    video_model: bool

@app.on_event("startup")
async def load_models():
    """Load models on startup with error handling"""
    global audio_model, image_model, video_model, models_loaded
    
    try:
        logger.info("Loading models from Hugging Face Hub...")
        
        # Download and load models with explicit error handling
        audio_model_path = hf_hub_download(
            repo_id="SindhuGattigoppula/Deepfake-models", 
            filename="audio_deepfake_model.keras"
        )
        audio_model = tf.keras.models.load_model(audio_model_path)
        logger.info("Audio model loaded successfully")
        
        image_model_path = hf_hub_download(
            repo_id="SindhuGattigoppula/Deepfake-models", 
            filename="efficient_final_model_finetuned_v3_90percent_cleaned.keras"
        )
        image_model = tf.keras.models.load_model(image_model_path)
        logger.info("Image model loaded successfully")
        
        video_model_path = hf_hub_download(
            repo_id="SindhuGattigoppula/Deepfake-models", 
            filename="video_phase2_final.keras"
        )
        video_model = tf.keras.models.load_model(video_model_path)
        logger.info("Video model loaded successfully")
        
        models_loaded = True
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "audio_model": audio_model is not None,
        "image_model": image_model is not None,
        "video_model": video_model is not None
    }

def decode_base64_media(data: str) -> bytes:
    """Decode base64 media data with validation"""
    if "," in data:
        data = data.split(",")[1]
    
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")

# Image detection endpoint
@app.post("/detect-image")
async def detect_image(request: ImageRequest):
    if not image_model:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
    try:
        # Decode and validate image
        image_bytes = decode_base64_media(request.image)
        
        # Process image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = image_model.predict(image, verbose=0)[0][0]
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        label = "Real" if prediction > 0.5 else "Fake"
        
        return {
            "result": label,
            "confidence": confidence,
            "error": None
        }
    except Exception as e:
        logger.error(f"Image detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# Audio feature extraction
def extract_audio_features(audio_bytes: bytes) -> np.ndarray:
    """Extract features from audio bytes"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(tmp_path, sr=22050)
        
        # Validate audio length
        duration = len(y) / sr
        if duration < 0.5:
            raise ValueError("Audio too short (minimum 0.5 seconds required)")
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # Combine features
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(spectral_bandwidth, axis=1),
            np.array([librosa.feature.zero_crossing_rate(y).mean()])
        ]).reshape(1, -1)
        
        return features
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

# Audio detection endpoint
@app.post("/detect-audio")
async def detect_audio(request: AudioRequest):
    if not audio_model:
        raise HTTPException(status_code=503, detail="Audio model not loaded")
    
    try:
        # Decode audio
        audio_bytes = decode_base64_media(request.audio)
        
        # Extract features
        features = extract_audio_features(audio_bytes)
        
        # Make prediction
        prediction = audio_model.predict(features, verbose=0)[0]
        
        # Apply threshold adjustment
        adjusted_threshold = 0.7
        confidence = float(max(prediction))
        label = "Fake" if prediction[1] >= adjusted_threshold else "Real"
        
        return {
            "result": label,
            "confidence": confidence,
            "error": None
        }
    except Exception as e:
        logger.error(f"Audio detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

# Video processing utilities
def extract_video_frames(video_path: str, max_frames: int = 30) -> list:
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            raise ValueError("Video too short (minimum 10 frames required)")
        
        # Calculate frame interval for sampling
        frame_interval = max(1, int(total_frames / max_frames))
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
                
            # Process frame
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
            
            if len(frames) >= max_frames:
                break
                
        return frames
    finally:
        cap.release()

# Video detection endpoint
@app.post("/detect-video")
async def detect_video(request: VideoRequest):
    if not video_model:
        raise HTTPException(status_code=503, detail="Video model not loaded")
    
    # Create temp file for video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        try:
            video_bytes = decode_base64_media(request.video)
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        except:
            temp_file.close()
            os.unlink(temp_file.name)
            raise
    
    try:
        # Extract and process frames
        frames = extract_video_frames(temp_path)
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames extracted")
        
        # Make predictions
        predictions = []
        for frame in frames:
            frame = np.expand_dims(frame, axis=0)
            pred = video_model.predict(frame, verbose=0)[0]
            predictions.append(pred)
        
        # Calculate average prediction
        avg_pred = np.mean(predictions, axis=0)
        confidence = float(max(avg_pred))
        label = "Fake" if avg_pred[0] > avg_pred[1] else "Real"
        
        return {
            "result": label,
            "confidence": confidence,
            "frames_analyzed": len(predictions),
            "error": None
        }
    except Exception as e:
        logger.error(f"Video detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)