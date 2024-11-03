# Facial Emotion Detection with OpenCV and Keras

Real-time facial emotion detection using deep learning, OpenCV, and Keras. The system can detect and classify seven different emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Features

- Real-time emotion detection from webcam feed
- Support for pre-recorded video analysis
- Detection of multiple faces simultaneously
- Seven emotion classification categories
- Easy-to-use interface

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy keras
```

## Project Structure

```
facial-emotion-detection/
│
├── models/
│   ├── emotion_model.json     # Model architecture
│   └── emotion_model.h5       # Model weights
│
├── cascades/
│   └── haarcascade_frontalface_default.xml    # Face detection cascade
│
└── main.py                    # Main application file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-emotion-detection.git
cd facial-emotion-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Loading the Model

```python
# Load model architecture
json_file = open('models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create and load the model
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("models/emotion_model.h5")
print("Loaded model from disk")
```

### Starting Video Capture

```python
# For webcam feed
cap = cv2.VideoCapture(0)

# For video file
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

### Main Detection Loop

```python
while True:
    # Capture frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    
    if not ret:
        break
        
    # Detect faces
    face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    # Process each face
    for (x, y, w, h) in num_faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Extract and preprocess face region
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Display result
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
```

## Emotion Categories

The model can detect the following emotions:
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Performance Notes

- The detection speed depends on your hardware capabilities
- Multiple face detection might slow down the frame rate
- Optimal lighting conditions improve accuracy
- Recommended minimum specs:
  - CPU: Intel i5 or equivalent
  - RAM: 8GB
  - GPU: Optional but recommended for better performance

## Troubleshooting

Common issues and solutions:

1. **No camera feed**
   - Check if your webcam is properly connected
   - Try changing the camera index (e.g., `cv2.VideoCapture(1)`)

2. **Low detection accuracy**
   - Ensure good lighting conditions
   - Keep face centered and at a reasonable distance
   - Check if the face is clearly visible and not obscured

3. **Performance issues**
   - Reduce frame resolution
   - Process every nth frame
   - Close other resource-intensive applications
