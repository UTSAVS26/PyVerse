import cv2
import tensorflow as tf
import numpy as np

# Load a pre-trained MobileNetV2 mask detection model (change to your model path or URL if available)
model = tf.keras.models.load_model('mask_detector_mobilenetv2.h5')  # Make sure the model is in the same folder or provide path

# Function to preprocess the image for MobileNetV2
def preprocess_image(face):
    face_resized = cv2.resize(face, (224, 224))  # Resize to MobileNetV2 input size
    face_normalized = face_resized / 255.0       # Normalize pixel values
    face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
    return face_expanded

# Function to perform mask detection
def detect_mask(frame):
    # Convert the frame to a tensor and preprocess it
    face_preprocessed = preprocess_image(frame)
    prediction = model.predict(face_preprocessed)

    # MobileNetV2 model output for binary classification (Mask / No Mask)
    mask_probability = prediction[0][0]  # Get the single output probability for "with_mask"
    
    # Set the threshold for mask detection
    threshold = 0.5
    label = "Mask" if mask_probability > threshold else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    return label, color

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process each frame for mask detection
    label, color = detect_mask(frame)

    # Draw label on the frame
    cv2.putText(frame, f'{label}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
