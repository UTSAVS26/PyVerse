# Facial Emotion Detection with OpenCV and Keras
This project is designed to detect and classify emotions in real-time using a pre-trained deep learning model, Keras, and OpenCV. The model recognizes seven emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised. The script captures video feed, detects faces, and displays the predicted emotion on the screen.
## Prerequisites
Before running the code, ensure you have the following libraries installed:
**opencv-python,
numpy,
keras**

pip install opencv-python numpy keras

## Files in the Project
- **emotion_model.json:** The structure of the pre-trained emotion detection model.
- **emotion_model.h5:** The weights of the pre-trained model.
- **haarcascade_frontalface_default.xml:** The Haar Cascade file for face detection.

- Loading the Model
- Load the model architecture from emotion_model.json.
- Load the model weights from emotion_model.h5.

- json_file = open('emotion_model.json', 'r')
- loaded_model_json = json_file.read()
- json_file.close()
- emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")


Starting the Video Feed
To test the model on real-time video or a prerecorded video, initialize the video capture:

python
cap = cv2.VideoCapture(0)  # for webcam feed
# Or replace with the path to a video file:
cap = cv2.VideoCapture(r"C:\Users\Admin\emotiontrain\WIN_20240630_00_58_04_Pro.mp4")
Detecting and Classifying Emotions
In the while loop, the following steps are repeated frame by frame:

Frame Capture: Capture each frame and resize it for better processing speed.
Face Detection: Use Haar Cascade to locate faces.
Preprocessing: For each detected face, convert to grayscale and resize to (48x48), as required by the model.
Prediction: Pass the preprocessed face through the model, get the emotion prediction, and map the predicted index to the emotion label.
Display Results: Draw a rectangle around each detected face and label it with the predicted emotion.
python
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Ending the Video Feed
When the user presses 'q', the video feed stops, and all OpenCV windows close:

python
Copy code
cap.release()
cv2.destroyAllWindows()
