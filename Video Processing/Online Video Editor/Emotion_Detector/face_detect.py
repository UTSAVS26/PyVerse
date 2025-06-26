import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

#load haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#load pretrained emotion classification model
model = load_model('face_model.h5', compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

#fixed frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Real-time loop to capture frames and detect emotions
while True:
    ret, frame = cap.read()
    if not ret:
        break

     # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
       
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)


        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), (50, 50, 50), -1)

        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Added footer text with name and exit instruction
    footer_text = " Made with <3 by Sneha  |  Press 'Q' to Quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (170,255,195)
    thickness = 1

    #center the footer text
    text_size = cv2.getTextSize(footer_text, font, font_scale, thickness)[0]
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = frame.shape[0] - 20


    cv2.putText(frame,footer_text,(text_x,text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    #show the live video
    cv2.imshow('Real-Time Emotion Detection - Press q to Quit', frame)

    #to exit press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release camera and close opencv window
cap.release()
cv2.destroyAllWindows()
