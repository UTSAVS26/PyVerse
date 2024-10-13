#Import the necessary libraries
import urllib # Used for fetching data from a URL (replaced with webcam in original implementation)
import cv2 #OpenCV library for image processing and computer vision tasks
import numpy as np
from keras.models import load_model

#Loading Classifiers and Model
classifier = cv2.CascadeClassifier(r"C:\Users\Sibangi Boxipatro\Desktop\Face Detection\Face Detection\haarcascade_frontalface_default.xml") #Loads a pre-trained Haar cascade classifier for face detection (haarcascade_frontalface_default.xml)

model = load_model(r"C:\Users\Sibangi Boxipatro\Desktop\Face Detection\Face Detection\final_model.h5")

URL = 'http://192.168.43.1:8080/shot.jpg' #Defines the URL of the image source (likely a webcam stream in a different implementation)

#Prediction function (get_pred_label)
def get_pred_label(pred):
    labels = ['Monali', 'Ritika', 'Seetal', 'Sibangi']
    return labels[pred]

#Preprocessing function (preprocess)
def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img
    


ret = True
while ret: #Main loop
    
    img_url = urllib.request.urlopen(URL) #Fetching image from URL
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
      
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),2)
        
    cv2.imshow("capture",frame)
    if cv2.waitKey(1)==ord('q'):
        break
#Displaying and exiting
cv2.destroyAllWindows()
#This code assumes the pre-trained model expects images of a specific size (100x100).
#The URL could be replaced with a webcam capture function to use a real-time video stream.
#Ensure you have OpenCV, NumPy, Keras (with TensorFlow backend), and the pre-trained model and classifier files for this code to work
