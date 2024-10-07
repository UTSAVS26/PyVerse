# Real-Time Face Recognition Attendance System

## 🎯 Goal
The primary goal of this project is to develop an automated attendance system that leverages face detection and facial recognition technologies. The purpose is to simplify the process of marking attendance by automatically identifying individuals from images and recording their presence in a CSV file.

## 🧵 Dataset
The dataset comprises known face images stored in a local directory for face encoding purposes.

No external datasets were used. Instead, the images used for encoding are captured manually, representing individuals to be identified by the system.

## 🧾 Description
This project integrates YOLOv8 for face detection and the face_recognition library for facial recognition to create a real-time attendance system. Users can upload images to the system, which detects and recognizes faces, then records attendance based on the identified individuals. The attendance is stored in a CSV file with details of who is present or absent. This tool is suitable for environments such as schools, workplaces, and events where accurate and efficient attendance tracking is essential.

## 🧮 What I had done!
Loaded and initialized the YOLOv8 model to detect faces.
Encoded known face images using the face_recognition library.
Built a Streamlit-based web interface for user interaction.
Allowed users to upload images for processing.
Used YOLOv8 for detecting faces in the uploaded images.
Applied face_recognition to match detected faces with known faces.
Recorded attendance based on the recognition results in a CSV file.
Marked absent students in the CSV file if their faces were not detected.

## 🚀 Models Implemented
YOLOv8 (Ultralytics YOLOv8n-face): Chosen for its ability to quickly and accurately detect faces in images, even with multiple faces or varying image quality. YOLO is a highly efficient object detection model, suitable for real-time applications like this one.
Face Recognition Model (based on HOG and Deep Learning): Utilized for matching detected faces with known faces. The face_recognition library offers state-of-the-art facial recognition with easy integration and reliable performance.
Why these models?

YOLOv8: Provides fast and accurate face detection in a single pass through the image, making it ideal for real-time systems.
Face Recognition: The face_recognition library offers a pre-built, highly accurate solution that is simple to integrate and highly optimized for identifying individuals.

## 📚 Libraries Needed
`face_recognition`

`cv2` (OpenCV)

`numpy`

`csv`

`os`

`streamlit`

`datetime`

`ultralytics`

## 📊 Exploratory Data Analysis Results
EDA is not applicable as this project focuses on real-time face recognition from images rather than a traditional dataset.

## 📈 Performance of the Models based on the Accuracy Scores
YOLOv8: High accuracy for face detection in various lighting conditions and image quality.
Face Recognition: Excellent matching accuracy, with near 100% recognition accuracy for high-quality images and well-trained encodings.
Since the task involves face detection and recognition, traditional model accuracy metrics like F1 score or confusion matrices do not directly apply. However, performance has been validated by successfully identifying individuals in various test cases.

## 📢 Conclusion
This real-time face recognition attendance system successfully integrates YOLOv8 for face detection and face_recognition for identification, automating the attendance process with high accuracy. The combination of these models ensures reliable detection and recognition across different images. The system has demonstrated effective performance in identifying faces, recording attendance, and providing an intuitive user experience.

## ✒️ Your Signature
Name: Anirudh P S

LinkedIn: www.linkedin.com/in/anirudh248

GitHub: https://github.com/anirudh-248

## Steps to run the project:

1) Create a virtual environment.
2) Before installing the requirements, install dlib separately using `pip install "dlib_file_path"`
3) Install the requirements using `pip install -r requirements.txt`
4) Run the project using `streamlit run face_rec.py` for image input (or) `streamlit run smart_attendance.py` for video input.
