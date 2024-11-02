# Drowsiness Detection System
The Drowsiness Detection System is a Computer Vision-based project aimed at improving road safety by detecting drowsiness in drivers. The system leverages machine learning and computer vision techniques to monitor eye closure and alert the user if signs of drowsiness are detected, potentially preventing accidents due to fatigue.

## Overview

Driving while drowsy is a leading cause of road accidents worldwide. This Drowsiness Detection System utilizes facial landmark detection and eye aspect ratio analysis to monitor drivers' eye activity. When it identifies prolonged eye closure, it triggers an alert to prevent accidents, aiming to enhance road safety and reduce the risk associated with driver fatigue.

## Features

- Real-time detection of drowsiness using a camera feed
- Alerts the driver through an audible alarm when drowsiness is detected
- Lightweight and efficient, suitable for deployment on various devices
- Configurable sensitivity settings for different levels of alertness

## Installation

To set up the Drowsiness Detection System on your local machine, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/shashmitha46/PyVerse.git
2. Navigate to the project directory:
```bash
cd PyVerse/main/Computer Vision/Drowsiness Detection System
```
3. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
## Description

The **Drowsiness Detection System** is designed to help prevent road accidents caused by drivers falling asleep at the wheel. By monitoring the driver’s eyes, it can detect signs of drowsiness and sound an alert to help keep the driver awake and safe.

- **Requirement of the Project**: Many accidents happen due to drowsy driving, especially on long or late-night drives. This project aims to provide a safety solution that warns drivers when they’re too tired.

- **Necessity**: It’s crucial to have a system that alerts drivers about drowsiness before an accident happens. This is especially important for drivers on long trips or driving in monotonous conditions.

- **Benefits and Use**: This system can be used in vehicles to enhance safety by monitoring drivers in real time. It’s beneficial for professional drivers and those who often drive long distances.

- **Approach**: We started by exploring how facial landmarks, especially around the eyes, can indicate drowsiness. Using computer vision tools like OpenCV and dlib, created a system that measures eye openness and triggers an alert when needed.

- **Additional Resources**: To develop the system, referred to tutorials on OpenCV and dlib, as well as articles and research papers on detecting drowsiness through eye analysis.

## EXPLANATION
The core main.py script operates as follows:

- Camera Feed Initialization: The script starts by accessing the connected camera.
- Facial Landmark Detection: Using a pre-trained model from dlib, it identifies facial landmarks, specifically focusing on the eye regions.
- Eye Aspect Ratio (EAR) Calculation: The script computes the EAR, a measure of eye openness, using specific eye landmarks.
- Drowsiness Detection: If the EAR remains below a predefined threshold for a specified duration, the system interprets this as drowsiness.
- Alert Mechanism: Upon detecting drowsiness, an audible alarm is triggered to alert the driver.
- Display and Exit: The system displays real-time video with annotations for monitoring and allows for termination with a key press.

## Libraries Needed
The following libraries are required for the project:

- cv2 (OpenCV) - for image processing functions
- numpy - for array operations
- dlib - for face landmark detection
- imutils.face_utils - for basic operations of facial landmark conversion
- playsound - for playing the alarm sound
  
## Usage
To run the Drowsiness Detection System, ensure you have a camera connected to your device and execute the following command:

```bash
python main.py
```
Once the program is running, it will continuously analyze the eye aspect ratio and emit an alert sound if it detects signs of drowsiness.

## Models and Techniques
The system employs the following:

- **Facial Landmark Detection:** Using dlib’s facial landmark predictor to locate key points around the eyes.
- **Eye Aspect Ratio (EAR):** Calculated based on the distance between specific eye landmarks. EAR helps in detecting prolonged eye closure.
- **Alert Mechanism:** An alert sound triggers when EAR values fall below a certain threshold for a continuous period, indicating drowsiness.

## Contributing
Contributions are welcome! If you'd like to improve the Drowsiness Detection System, please fork the repository and submit a pull request.
