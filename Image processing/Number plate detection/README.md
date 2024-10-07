# Car-Number-Plates-Detection

## Hardware Requirements: 
 - Camera (to capture images/video feed) 
 - Computer or embedded system capable of running OpenCV and required libraries 
 - Sufficient memory and processing power for real-time image processing 

## Software Requirements: 
 - OpenCV: An open-source computer vision library for image and video processing. 
 - EasyOCR: A Python library for Optical Character Recognition (OCR). 
 - Matplotlib: A plotting library for Python (optional, for visualization purposes). 
 - Google Colab (if using cloud-based computing resources). .

## Resources: 
 - Haarcascade XML file (haarcascade_russian_plate_number.xml): This file contains 
   the trained data for detecting Russian number plates. It is used in the project for plate 
   detection. 
 - Sample image dataset: Contains images of vehicle license plates, used for testing and 
   training the detection algorithm. 

## How to run
 - Just run the numer_plate.py file using python number_plate.py.
 - The program will capture images from the camera and display the detected number plates.
 - The detected number plates will be saved in the 'detected_plates' folder.
 - The program will also display the recognized text from the detected number plates.
 - give path of saved image to jupyter notebok file.
