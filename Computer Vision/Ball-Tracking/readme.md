
# Ball Tracking with OpenCV

This project implements ball tracking using OpenCV, a powerful computer vision library. The primary objective is to detect and track a moving ball in real-time video streams. The system utilizes color detection, contour analysis, and other techniques to achieve accurate tracking.


## Introduction

The ball tracking system uses OpenCV to analyze video frames, identifying the position of a ball and tracking its movement. This project is particularly useful for applications in sports analysis, robotics, and interactive installations.

## Features

- Real-time ball detection and tracking
- Adjustable color detection settings
- Display of ball trajectory
- Support for video file input or webcam feed

## How It Works

### Color Detection

1. **HSV Color Space**: The project converts the video frames from the BGR color space (default in OpenCV) to the HSV (Hue, Saturation, Value) color space. HSV is often better for color detection because it separates color information (hue) from intensity (value).

2. **Color Range**: The user can specify the HSV range for the ball color they want to track. 

3. **Mask Creation**: A mask is created using the defined color range. This mask isolates the pixels within the specified color range.

### Contour Detection

1. **Find Contours**: After creating the mask, the contours of the detected objects (potential balls) are found using `cv2.findContours`. This function retrieves contours from the binary mask.


2. **Filter Contours**: The contours are filtered based on area or other criteria to identify the contour that corresponds to the ball. Typically, the largest contour is assumed to be the ball.

### Tracking the Ball

1. **Calculate Centroid**: Once a valid contour is detected, the centroid (center) of the ball is calculated using moments.

2. **Draw Trajectory**: The trajectory of the ball can be visualized by storing the centroid coordinates and drawing lines on the frame.

3. **Display Results**: The original frame with the detected ball and its trajectory is displayed in a window.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- OpenCV library
- NumPy


