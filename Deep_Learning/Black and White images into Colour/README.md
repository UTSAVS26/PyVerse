# 🌌☄️Black and White Image Colorization

This project demonstrates how to colorize black-and-white images using a pre-trained deep learning model with OpenCV and Caffe. The model was developed by Richard Zhang et al., and it automatically adds color to grayscale images by predicting the *a* and *b* color channels in the LAB color space. This technique is useful for revitalizing old photos or generating color data from grayscale images.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Overview
This project aims to take black-and-white (grayscale) images and convert them to color using a deep learning model. The Caffe-based model predicts the chrominance values (color) while the lightness channel is preserved from the grayscale image. This technique uses machine learning to infer the most likely colors for various regions of an image.

The process involves:
- Loading a pre-trained model that was trained on thousands of color images.
- Transforming the grayscale image to LAB color space.
- Feeding the lightness channel (*L*) into the model to predict the *a* and *b* channels (chrominance).
- Combining these channels to produce a full-color image.

## Requirements
- **Python 3.x**
- **OpenCV (cv2)**
- **NumPy**
  
## Installation

1. Install dependencies:
   - Install NumPy and OpenCV:
     ```bash
     pip install numpy opencv-python
     ```

2. Download the necessary model files and save them in the same folder:
   - [colorization_deploy_v2.prototxt](https://github.com/richzhang/colorization/tree/caffe/colorization/models)
   - [pts_in_hull.npy](https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy)
   - [colorization_release_v2.caffemodel](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)

3. Giving credits to Richard Zhang for discovering the colorization technique, Check below for more details
   - https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
   - http://richzhang.github.io/colorization/
   - https://github.com/richzhang/colorization/

## Usage

1. Update the directories in the code according to your setup.

2. To run the script:
   - Open Command Prompt.
   - Navigate to the folder containing the code and model files using the `cd` command.
     ```bash
     cd C:\Users\your-directory\PycharmProjects\Colourisation
     ```
   - Run the script and provide the path to the black-and-white image as a parameter:
     ```bash
     python p1.py --image images/space2.jpg
     ```

## References
- [Richard Zhang Colorization Project](http://richzhang.github.io/colorization/)
- [OpenCV Colorization Sample](https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py)
