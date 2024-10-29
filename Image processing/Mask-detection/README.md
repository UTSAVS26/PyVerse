# Mask Detection Using MobileNetV2

## Project Overview
This project implements a real-time mask detection system using deep learning techniques. The model is based on the MobileNetV2 architecture, which is pre-trained on the ImageNet dataset. The system can classify individuals as wearing a mask or not wearing a mask using live video feeds from a webcam.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)

## Features
- Real-time mask detection from webcam feed.
- Utilizes MobileNetV2 for efficient image classification.
- Simple and user-friendly interface.
- Easy to modify and extend for other use cases.

## Requirements
- Python 3.6 or higher
- TensorFlow
- OpenCV
- NumPy
- Other standard libraries (included in the Python standard library)

You can install the required libraries using pip:

```bash
pip install tensorflow opencv-python numpy
```

## Dataset Structure
The dataset should contain two main folders: `annotations` and `images`. The structure should look like this:

```
dataset/
│
├── annotations/
│   ├── example1.xml
│   ├── example2.xml
│   └── ...
│
└── images/
    ├── example1.png
    ├── example2.png
    └── ...
```

### XML Annotation Format
Each XML file corresponds to an image and should contain bounding box information and labels. Example XML structure:

```xml
<annotation>
    <folder>images</folder>
    <filename>example1.png</filename>
    <size>
        <width>512</width>
        <height>366</height>
        <depth>3</depth>
    </size>
    <object>
        <name>with_mask</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
    <object>
        <name>without_mask</name>
        <bndbox>
            <xmin>250</xmin>
            <ymin>150</ymin>
            <xmax>350</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>
```

## Installation
1. Clone this repository to your local machine:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required libraries as mentioned above.

## Training the Model
To train the mask detection model, follow these steps:

1. Prepare your dataset or use the dataset given in this folder.
2. Use the provided training script (`Model-training.py`) to train the model:
   ```bash
   python Model-training.py
   ```

3. After training, the model will be saved as `mask_detector_mobilenetv2.h5`.

## Using the Model
To use the trained model for real-time mask detection:

1. Run the mask detection script (`MaskDetect.py`):
   ```bash
   python MaskDetect.py
   ```

2. The webcam feed will open, and the model will classify whether individuals are wearing a mask or not in real-time. Press 'q' to exit.

