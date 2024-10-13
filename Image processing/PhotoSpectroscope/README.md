# Spectroscopy Analysis

Analyze spectroscopy data captured from a live video feed. This program captures video frames and extracts spectral information.

## Table of Contents

- [Modules Needed](#modules-needed)
- [Main Code](#main-code)
- [Usage](#usage)
- [Functions](#functions)


## Modules Needed

```python

import math 
import os

import tkinter as tk
import tkinter.font as tkFont

from tkinter import *
from tkinter import ttk

import cv2

import matplotlib.image as img
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image, ImageTk
```

# Modules TO Be Installed

| Module                            | Installation Command                   |
| --------------------------------- | -------------------------------------- |
| `numpy`                            | `pip install numpy`                    |
| `opencv-python`                   | `pip install opencv-python`           |
| `matplotlib`                       | `pip install matplotlib`               |
| `Pillow`                           | `pip install Pillow`                   |
| `tkinter`                          | ( Included With Python )        |



## Main Code

```python
def capture():
    """
    Capture video frames from the default camera.

    Returns:
        list: A list of intensity values representing the captured frame.
    """
    global rlable

    cap = cv2.VideoCapture(0)

    roi_selected = False

    while True:
        ret, frame = cap.read()

        k = cv2.waitKey(1)

        if k & 0xFF == ord("s") and roi_selected == True:
            shape = cropped.shape
            r_dist = []
            b_dist = []
            g_dist = []
            i_dist = []
            for i in range(shape[1]):
                r_val = np.mean(cropped[:, i][:, 0])
                b_val = np.mean(cropped[:, i][:, 1])
                g_val = np.mean(cropped[:, i][:, 2])
                i_val = (r_val + b_val + g_val) / 3

                r_dist.append(r_val)
                g_dist.append(g_val)
                b_dist.append(b_val)
                i_dist.append(i_val)

        elif k & 0xFF == ord("r"):
            r = cv2.selectROI(frame)
            roi_selected = True

        elif k & 0xFF == ord("q"):
            break

        else:
            if roi_selected:
                cropped = frame[
                    int(r[1]): int(r[1] + r[3]), int(r[0]): int(r[0] + r[2])
                ]
                cv2.imshow("ROI", cropped)
            else:
                cv2.imshow("FRAME", frame)

    cap.release()
    cv2.destroyAllWindows()
    return i_dist


# Define a function to normalize a spectrum
def normalise(spectrumIn):
    """
    Normalize a spectrum by dividing all values by the maximum value.

    Args:
        spectrumIn (list): List of intensity values.

    Returns:
        list: Normalized spectrum.
    """
    spectrumOut = []

    maxPoint = max(spectrumIn)

    for value in spectrumIn:
        spectrumOut.append(value / maxPoint)

    return spectrumOut


# Define a function to calculate transmittance
def transmittance(reference, sample):
    """
    Calculate transmittance from a sample spectrum and a reference spectrum.

    Args:
        reference (list): Reference spectrum.
        sample (list): Sample spectrum.

    Returns:
        list: Transmittance values.
    """
    transmittance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            transmittance.append(0)
        else:
            transmittance.append(sample[i] / reference[i])

    return transmittance


# Define a function to calculate absorbance
def absorbance(reference, sample):
    """
    Calculate absorbance from a sample spectrum and a reference spectrum.

    Args:
        reference (list): Reference spectrum.
        sample (list): Sample spectrum.

    Returns:
        list: Absorbance values.
    """
    absorbance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            absorbance.append(0)
        else:
            absorbance.append(-math.log(sample[i] / reference[i], 10) / 5)

    return absorbance


# Define a function to calculate reflectance
def reflectance(reference, sample):
    """
    Calculate reflectance from a sample spectrum and a reference spectrum.

    Args:
        reference (list): Reference spectrum.
        sample (list): Sample spectrum.

    Returns:
        list: Reflectance values.
    """
    reflectance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            reflectance.append(0)
        else:
            reflectance.append(
                1
                - (sample[i] / reference[i])
                + (-math.log(sample[i] / reference[i], 10) / 5)
            )

    return reflectance

pixel = [115, 146, 193, 250, 312, 329, 404]
wavelength = [405.4, 436.6, 487.7, 546.5, 611.6, 631.1, 708]
reference = [  # Your reference data
    # ...
]

# Code for data processing using the defined functions
```

## Usage

1. Install the required modules (if not already installed) using `pip`:

   ```bash
   pip install numpy opencv-python matplotlib Pillow
   ```

2. Run the script to capture and process spectral data.

   ```bash
   python Spectroscope.py
   ```

3. Follow on-screen instructions to capture and analyze the data.

## Functions

- `capture()`: Captures and processes live video frames.
- `normalise(spectrumIn)`: Normalizes spectral data.
- `transmittance(reference, sample)`: Calculates transmittance.
- `absorbance(reference, sample)`: Calculates absorbance.
- `reflectance(reference, sample)`: Calculates reflectance.
