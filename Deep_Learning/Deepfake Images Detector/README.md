# Deepfake Image Detector

## Overview
This project implements a deep learning-based detector for identifying deepfake images. It explores and compares three different neural network architectures: a custom Convolutional Neural Network (CNN), Capsule Networks (CapsNet), and transfer learning using the Xception model. The goal is to classify images as either 'Real' or 'Fake'.

## Dataset
- **Source:** [Kaggle - Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- **Classes:** 'Real' and 'Fake'
- **Structure:**
  - Train
  - Validation
  - Test

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/UTSAVS26/PyVerse.git
cd PyVerse/Deep_Learning/'Deepfake Images Detector'
```

### 2. Download the Dataset
- The dataset is downloaded using [kagglehub](https://github.com/Kaggle/kagglehub).
- Make sure you have a Kaggle account and API credentials set up.

In the notebook, run:
```python
!pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install tensorflow matplotlib kagglehub
```

### 4. Run the Notebook
Open `Deepfake_Image_Detector.ipynb` in Jupyter Notebook or any compatible environment and follow the cells sequentially.

## Project Structure
- `Deepfake_Image_Detector.ipynb` — Main notebook containing all code, EDA, model training, and evaluation.
- `README.md` — Project documentation.

## Models Implemented
1. **Custom CNN**
   - Built from scratch using Keras.
   - Consists of multiple Conv2D, BatchNorm, MaxPooling, Dropout, and Dense layers.
2. **Capsule Network (CapsNet)**
   - Custom implementation of Capsule layers for image classification.
   - Uses dynamic routing and squash activation.
3. **Xception (Transfer Learning)**
   - Utilizes pre-trained Xception model (ImageNet weights) as a feature extractor.
   - Custom dense layers added for binary classification.

## Training & Evaluation
- Due to dataset size, only a subset of data is used for demonstration.
- Early stopping is used to prevent overfitting.
- Training and validation accuracy/loss are visualized for each model.

### Results (on small subset)
| Model     | Test Accuracy |
|-----------|---------------|
| CNN       | ~44%          |
| CapsNet   | ~55%          |
| Xception  | ~66%          |

> **Note:** Results are based on a small subset and few epochs. For better performance, train on the full dataset and tune hyperparameters.

## Conclusion
- **Xception** (transfer learning) outperformed the other models on the subset and is recommended for best results.
- **Custom CNN** and **CapsNet** can be further improved with more data and tuning.

## Visualization
- The notebook includes code to visualize class distribution, sample images, pixel value distribution, and training curves.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- matplotlib
- kagglehub

## License
This project is for educational purposes. Please check the dataset's license for usage restrictions. 
