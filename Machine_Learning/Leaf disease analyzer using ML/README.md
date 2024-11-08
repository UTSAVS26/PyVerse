# AI-Powered Plant Disease Detection

This project uses **Convolutional Neural Networks (CNNs)** to detect and classify plant diseases from leaf images. Leveraging **Python libraries** for data processing, **TensorFlow** for model training, and **Flask** for deployment, this solution aims to offer scalable and efficient disease detection to support agricultural productivity.

## Table of Contents
- [Overview](#overview)
- [Data Classification](#data-classification)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Steps Involved](#steps-involved)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview
Plant diseases can cause significant agricultural losses. This project targets early disease detection by focusing on leaf images, which often display the first visible signs of disease. The system is built using CNNs for high accuracy in image classification, aiming to provide farmers and researchers with a practical tool for real-time disease diagnosis.

## Data Classification
Images of various plant diseases are used for training the model. Each image is labeled based on the plant type and disease, creating a structured dataset for the CNN model to classify different diseases accurately. Key classification categories include:
- Plant species
- Healthy vs. diseased
- Specific diseases for each plant species

## Project Structure
```
Plant_Disease_Detection/
├── data/                   # Dataset of leaf images
├── notebooks/              # Jupyter notebooks for model training and evaluation
├── src/                    # Source code for preprocessing, training, and Flask app
│   ├── preprocess.py       # Data preprocessing script
│   ├── train.py            # Model training script
│   ├── app.py              # Flask web app for user interface
├── static/                 # Static files for web interface (e.g., CSS)
├── templates/              # HTML templates for web interface
└── README.md               # Project documentation
```

## Dependencies
- **Python 3.8+**
- **TensorFlow**: `pip install tensorflow`
- **PIL** (Pillow): `pip install pillow`
- **Seaborn**: `pip install seaborn`
- **Matplotlib**: `pip install matplotlib`
- **Flask**: `pip install flask`
- **Spicy**: `pip install spicy`

## Steps Involved
1. **Data Collection & Labeling**: Gather and label images of plant leaves with various diseases.
2. **Data Preprocessing**:
   - Resizing and normalizing images.
   - Applying data augmentation (rotation, flipping, etc.) to improve model generalization.
3. **Model Building**:
   - Build a CNN using TensorFlow.
   - Train the model on labeled images.
   - Optimize the model through hyperparameter tuning and validation.
4. **Deployment**:
   - Create a web interface using Flask to allow users to upload images and view the diagnosis results.
5. **Evaluation**:
   - Test the model on unseen data to ensure accurate disease detection.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranawk/Leaf_disease_analyzer.git
   cd Leaf_disease_analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python src/app.py
   ```
   The app will be available at `http://localhost:5000`. Upload a leaf image, and the system will display the predicted disease.

## Results
The model achieved high accuracy in classifying diseases across multiple plant species. Key metrics:
- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 92%

Example of predictions on test images:
- Healthy vs. diseased detection: 95% accuracy
- Disease-specific classification: 90% accuracy

## Future Work
- **Expand Dataset**: Include additional plant species and disease types.
- **Mobile Integration**: Develop a mobile-friendly interface or integrate with drones for real-time monitoring in fields.
- **Cloud Deployment**: Shift to cloud for handling larger datasets and real-time processing.

