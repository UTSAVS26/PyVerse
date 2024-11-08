# Medical Diagnosis and Prescription System using Deep Learning
## Overview
This project implements a Medical Diagnosis and Prescription System using Deep Learning techniques. By utilizing the power of Long Short-Term Memory (LSTM) networks, this system is capable of analyzing a patient's symptoms and providing an accurate diagnosis along with the recommended medication. The LSTM model, a variant of Recurrent Neural Networks (RNN), is particularly suited for processing sequential data like patient symptoms and offers significant advantages when dealing with textual data.

The model is trained on a dataset consisting of patient symptoms, diagnosed diseases, and prescribed medications, and is designed to deliver predictions efficiently and accurately.

## Key Features
Patient Symptom Analysis: The model takes a textual description of patient symptoms as input.
Disease Prediction: Based on the symptoms, the system predicts the most likely disease.
Medication Recommendation: The system also provides a list of medications suitable for treating the diagnosed disease.
Deep Learning Architecture: The model uses an LSTM architecture to capture long-term dependencies in the text data, which is essential for understanding symptom sequences.
Technology Stack
TensorFlow: Used to build and train the deep learning model.
LSTM (Long Short-Term Memory): A type of Recurrent Neural Network that can handle sequences of data and capture long-term dependencies.
Python: Programming language used for data preprocessing, model building, and implementation.
Pandas & Numpy: Libraries for data manipulation and numerical processing.
Keras: High-level API for TensorFlow to build and train neural networks.
## Dataset
The dataset used in this project consists of:

Patient Symptoms: Textual descriptions of symptoms experienced by patients.
Diagnosed Diseases: The confirmed disease or condition diagnosed for each patient.
Medications: The prescribed medication(s) for each patient based on their diagnosis.
Each data point includes a patient's symptoms, the disease they were diagnosed with, and the medications prescribed to treat it.

## Model Architecture
Input Layer: Accepts the textual input representing patient symptoms.
Embedding Layer: Converts the input text into a dense vector representation, capturing the semantic meaning of words.
LSTM Layer: Processes the sequential data to capture long-term dependencies between symptoms.
Dense Layers: Two separate dense layers are used:
One for predicting the disease based on the LSTM output.
Another for recommending the appropriate medication.
Output Layer: Provides both the predicted disease and the corresponding medication recommendation.
## Requirements
To run this project, you will need the following libraries installed:
```python

pip install tensorflow
pip install pandas
pip install numpy
pip install scikit-learn
```
How to Run the Project
Clone the repository:

```python

git clone <repository_url>
```
Install the required dependencies:

```python

pip install -r requirements.txt
```
Prepare the dataset: Ensure that the dataset (CSV file containing patient symptoms, diseases, and medications) is available in the correct format and is loaded into the script.

Train the model: Use the script to train the LSTM model on the dataset:

Test the model: Once trained, you can test the model by providing patient symptoms and getting the predicted disease and medication recommendation

## Future Improvements
Expand Dataset: Increasing the dataset size and variety will improve model accuracy and generalization.
Additional Features: Incorporate additional patient information like medical history, age, and lifestyle factors to refine the predictions.
Interactive Interface: Develop a user-friendly interface for easier use by healthcare professionals and patients.
## Conclusion
This project demonstrates the potential of Deep Learning in the medical field by providing a system that can quickly diagnose diseases and recommend medications based on a patient’s symptoms. By utilizing LSTM networks, we can handle the sequential nature of symptom descriptions and deliver precise results.

This approach could revolutionize the way medical diagnoses are made and reduce reliance on manual diagnoses and prescriptions.