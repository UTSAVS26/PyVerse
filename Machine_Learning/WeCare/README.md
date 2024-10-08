# Wecare--health-
Our Symptom-Based Disease Diagnosis Web App brings the power of machine learning and healthcare information to the fingertips of users. It serves as a valuable resource for individuals looking to gain insights into their health conditions quickly and conveniently.


### Model Description
The medicine prediction system utilizes a combination of machine learning models to diagnose various health conditions based on patient-reported symptoms. The system primarily employs the following models:

Logistic Regression: For binary classification of diseases based on symptom presence or absence.
Decision Trees: To provide interpretable predictions and visualize the decision-making process for diagnosing conditions.
Random Forest: To enhance prediction accuracy through ensemble learning, aggregating multiple decision trees.
Support Vector Machine (SVM): To classify complex patterns in symptoms and improve the robustness of predictions.
These models analyze a comprehensive dataset containing symptoms associated with multiple medical conditions, including common ailments like fungal infections and allergies. With an emphasis on accuracy and efficiency, the system is designed to assist healthcare professionals and patients in identifying potential health issues promptly, facilitating timely medical intervention.


### Database Description

This dataset contains medical symptom data for predicting different diseases based on various symptoms. It is structured to map a range of symptoms (as feature inputs) to specific disease outcomes (as labels). The dataset comprises 132 different symptoms and provides binary indicators for each, where `1` represents the presence of the symptom and `0` represents its absence. The final column in each row identifies the corresponding disease based on the symptoms.

### Key Features:
1. **Symptoms**: The dataset includes 132 symptoms such as:
   - **itching**
   - **skin_rash**
   - **nodal_skin_eruptions**
   - **continuous_sneezing**
   - **joint_pain**
   - **vomiting**
   - **weight_gain**
   - **restlessness**
   - **headache**
   - **high_fever**
   - **cough**
   - **abdominal_pain**
   - **blurred_and_distorted_vision**
   - **redness_of_eyes**
   - **chest_pain**
   - **neck_pain**
   - **dizziness**
   - **muscle_pain**
   - **cramps**
   - **loss_of_balance**
   - **visual_disturbances**
   - **irritability**
   - **coma**

   These symptoms capture a wide variety of medical conditions across different bodily systems.

2. **Disease Labels**: The output labels include several diseases, including but not limited to:
   - Fungal Infection
   - Allergy
   - Various other medical conditions

Each row in the dataset represents a set of symptoms for a patient, where `1` indicates that the symptom is present and `0` means absent. The dataset is labeled with the correct disease in the last column for supervised learning.

### Purpose:
The dataset is designed for use in machine learning models aimed at medical diagnosis. With this data, predictive models can be trained to accurately classify diseases based on patient symptoms, aiding healthcare professionals in decision-making.

### Applications:
- **Disease Prediction**: Based on patient-reported symptoms.
- **Symptom Analysis**: Understanding the correlations between symptoms and specific diseases.
- **Medical Training**: Useful in building diagnostic tools and systems that predict illnesses based on symptoms.

This dataset can be leveraged for classification models, including logistic regression, decision trees, random forests, or neural networks, aimed at improving healthcare decision systems.