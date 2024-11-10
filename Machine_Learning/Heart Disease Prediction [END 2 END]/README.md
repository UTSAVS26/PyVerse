# Heart Disease Detection using Machine Learning
## Project Overview
This project aims to predict the presence of heart disease in a patient using machine learning classifiers. We explore and preprocess the dataset, then implement and evaluate three different models:

K-Nearest Neighbors (KNN)
Decision Tree Classifier
Random Forest Classifier
Table of Contents
Dataset
Exploratory Data Analysis (EDA)
Data Preprocessing
Feature Engineering
Modeling
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Random Forest Classifier
Evaluation

## Dataset
The dataset used is the Heart Disease UCI dataset, which contains 303 observations and 14 features. The target variable (target) indicates the presence of heart disease, with 1 meaning "heart disease" and 0 meaning "no heart disease."

Data Variables:
Age: Age of the patient
Sex: Gender (1 = Male, 0 = Female)
Cp: Chest pain type (0-3)
Trestbps: Resting blood pressure (in mm Hg)
Chol: Serum cholesterol in mg/dl
Fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
Restecg: Resting electrocardiographic results (0-2)
Thalach: Maximum heart rate achieved
Exang: Exercise-induced angina (1 = yes, 0 = no)
Oldpeak: ST depression induced by exercise
Slope: Slope of the peak exercise ST segment (0-2)
Ca: Number of major vessels colored by fluoroscopy (0-4)
Thal: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
Target: Heart disease presence (1 = yes, 0 = no)
## Exploratory Data Analysis (EDA)
The EDA phase helps us understand the distribution and relationship of various features. Key steps include:

Visualizing the distribution of numerical variables (e.g., age, chol, thalach).
Exploring categorical variables (e.g., sex, cp, thal) through bar plots.
Investigating correlations between variables using heatmaps.
Handling missing data in the ca and thal columns.
## Data Preprocessing
Data preprocessing includes:

Handling Missing Values: We handle missing values in key columns such as ca and thal using either median imputation or deleting rows with missing data.
Feature Scaling: We scale continuous variables like age, chol, trestbps, thalach, and oldpeak using StandardScaler.
Encoding Categorical Variables: Convert categorical variables (sex, cp, restecg, slope, thal) into numerical formats using one-hot encoding or label encoding.
## Feature Engineering
In this phase, we create new features or modify existing ones to improve model performance. Examples include:

Combining cp (chest pain) and thalach (max heart rate) to create a new feature that captures heart rate response under stress.
Binning continuous features like age into age groups.
## Modeling
We implemented three machine learning classifiers:

1. K-Nearest Neighbors (KNN)
Algorithm: KNN is a simple, instance-based learning algorithm where classification is based on the majority vote from the k-nearest neighbors.
Parameters Tuned: Number of neighbors (n_neighbors), distance metric (p for Manhattan or Euclidean distance).
2. Decision Tree Classifier
Algorithm: Decision trees partition the data recursively based on feature splits that maximize information gain or minimize impurity (e.g., Gini impurity).
Parameters Tuned: Maximum depth of the tree (max_depth), minimum samples to split a node (min_samples_split).
3. Random Forest Classifier
Algorithm: Random forests are an ensemble of decision trees trained on random subsets of data and features, which enhances robustness and accuracy.
Parameters Tuned: Number of trees (n_estimators), maximum depth (max_depth), and features considered for splits (max_features).
## Evaluation
We evaluate model performance using:

Accuracy Score: Percentage of correct predictions.
Precision, Recall, F1-Score: Used for understanding model performance in detecting true positives and minimizing false positives.
Confusion Matrix: Visual representation of true positives, true negatives, false positives, and false negatives.
Cross-Validation: 5-fold cross-validation to check model generalization.