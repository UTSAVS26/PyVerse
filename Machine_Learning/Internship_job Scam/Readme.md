# 💼 Internship/Job Scam Detection

This project focuses on building a machine learning model that detects whether a job or internship posting is genuine or fraudulent based on its **title** and **description**. The dataset used for training is sourced from Kaggle.

---

## 📂 Dataset

- **Source:** [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Features used:** 
  - `Title`
  - `Description`
- **Target:** 
  - `fraudulent` (0 = genuine, 1 = fraudulent)

---

## 🧪 Problem Statement

Given only the title and description of a job/internship posting, build a machine learning pipeline to classify whether the posting is real or a scam.

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas
- TfidfVectorizer
- SMOTE (imbalanced-learn)
- GridSearchCV
- Jupyter Notebook
---
## 🚀 Setup & Usage

### Installation
```bash
# Clone the repository
git clone <https://github.com/Fatimibee/PyVerse>
cd "Machine_Learning/Internship_job Scam"

# Install dependencies
pip install -r requirement.txt

# Download dataset from Kaggle
# Place fake_job_postings.csv in the project directory

### Trainng the model
# Run the Jupyter notebook
jupyter notebook job_internship_spam_GSSOC.ipynb


### Running the Web App
# Ensure the model file exists
streamlit run page.py

### 🔍 Workflow

1. **Preprocessing:**
   - Removed missing or null values in `Title` and `Description`
   - Applied TF-IDF vectorization separately to both text columns

2. **Handling Imbalanced Data:**
   - Used SMOTE to balance the classes since genuine jobs vastly outnumber scams

3. **Model Selection:**
   - Multiple classifiers were evaluated using `GridSearchCV` including:
     - `LogisticRegression`
     - `RandomForestClassifier`
     - `SVC`
     - `ComplementNB`
   - **Best model selected:** `LogisticRegression` (based on recall score)

4. **Pipeline:**
   - Combined all steps (TF-IDF, SMOTE, classifier) using `Pipeline` and `ColumnTransformer`
   - Hyperparameter tuning done via `GridSearchCV` on this pipeline

---

## 🧠 Final Model

- **Model:** `LogisticRegression`
- **Pipeline Components:**
  - `TfidfVectorizer` for both title and description
  - `SMOTE` for class imbalance
  - `LogisticRegression` as classifier

---

## 📊 Final Classification Report

After hyperparameter tuning and model selection, the best model yielded the following results:


                precision    recall  f1-score   support

           0       0.98      0.99      0.98      3023
           1       0.69      0.59      0.64       135

    accuracy                           0.97      3158
   macro avg       0.84      0.79      0.81      3158
weighted avg       0.97      0.97      0.97      3158

---

## 🧪 Sample Inference

```python
job_title = "Earn $200 daily from home!"
description = "No experience required. Just 2 hours a day. Immediate joining. Limited seats."
pipeline.predict([[job_title, description]]) 
# Output: [1] -> Scam

### 📬 Future Improvements

Improve recall for class 1 (fake internships)

Add more contextual features (company profile, website link)

Use transformer-based models for better text understanding