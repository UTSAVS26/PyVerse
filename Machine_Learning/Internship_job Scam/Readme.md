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

---

## 🔍 Workflow

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
   - **Best model selected:** `LogisticRegression` (based on F1 score)

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

## 📈 Results

| Metric        | Value             |
|---------------|-------------------|
| Accuracy      | ~97%              |
| Precision     | 0.69 (fraud class) |
| Recall        | 0.59  (fraud class) |
| F1 Score      | 0.64  (fraud class) |

> **Note:** Due to class imbalance, precision is high but recall for fraud class is lower.

---

## 🧪 Sample Inference

```python
job_title = "Earn $200 daily from home!"
description = "No experience required. Just 2 hours a day. Immediate joining. Limited seats."
pipeline.predict([[job_title, description]]) 
# Output: [1] -> Scam
