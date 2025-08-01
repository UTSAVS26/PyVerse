# ❤️ Heart Disease Prediction using Machine Learning  

This project builds and compares multiple classification models to predict whether a patient is likely to develop heart disease based on medical attributes. The dataset is processed and evaluated using **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers. Evaluation metrics such as accuracy, precision, recall, and F1-score are used to assess performance.

---

## 📌 Objective  
To predict the presence or absence of heart disease using health-related features and compare classification models to identify the most effective one.

Comparing and evaluating three classification models:

- Logistic Regression  
- Decision Tree  
- Random Forest  

... and determining which performs best for heart disease prediction.

---

## 📂 Dataset  
**Source:** https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
**Target Variable:** `target` (1 = disease, 0 = no disease)

**Features:**

- age  
- sex  
- chest pain type (cp)  
- resting blood pressure (trestbps)  
- cholesterol (chol)  
- fasting blood sugar (fbs)  
- resting ECG results (restecg)  
- max heart rate (thalach)  
- exercise induced angina (exang)  
- ST depression (oldpeak)  
- slope of peak exercise ST segment (slope)  
- number of major vessels (ca)  
- thalassemia (thal)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked dataset shape, data types, null values (none found).  
- Reviewed class balance of the target variable.  
- Plotted histograms and bar plots for distributions and relationships.  
- Found correlations between features using a heatmap.  
- Detected outliers using boxplots.

---

## ⚙️ Preprocessing

- Used `StandardScaler` for numerical features.  
- One-hot encoded categorical variables (cp, restecg, slope, thal).  
- Split data using `train_test_split` (test size = 20%, stratified).  
- Models were trained on the same preprocessed dataset for fair comparison.

---

## 🧠 Model Performance

| Model              | Accuracy | Precision | Recall | F1-Score | Notes                          |
|--------------------|----------|-----------|--------|----------|--------------------------------|
| Logistic Regression| 91.0%    | 0.91      | 0.91   | 0.91     | Best performer in this project |
| Decision Tree      | 88.0%    | 0.88      | 0.88   | 0.88     | Easy to interpret              |
| Random Forest      | 88.0%    | 0.88      | 0.88   | 0.88     | Robust, but didn't outperform  |
| GridSearchCV-RF    | 86.6%    | 0.86      | 0.87   | 0.86     | Tuned RF performed worse       |
| Voting Classifier  | 86.0%    | 0.86      | 0.86   | 0.86     | No improvement seen            |

---

## 🔧 Hyperparameter Tuning (Random Forest)

Used `GridSearchCV` on Random Forest with:

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

**Best Params:**
{'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200}
However, test accuracy dropped to 86.6%, indicating over-regularization or underfitting.

---
## Interpretation & Insights

- Logistic Regression was the most reliable and highest-performing model.
- Random Forest's performance slightly decreased after tuning, suggesting the default hyperparameters were already near-optimal.
- Model generalization was more important than training accuracy.
- No evidence of overfitting; test accuracies were close to training

---
## 📌 Final Verdict
Although Random Forest is powerful and widely used, Logistic Regression performed best on this dataset with minimal tuning. It provided 91% accuracy and is interpretable, efficient, and generalizes well.

---
## 👤 Author
GitHub: archangel2006
