# 🍄 Mushroom Classification 

This project builds and compares multiple classification models to determine whether a mushroom is **edible** or **poisonous** based on its features. The dataset is processed and evaluated using **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Random Forest** classifiers. Evaluation metrics such as accuracy, precision, recall, and F1-score are used to compare model performance.

---

## 📌 Objective  
To classify mushrooms as **edible (e)** or **poisonous (p)** based on their categorical features. The goal is to build a robust model that generalizes well and can assist in safe mushroom identification.

Comparing and evaluating three classification models:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest

… and determine which performs best for mushroom classification.

---

## 📂 Dataset  
**Source:** [UCI Mushroom Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) 

**Target Variable:** `class` (e = edible, p = poisonous)

**Features :**
- cap-shape
- cap-surface
- cap-color
- bruises
- odor
- gill-attachment
- gill-spacing
- gill-size
- gill-color  
- stalk-shape
- stalk-root
- stalk-surface-above-ring
- stalk-surface-below-ring
- stalk-color-above-ring
- stalk-color-below-ring
- veil-type	veil-color
- ring-number
- ring-type
- spore-print-color
- population
- habitat

---

## 🔍 Exploratory Data Analysis (EDA)
- Checked shape, data types, null values (none found).
- Examined class distribution – fairly balanced.
- Analyzed frequency of each feature's unique values.
- Visualized correlation between selected categorical features and class.
- Inspected label distribution with bar plots.

---

## ⚙️ Preprocessing  
- Label encoding applied to all categorical variables.
- Train-test split using `train_test_split` (test size = 20%).
- Feature scaling using `StandardScaler` (for KNN & Logistic Regression).
- All models trained on the same preprocessed data for fair comparison.

---

## 🧠 Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | interpretation |
|--------------------|----------|-----------|--------|----------|-------|
| KNN          | **100%**   | 1.00      | 1.00   | 1.00     | Great accuracy |
| Logistic Regression | 94.77%   | 0.94      | 0.94   | 0.94     | Strong linear separator |
| Random Forest       | **100%** | 1.00      | 1.00   | 1.00     | Perfect accuracy, robust model |

---

## 🏆 Final Verdict

Both **KNN** and **Random Forest** achieved perfect classification on the test set. However:

- **Random Forest** may generalize better to unseen data due to ensemble learning.
- **KNN** is simpler but slower for large datasets at inference time.

📌 In practical applications, **Random Forest** is preferred due to efficiency and robustness.

---
## 👤 Author

GitHub: [archangel2006](https://github.com/archangel2006/)

