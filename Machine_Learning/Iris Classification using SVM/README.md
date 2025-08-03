# 🌸 Iris Flower Classification using SVM  

This project builds a multi-class classification model to identify the species of an Iris flower based on its physical characteristics. The model uses **Support Vector Machine (SVM)** with **GridSearchCV** for hyperparameter optimization. The final model achieves high accuracy and generalizes well on unseen data.

---

## 📌 Objective  
To classify Iris flowers into one of three species — *Setosa*, *Versicolor*, or *Virginica* — using measurements like petal length, sepal width, etc. The model is evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

---

## 📂 Dataset  
**Source:** `sklearn.datasets.load_iris()`  
**Target Variable:** `Species` (0 = Setosa, 1 = Versicolor, 2 = Virginica)

**Features:**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

---

## 🔍 Exploratory Data Analysis (EDA)
- Verified data shape, types, and missing values.
- Created pair plots colored by species.
- Plotted class distribution with custom x-tick labels.
- Heatmap used to visualize confusion matrix post-evaluation.

---

## ⚙️ Preprocessing  
- Target labels mapped to species names.
- Train-test split with stratification (80-20).
- Feature scaling using `StandardScaler`.
- Hyperparameter tuning via `GridSearchCV` with 5-fold CV.

---

## 🧠 Model Details

| Model | Accuracy | Notes |
|-------|----------|-------|
| SVM (GridSearchCV best) | **94.17%** | Best kernel & hyperparams selected automatically |

**Best Hyperparameters:**
- Kernel: `'rbf'`
- C: `10`
- Gamma: `0.01`

---

## 📈 Evaluation  

### 🔢 Classification Report

| Metric     | Class 0 (Setosa) | Class 1 (Versicolor) | Class 2 (Virginica) | Interpretation |
|------------|------------------|-----------------------|----------------------|----------------|
| Precision  | 1.00             | 0.92                  | 0.90                 | Very few false positives |
| Recall     | 0.97             | 0.90                  | 0.95                 | Some confusion between Class 1 & 2 |
| F1-Score   | 0.99             | 0.91                  | 0.93                 | Well-balanced performance |
| Support    | 40               | 40                    | 40                   | Balanced dataset |
| Accuracy   | —                | —                     | —                    | **94.17% overall** |
| Macro Avg  | —                | —                     | —                    | Precision: 0.94, Recall: 0.94, F1: 0.94 |
| Weighted Avg| —               | —                     | —                    | Balanced performance across classes |

---

### 🧮 Confusion Matrix

| Actual \ Pred | Pred 0 | Pred 1 | Pred 2 | Interpretation |
|---------------|--------|--------|--------|----------------|
| **Actual 0**  | 39     | 1      | 0      | 1 Setosa misclassified |
| **Actual 1**  | 0      | 36     | 4      | 4 Versicolor → Virginica |
| **Actual 2**  | 0      | 2      | 38     | 2 Virginica → Versicolor |

---

## 💡 Prediction Function

You can predict Iris species using:

```python
predict_flower([5.1, 3.5, 1.4, 0.2])  # Returns: 'setosa'
```

---

## 🔢 Final Model  
- **Model:** SVM (from `sklearn.svm.SVC`)
- **Scaler:** StandardScaler
- **Hyperparameter Tuning:** `GridSearchCV (cv=5)`

---

## 👤 Author  
**GitHub:** [@archangel2006](https://github.com/archangel2006)  
