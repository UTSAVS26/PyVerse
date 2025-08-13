# 📱 Student Social Media Addiction Analysis

This project investigates the relationship between students' social media usage and its impact on academic performance and addiction levels. It includes data preprocessing, encoding, feature scaling, and predictive modeling using classification and regression techniques.

---

## 📂 Dataset Overview

- **File**: `Students Social Media Addiction.csv` (present in the same directory)
  
- **Rows**: 705
- **Columns**: 13
- **Target Columns**:
  - `Affects_Academic_Performance` (Classification)
  - `Addicted_Score` (Regression)

---

## 🔍 Columns Description

| Column Name                    | Description |
|-------------------------------|-------------|
| `Student_ID`                  | Unique ID of each student |
| `Age`                         | Age of student |
| `Gender`                      | Gender (Male/Female) |
| `Academic_Level`              | High School / Undergraduate / Graduate |
| `Country`                     | Student’s country |
| `Avg_Daily_Usage_Hours`       | Daily average social media use in hours |
| `Most_Used_Platform`          | Instagram, YouTube, etc. |
| `Affects_Academic_Performance`| Whether social media affects academics (Yes/No) |
| `Sleep_Hours_Per_Night`       | Avg sleep duration |
| `Mental_Health_Score`         | Score representing mental health |
| `Relationship_Status`         | Single / In Relationship / Complicated |
| `Conflicts_Over_Social_Media`| Number of conflicts due to social media |
| `Addicted_Score`              | Numeric score representing social media addiction |

---

## 🔧 Preprocessing Steps

- **Null Check**: No missing values in the dataset.
- **Dropped**: `Student_ID` as it’s non-informative for modeling.

---

## 🧠 Feature Engineering

### 🎯 Encoding

| Encoding Type     | Columns |
|------------------|--------|
| One-Hot Encoding | `Country`, `Most_Used_Platform`, `Relationship_Status` |
| Label Encoding   | `Gender`, `Affects_Academic_Performance` |
| Ordinal Encoding | `Academic_Level` (`High School` < `Undergraduate` < `Graduate`) |

### 📏 Scaling

| Scaler Used     | Columns |
|-----------------|---------|
| `StandardScaler`| `Age`, `Mental_Health_Score`, `Conflicts_Over_Social_Media`, `Addicted_Score` |
| `MinMaxScaler`  | `Sleep_Hours_Per_Night` |
| `RobustScaler`  | `Avg_Daily_Usage_Hours` |

---

## 📊 Visualization

- Used `seaborn` to analyze distributions and choose appropriate scalers.
- Plotted Boxplots and KDEs to visualize skewness and outliers.

---

## 🧪 Modeling

**1. Classification: Predicting Academic Performance Impact (`Affects_Academic_Performance`)**

- Model: **Logistic Regression**
- Accuracy: **100%**
- Cross-Validated Accuracy: **~99.6%**
- **Confusion Matrix**:
  [[46, 0],
[ 0, 95]]
- No false positives or negatives – Excellent precision and recall.


---

**2. Regression: Predicting Addiction Score (`Addicted_Score`)**

- **Model**: Linear Regression
- **Evaluation**:
- Used RMSE and R² Score to evaluate.
- Good model fit based on predicted vs actual plots.
- **Train/Test Split**: 80/20

---

## ✅ Final Dataset

- **Final Feature Set**: 134 columns
- Combined:
- Encoded categorical data
- Scaled numeric features

---

## 👤 Author

- **GitHub**: [@archangel2006](https://github.com/archangel2006)


