# 📊 Customer Churn Prediction

This machine learning project aims to predict whether a telecom customer is likely to churn (i.e., stop using the service). Accurate churn prediction enables businesses to take proactive steps to retain customers and improve profitability.

---

## 📁 Dataset

We use the **Telco Customer Churn dataset**, which includes customer account details, services used, and billing information.

- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Target: `Churn` (Yes/No)

---

## 🧠 Features Used

Some key features in the dataset:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `InternetService`, `Contract`, `PaymentMethod`
- `MultipleLines`, `OnlineSecurity`, etc.

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Churn correlations
   - Customer behavior insights

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - Additional models (optional)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve

---

## 📈 Results

Model performance (example):
- Accuracy: **85%**
- Precision: **80%**
- Recall: **75%**
*(Replace with your actual model results)*

---

## 🛠️ How to Run This Project

1. **Clone the repository:**
```bash
git clone https://github.com/abinayagoudjandhyala/PyVerse.git

cd PyVerse/Machine_Learning/Customer_Churn_Prediction
````

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Launch the notebook:**

```bash
jupyter notebook CustomerChurnPrediction.ipynb
```

## 🙋‍♀️ Contributed Under

This project was contributed by **@abinayagoudjandhyala** as part of
**GirlScript Summer of Code 2025 (GSSoC'25)**.

---

