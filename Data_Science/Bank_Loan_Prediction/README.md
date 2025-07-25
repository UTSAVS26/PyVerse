
# 💰 Bank Loan Status Prediction using Machine Learning

## 📌 Project Overview

Loan approval is a critical process in financial institutions, requiring accurate evaluation of customer eligibility. This project presents a **machine learning-based solution** that predicts whether a loan application should be **approved or rejected**, using applicant financial history and relevant features.

A **user-friendly web app** is developed using **Streamlit**, and includes both **batch processing** and **single predictions**, enabling institutions or individuals to assess loan approval eligibility efficiently.

---

## 📂 Dataset Information

The project uses two datasets:

- `credit_train.csv` – For training and validating the machine learning models.
- `credit_test.csv` – For testing or making predictions.

### Key Features Used:
- Credit Score  
- Annual Income  
- Current Loan Amount  
- Employment Status  
- Years in Current Job  
- Home Ownership  
- Monthly Debt  
- Number of Credit Problems  
- Bankruptcies  
- Tax Liens  
- and more...

---

## 🧠 Machine Learning Models Used

The following **supervised learning** algorithms were applied:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machines (SVM)  
- Gradient Boosting Models:
  - XGBoost  
  - LightGBM  

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  

---

## 🔧 Preprocessing Techniques

- Handling missing values  
- Feature scaling (e.g., StandardScaler)  
- Categorical encoding (e.g., OneHotEncoder / LabelEncoder)  

These steps ensure that models are trained on clean and consistent data, improving performance and generalization.

---

## 🚀 Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bank-loan-prediction.git
   cd bank-loan-prediction
   ```

2. **Install required Python libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm streamlit plotly
   ```

---

## 🖥️ Running the Project

### 📊 To run graphs and analysis:
```bash
python loan_prediction_system.py
```

### 🌐 To launch the web app (Streamlit):
```bash
streamlit run loan_web_app.py
```

The application supports:
- Batch processing (CSV upload)
- Single prediction based on manual input

---

## 📷 Output

- Visualizations such as feature distributions and performance graphs  
- Real-time prediction through the web interface  
- Summary statistics and model evaluation reports

---

## 📄 Documentation

For help or clarification:
- Refer to the project **documentation**
- Revisit the **abstract** at the top for a clear understanding of purpose and flow

---

## 📚 Abstract Summary

This project uses machine learning to predict bank loan approval status by analyzing applicant features. The aim is to automate the decision-making process and reduce manual errors. It provides a smart, fast, and scalable system for banks and lenders through both a backend pipeline and a clean web interface.

---

## 👨‍💻 Author

**Vedam Venkata Sarma**  
B.Tech Final Year – CSE (Data Science)  
Passionate about Data Science, Web Development, and AIML  
📬 _Looking for career opportunities in the same fields_  
