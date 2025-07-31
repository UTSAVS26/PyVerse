# 🧠 Student Burnout Risk Predictor

A simple ML-powered Streamlit web app that predicts **burnout risk in students** based on daily lifestyle habits.  
Built for students to track wellness and detect burnout risk early.

---
## 📸 **Demo Screenshot**
![demo](/image/demo1.png)
![demo](/image/demo2.png)
![demo](/image/demo3.png)

## 💡 **Problem Statement**
Many students experience burnout due to academic pressure, irregular routines, and lack of self-care.  
This project aims to detect burnout risk early using ML based on daily lifestyle inputs.

---

## 🎯 **Features**
✅ Predicts burnout risk:  
- ✅ Healthy
- ⚠️ Mild Burnout Risk
- 🚨 High Burnout Risk

✅ Simple frontend (Streamlit)  
✅ Trained on balanced synthetic dataset  
✅ Easy to extend and improve

---

## 🛠 **Tech Stack**
- **Python 3**
- **scikit-learn**: ML model (Decision Tree)
- **Streamlit**: frontend UI
- **joblib**: model persistence
- **Pandas & NumPy**: data handling

---

## 📦 **Project Structure**
```text
student_burnout_predictor/
├── app.py # Streamlit frontend
├── predict.ipynb # Script to train and save ML model
├── burnout_dataset.csv # Balanced synthetic dataset (integer)
├── burnout_model_dt.joblib # Trained ML model
└── README.md # Project description
```

---

## ✅ **Setup & Run**

### ✅ 1. Clone this repo
```bash
git clone <repo-url>
cd student_burnout_predictor
```

### ✅ 2.Install Dependencies
```bash
pip install -r requirements.txt
```
### ✅ 3. Train the model
```bash
python train_model.py  
```
### ✅ 4. Run the app
```bash
streamlit run app.py

```
### 📊 **Dataset**
- Synthetic balanced dataset with:
- 200 Healthy
- 200 Mild Burnout Risk
- 200 High Burnout Risk
- Based on sleep, screen time, activity, mood, assignments, caffeine, and social hours
- Stored as: synthetic_burnout_data_int.csv

### 🌱 **How It Works**
-  Collect daily inputs
-  Predict burnout risk using a trained Decision Tree model
-  Show result and inputs

### 📈  **Future Scope**
- Collect real survey data
- Visual dashboards / trends
- Personalized self-care suggestions
- Integration with wearables (e.g., Fitbit)
- Deploy online (Streamlit Cloud)

### 🤝 **Contributing**
- PRs and feedback welcome!
- Ideas to help:
- Improve dataset realism
- Add charts or explainability
- Upgrade UI/UX



