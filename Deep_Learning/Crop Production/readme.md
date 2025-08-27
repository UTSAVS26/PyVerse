
# 🌾 Crop Production Prediction using ANN

This project predicts **crop production in tons** based on various agricultural and environmental factors using an Artificial Neural Network (ANN).

## 📌 Dataset Features

The dataset contains the following columns:

- **N** → Nitrogen content  
- **P** → Phosphorus content  
- **K** → Potassium content  
- **pH** → Soil pH value  
- **rainfall** → Rainfall in mm  
- **temperature** → Temperature in °C  
- **Area_in_hectares** → Cultivated area in hectares  
- **Production_in_tons** → **(Target variable)** Crop production in tons  
- **Yield_ton_per_hec** → Yield per hectare  

## 🎯 Target Variable
The target variable is:
- **Production_in_tons**

## 🏗 Model Architecture

The ANN is built with **Keras** and uses hyperparameter tuning.  
Best model parameters obtained:

**Input Layer**: 158 neurons (SELU)  
**Hidden Layer 1**: 18 neurons (Tanh)  
**Hidden Layer 2**: 28 neurons (ReLU)  
Dropouts applied for regularization  
**Output Layer**: 1 neuron (Regression Output)  
**Optimizer**: RMSprop (lr=0.001)  

 

## ⚙️ Training & Evaluation
- **Optimizer:** rmsprop  
- **Loss Function:** Mean Squared Error (MSE)  
- **Metric:** R² Score  

### 📊 Results
- **Training R² Score:** `0.90`  
- **Testing R² Score:** `0.89`  

These results indicate that the ANN model generalizes well and provides high prediction accuracy.

---

## 🚀 Installation & Usage
### 1️⃣ Clone Repository
```bash
git clone https://github.com/UTSAVS26/PyVerse

### Install dependencies for this project
cd "Deep_Learning/Crop Production"
pip install -r requirements.txt

###Run the web app
streamlit run page.py
```