
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

- `num_layers`: **4**  
- `units0`: **178**, `activation0`: **selu**, `dropout`: **0.1**  
- `units1`: **198**, `activation1`: **relu**  
- `units2`: **88**, `activation2`: **tanh**  
- `units3`: **18**, `activation3`: **relu**  
- `units4`: **198**, `activation4`: **relu**  
- `optimizer`: **rmsprop**  

## ⚙️ Training & Evaluation
- **Optimizer:** rmsprop  
- **Loss Function:** Mean Squared Error (MSE)  
- **Metric:** R² Score  

### 📊 Results
- **Training R² Score:** `0.9297`  
- **Testing R² Score:** `0.9231`  

These results indicate that the ANN model generalizes well and provides high prediction accuracy.

---

## 🚀 Installation & Usage
### 1️⃣ Clone Repository
```bash
git clone https://github.com/Fatimibee/PyVerse

cd Pyverse

# Install Dependencies
pip install -r requirements.txt