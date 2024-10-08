# Online Laptop Price Predictor 💻

Welcome to the **Online Laptop Price Predictor**! This web tool allows you to estimate laptop prices based on various specifications such as brand, RAM, storage, processor, and more. Whether you're a buyer, seller, or just curious about laptop prices, this tool provides accurate price predictions to assist you in making informed decisions.

### **Link to the Predictor**
[Try the Online Laptop Price Predictor here!](https://online-laptop-price-predictor-by-rahulakavector.streamlit.app/)

---

## Project Overview

This application uses a **Random Forest Regressor model** to predict laptop prices. The model was trained on a comprehensive dataset of laptops, including attributes like brand, RAM size, storage, weight, display type, and CPU brand. The prediction is based on these specifications, offering an estimated price for the laptop configuration you select.

### Key Features
- **User-Friendly Interface:** Easy to use with dropdown selections for each laptop specification.
- **Accurate Predictions:** Predictions are powered by a machine learning model trained on real-world data.
- **Specification-Based:** Customize specifications such as:
  - Brand (e.g., HP, Dell, Apple)
  - RAM (in GB)
  - Storage (SSD, HDD)
  - Screen Size (in inches)
  - Touchscreen and IPS display options
  - CPU and GPU brands
  - Operating System

---

## How It Works

1. **Input Specifications:** Users select their desired laptop configurations, such as brand, screen resolution, RAM size, and CPU type.
2. **Prediction:** Based on the inputs, the machine learning model predicts the most likely price of the laptop.
3. **Price Display:** The predicted price is displayed in INR (₹) with a neat format.

### Technologies Used:
- **Frontend:** Streamlit for creating the web-based interface.
- **Backend:** Python (Pandas, Numpy, Pickle) for model handling and data processing.
- **Model:** Random Forest Regressor for price prediction.

---

## Dataset & Model Information

The underlying dataset for this project contains detailed information about laptop configurations and their corresponding prices. Features engineered from this data include:

- **Brand and Model**
- **RAM (in GB)**
- **Storage (SSD and HDD sizes)**
- **Screen Resolution (PPI calculated from resolution and size)**
- **Processor (CPU)**
- **Graphics Card (GPU)**
- **Weight and Additional Features (e.g., Touchscreen, IPS Display)**

### Feature Engineering

The dataset went through several preprocessing steps:
- **Categorization:** Key features like CPU brand, GPU brand, and operating systems were categorized.
- **Cleaning:** Unnecessary columns were removed, and null values were handled.
- **Resolution Adjustment:** Screen resolution and pixel density were computed based on display size.
  
The model was trained using **Random Forest**, known for its accuracy and robustness in handling structured data.

---

## Conclusion

This project serves as a powerful tool for those looking to get a quick, accurate prediction of laptop prices based on specific configurations. It demonstrates the potential of machine learning in delivering valuable insights in a user-friendly format.

**Try it out now:** [Online Laptop Price Predictor](https://online-laptop-price-predictor-by-rahulakavector.streamlit.app/)

---

© Rahul-AkaVector. All rights reserved.