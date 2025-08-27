import streamlit as st
import joblib
from keras.models import load_model
import pandas as pd
from pathlib import Path

@st.cache_resource(show_spinner=False)
def load_artifacts():
   base = Path(__file__).resolve().parent
   model = load_model(base / "artifacts" / "my_model (1).keras")
   preprocessor = joblib.load(base / "artifacts" / "preprocessor (1).pkl")
   y_scaler = joblib.load(base / "artifacts" / "y_scaler (2).pkl")
   return model, preprocessor, y_scaler

# model = load_model(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\my_model (1).keras")
# preprocessor = joblib.load(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\preprocessor (1).pkl")
# y_scaler=joblib.load(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\y_scaler (2).pkl")
def show_crop_production_page():
    model, preprocessor, y_scaler = load_artifacts()
    st.title(" 🌱 Crop Production Prediction 🌱")
    st.write("Predict the crop production based on various agricultural parameters.")
    st.write("Please provide the following details:")
    col1, col2 = st.columns(2)
    with col1:
        states = [
            'Select','ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'GOA',
           'GUJARAT', 'HARYANA', 'JAMMU AND KASHMIR', 'KARNATAKA', 'KERALA',
           'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR', 'MEGHALAYA', 'MIZORAM',
           'NAGALAND', 'ODISHA', 'PUNJAB', 'RAJASTHAN', 'TAMIL NADU',
           'TELANGANA', 'UTTAR PRADESH', 'WEST BENGAL', 'CHANDIGARH',
           'DADRA AND NAGAR HAVELI', 'HIMACHAL PRADESH', 'PUDUCHERRY',
           'SIKKIM', 'TRIPURA', 'ANDAMAN AND NICOBAR ISLANDS', 'CHHATTISGARH',
           'UTTARAKHAND', 'JHARKHAND'
        ]
        state = st.selectbox("Select your State:", states)
        area = st.number_input("Area (in hectares)", min_value=0.01, step=0.01, format="%.2f")
        rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, max_value=10000.0, step=1.0)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)

        crop_type = st.selectbox("Select crop type :", ["Select","Kharif", "Rabi", "Summer", "Whole Year"])
        crops = [
        'Select', 'COTTON', 'HORSEGRAM', 'JOWAR', 'MAIZE', 'MOONG', 'RAGI', 'RICE',
        'SUNFLOWER', 'WHEAT', 'SESAMUM', 'SOYABEAN', 'RAPESEED', 'JUTE',
        'ARECANUT', 'ONION', 'POTATO', 'SWEETPOTATO', 'TAPIOCA',
        'TURMERIC', 'BARLEY', 'BANANA', 'CORIANDER', 'GARLIC',
        'BLACKPEPPER', 'CARDAMOM', 'CASHEWNUTS', 'BLACKGRAM', 'COFFEE',
        'LADYFINGER', 'BRINJAL', 'CUCUMBER', 'GRAPES', 'MANGO', 'ORANGE',
        'PAPAYA', 'TOMATO', 'CABBAGE', 'BOTTLEGOURD', 'PINEAPPLE',
        'CARROT', 'RADISH', 'BITTERGOURD', 'DRUMSTICK', 'JACKFRUIT',
        'CAULIFLOWER', 'WATERMELON', 'ASHGOURD', 'BEETROOT', 'POMEGRANATE',
        'RIDGEGOURD', 'PUMPKIN', 'APPLE', 'GINGER'
        ]
        crop = st.selectbox("Select a Crop:", crops)
        N = st.number_input("Nitrogen Content in soil (kg/ha)", min_value=0.0, max_value=1000.0, step=1.0)
        P = st.number_input("Phosphorus Content in soil (kg/ha)", min_value=0.0, max_value=1000.0, step=1.0)
        K = st.number_input("Potassium Content in soil (kg/ha)", min_value=0.0, max_value=1000.0, step=1.0)
        pH = st.number_input("pH value of soil", min_value=0.0, max_value=14.0, step=0.1, format="%.1f")
        production_in_tons = st.number_input("Historical Production (tons)", min_value=0.0, step=0.01, format="%.2f")
        

    with col2:
        if st.button("Predict Crop Production"):
            if state =="Select" or crop_type == "Select" or crop == "Select":
                st.error("Please select valid options for State, Crop Type, and Crop.")
                return 
            if not (0.0 <= pH <= 14.0) or area <= 0 or rainfall < 0 or N < 0 or P < 0 or K < 0 or production_in_tons < 0:
                st.error("Please enter valid values. Note: pH must be between 0 and 14.")
                return 
            
            input_data={
                'State_Name':state.lower(),
                'Crop_Type':crop_type.lower(),
                'Crop':crop.lower(),
                'N':N,
                'P':P,
                'K':K,
                'pH':pH,
                'rainfall':rainfall,
                'temperature':temperature,
                'Area_in_hectares':area,
                'Yield_ton_per_hec':production_in_tons/area if area!=0 else 0
            }
            input_data = pd.DataFrame([input_data])
            processed_data = preprocessor.transform(input_data)

            with st.spinner("Predicting..."):
               scaled_pred = model.predict(processed_data, verbose=0)
            prediction = y_scaler.inverse_transform(scaled_pred)

            st.success(f"The predicted crop production is {prediction[0][0]:.2f} tons")
            
            st.balloons()

show_crop_production_page()

