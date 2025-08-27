import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

model = load_model(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\my_model (1).keras")
preprocessor = joblib.load(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\preprocessor (1).pkl")
y_scaler=joblib.load(r"D:\Desktop\PyVerse\Deep_Learning\Crop Production\y_scaler (2).pkl")
def show_crop_production_page():
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
        area = st.number_input("Area (in hectares)")
        rainfall = st.number_input("Rainfall (in mm)")
        temperature = st.number_input("Temperature (in degree Celsius)")
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
        N = st.number_input("Nitrogen Content in soil (in kg/ha)")
        P = st.number_input("Phosphorus Content in soil (in kg/ha)")
        K = st.number_input("Potassium Content in soil (in kg/ha)")
        pH = st.number_input("PH value of soil")
        production_in_tons =st.number_input("Historical Production (in tons)")

    with col2:
        if st.button("Predict Crop Production"):
            if state =="Select" or crop_type == "Select" or crop == "Select":
                st.error("Please select valid options for State, Crop Type, and Crop.")
                return 
            if area <=0 or rainfall <=0 or temperature <=0 or N<0 or P<0 or K<0 or pH<=0 or production_in_tons<0:
                st.error("Please enter valid Positive values for all numerical inputs.")
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

            scaled_pred = model.predict(processed_data)
            prediction = y_pred = y_scaler.inverse_transform(scaled_pred)

            st.success(f"The predicted crop production is {prediction[0][0]:.2f} tons")
            
            st.balloons()

show_crop_production_page()

