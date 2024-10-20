import pandas as pd
import pickle
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Online House Price Predictor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="auto",
)


# ======================== This  section will remove the hamburger and watermark and footer and header from streamlit ===========
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            footer:after {
                            content:'\u00A9 Rahul-AkaVector. All rights reserved.';
	                        visibility: visible;
	                        display: block;
	                        position: relative;
	                        #background-color: red;
	                        padding: 5px;
	                        top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================== This  section will remove the hamburger and watermark and footer and header from streamlit ===========
st.title("ONLINE HOUSE PRICE PREDICTION  🏘️🏘️🏘️")
st.markdown("<p style='text-align: right;'>by&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VECTOR 💻👨‍💻</p>", unsafe_allow_html=True)
st.text("""An online house price prediction tool is a user-friendly web-based application that allows users to predict 
the price of houses effortlessly. 🏠💰 With intuitive features and a user-friendly interface, it enables individuals to 
select the location, number of bedrooms (BHK), number of bathrooms, and the square footage of the house, all within a 
convenient online platform. 🌐🔮 Whether you're a home buyer, seller, or simply curious about property prices, an online
house price prediction tool provides a hassle-free solution for obtaining estimated house prices, allowing you to make 
informed decisions and plan accordingly. 📈🏡""")


data = pd.read_csv("cleaned_housedata.csv")
pipe = pickle.load(open("RidgeModel.pkl","rb"))

locations = sorted(data['location'].unique())
# bhk = sorted(data['bhk'].unique())
# bath = sorted(data['bath'].unique())
# sqft = sorted(data['total_sqft'].unique())

st.subheader("Select Data")
ind = locations.index("Whitefield")
loc_opt = locations
location = st.selectbox("Select location 🌍 ", loc_opt, index=ind)
bhk_opt = [1, 2, 3, 4, 5, 6]
bhk = st.selectbox("Select BHK 🛏️ ", bhk_opt ,index=2)
bath_opt = [1, 2, 3, 4, 5, 6]
bath = st.selectbox("Select number of bathrooms 🛀 ", bath_opt , index=1)
sqft = st.number_input("Enter Square Foot between 300 - 14000 🏡", min_value=300, max_value=14000, step=1,value = 3000)

prediction = None

if st.button("Predict 🔮 "):
    if sqft:
        input_data = pd.DataFrame([[location, sqft, float(bath), bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = np.round((pipe.predict(input_data)[0] * 1e5), 2)
        print(prediction)


if prediction is not None:
    formatted_prediction = "{:,.2f}".format(prediction)
    st.success(f"The predicted price of the house is  ₹{formatted_prediction} 💸")
