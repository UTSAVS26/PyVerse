import pandas as pd
import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Online Laptop Price Predictor",
    page_icon="💻",
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
st.title("ONLINE LAPTOP PRICE PREDICTION  💻💻💻")
st.markdown("<p style='text-align: right;'>by&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VECTOR 💻👨‍💻</p>",
            unsafe_allow_html=True)
st.text("""A laptop price predictor is a user-friendly web tool that estimates laptop prices based on specifications such as brand,
processor type, RAM size, storage capacity, and display size. 💻💰 It helps buyers, sellers, and curious individuals 
obtain estimated laptop prices, enabling informed decisions and budget planning. 📈💡 With easy online access, it allows 
effortless price comparisons and ensures value for money. 🌐💸""")
st.write("---")
st.header("Select Laptop 💻 Specifications")

# data = pickle.load(open('laptop_price_data.pkl', 'rb'))

data = pd.read_csv("cleaned_laptop_price_data.csv")
pipe = pickle.load(open("RandomForestModel.pkl", "rb"))

col1, col2 = st.columns(2)

company_list = sorted(data['Company'].unique())
company = col1.selectbox('Company', company_list,index=7)

type_list = sorted(data['TypeName'].unique())
typename = col1.selectbox('TypeName', type_list,index=4)

ram_list = sorted(data['Ram'].unique())
ram = col2.selectbox('Ram ( in GB )', ram_list,index=3)

weight = col2.number_input("Enter Weight of the laptop ( Between 0.69 and 4.69 ) ", min_value=0.69, max_value=4.69,
                         step=0.1,value=2.5)

touchscreen = col1.selectbox('Touch Screen', ['Yes', 'No'], index=1)
if touchscreen == "Yes":
    touchscreen = 1
else:
    touchscreen = 0

ips = col1.selectbox('IPS Display', ['Yes', 'No'], index=0)
if ips == "Yes":
    ips = 1
else:
    ips = 0

inches_list = [18.4, 17.3, 17.0, 15.6, 15.4, 15.0, 14.1, 14.0, 13.9, 13.5, 13.3, 13.0, 12.5, 12.3, 12.0, 11.6, 11.3,
               10.1]
inch = col2.selectbox('Screen Size (inches)', inches_list,index=3)

resolution_list = ['1920x1080', '1366x768', '3840x2160', '2560x1440', '2880x1800', '1600x900', '2560x1600', '2736x1824',
                   '1080x1920', '2560x1080', '1440x900', '1280x800']
resolution = col2.selectbox('Screen Resolution', resolution_list)

res = resolution.split('x')
ppi = ((int(res[0]) ** 2) + (int(res[1]) ** 2)) ** 0.5 / inch

cpu_list = sorted(data['Cpu brand'].unique())
cpu = col1.selectbox('CPU', cpu_list,index=3)

# ssd_list = sorted(data['SSD'].unique())
ssd_list = [0, 64, 128, 240, 256, 512]
ssd = col2.selectbox('SSD ( in GB )', ssd_list,index=4)

# hdd_list = sorted(data['HDD'].unique())
hdd_list = [0, 128, 256, 500, 512, 1024, 2048]
hdd = col2.selectbox('HDD ( in GB )', hdd_list, index=5)

gpu_list = sorted(data['GPU brand'].unique())
gpu = col1.selectbox('GPU', gpu_list,index=1)

os_list = sorted(data['os'].unique())
os = col1.selectbox('Operating System', os_list,index=2)


prediction = None
st.title("")
st1, st2, st3,st4,st5 = st.columns(5)

if st3.button("Predict 🔮 "):
    input_data = pd.DataFrame([[company, typename, ram, weight, touchscreen, ips, ppi, cpu, ssd, hdd, gpu, os]],
                              columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu brand',
                                       'SSD', 'HDD', 'GPU brand', 'os'])
    prediction = np.round((pipe.predict(input_data)[0]), 2)
    prediction = np.exp(prediction)
    # print(prediction)

if prediction is not None:
    formatted_prediction = "{:,.2f}".format(prediction)
    st.title(f"The predicted price of the Laptop 💻 is  ₹{formatted_prediction} 💸")



