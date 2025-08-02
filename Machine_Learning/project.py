import streamlit as st
import pickle
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

# Load trained model
model = pickle.load(open("D:/hs/model23.pkl", "rb"))  # Use relative path for portability

# Get coordinates from pincode using OpenStreetMap
def get_location_osm(pincode):
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

# Get nearby hospitals using Overpass API
def get_nearby_hospitals(lat, lon, radius=5000):
    try:
        query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:{radius},{lat},{lon});
          way["amenity"="hospital"](around:{radius},{lat},{lon});
          relation["amenity"="hospital"](around:{radius},{lat},{lon});
        );
        out center;
        """
        response = requests.get("http://overpass-api.de/api/interpreter", params={"data": query})
        hospitals = response.json()['elements']
        results = []
        for h in hospitals:
            name = h.get("tags", {}).get("name", "Unnamed Hospital")
            hlat = h.get("lat") or h.get("center", {}).get("lat")
            hlon = h.get("lon") or h.get("center", {}).get("lon")
            if hlat and hlon:
                results.append((name, hlat, hlon))
        return results
    except:
        return []

# Plot hospitals on Folium map
def plot_hospitals_on_map(lat, lon, hospitals):
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], tooltip="Your Location", icon=folium.Icon(color='red')).add_to(m)
    for name, hlat, hlon in hospitals:
        folium.Marker([hlat, hlon], tooltip=name, icon=folium.Icon(color='green')).add_to(m)
    return m

# ---- Streamlit UI ----
st.title("🩺 Stroke Risk Predictor + Nearby Hospital Locator")

st.write("### Enter your health details below:")

# Inputs — must match model training order
gender = st.selectbox("Gender (0=F, 1=M, 2=Other)", [0, 1, 2])
age = st.number_input("Age", 0.0, 120.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", [0, 1])
work_type = st.selectbox("Work Type (0–4)", [0, 1, 2, 3, 4])
residence_type = st.selectbox("Residence Type (0=Rural, 1=Urban)", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")
smoking_status = st.selectbox("Smoking Status (0–3)", [0, 1, 2, 3])
pincode = st.text_input("Enter Pincode")

# ----- Predict Button -----
if st.button("Predict & Find Hospitals"):
    user_input = [gender, age, hypertension, heart_disease, ever_married,
                  work_type, residence_type, avg_glucose_level, bmi, smoking_status]

    prediction = model.predict([user_input])[0]
    st.session_state["prediction"] = prediction

    if prediction == 1:
        lat, lon = get_location_osm(pincode)
        st.session_state["lat"] = lat
        st.session_state["lon"] = lon
        if lat and lon:
            hospitals = get_nearby_hospitals(lat, lon)
            st.session_state["hospitals"] = hospitals
        else:
            st.session_state["hospitals"] = []
    else:
        st.session_state["hospitals"] = []

# ----- Display Results -----
if "prediction" in st.session_state:
    if st.session_state["prediction"] == 1:
        st.error("⚠️ Risk of Stroke Detected")

        lat = st.session_state.get("lat")
        lon = st.session_state.get("lon")
        hospitals = st.session_state.get("hospitals", [])

        if lat and lon:
            if hospitals:
                st.success("Nearby Hospitals:")
                for i, (name, hlat, hlon) in enumerate(hospitals[:5], 1):
                    st.write(f"{i}. {name} ({hlat:.4f}, {hlon:.4f})")
                m = plot_hospitals_on_map(lat, lon, hospitals[:5])
                st_folium(m, width=700)
            else:
                st.warning("No hospitals found nearby.")
        else:
            st.warning("Invalid or unrecognized pincode.")
    else:
        st.success("✅ No immediate stroke risk detected.")
