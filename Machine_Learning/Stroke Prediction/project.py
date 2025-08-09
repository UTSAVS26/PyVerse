import streamlit as st
import pickle

import requests
import folium
from streamlit_folium import st_folium

# Load trained model
# Load trained model
import os

model_path = os.path.join(os.path.dirname(__file__), "model123.pkl")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure model123.pkl is in the same directory as this script.")
    st.stop()

# Get coordinates from pincode using OpenStreetMap
def get_location_osm(pincode):
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&country=India&format=json"
            response = requests.get(url, 
                              headers={"User-Agent": "HeartStrokePrediction/1.0"},
                              timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except requests.RequestException as e:
        st.error(f"Failed to get location: {str(e)}")
    except (KeyError, IndexError, ValueError) as e:
        st.error(f"Invalid response format: {str(e)}")
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
        out center 10;
        """
        response = requests.get("https://overpass-api.de/api/interpreter", 
                            params={"data": query},
                            timeout=30)
        response.raise_for_status()
        hospitals = response.json()['elements']
        results = []
        for h in hospitals:
            name = h.get("tags", {}).get("name", "Unnamed Hospital")
            hlat = h.get("lat") or h.get("center", {}).get("lat")
            hlon = h.get("lon") or h.get("center", {}).get("lon")
            if hlat and hlon:
                results.append((name, hlat, hlon))
        return results
        except requests.RequestException as e:
          st.warning(f"Could not fetch hospital data: {str(e)}")
          return []
        except (KeyError, ValueError) as e:
          st.warning(f"Invalid hospital data format: {str(e)}")
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
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", [0, 1])
work_type = st.selectbox("Work Type (0–4)", [0, 1, 2, 3, 4])
residence_type = st.selectbox("Residence Type (0=Rural, 1=Urban)", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", 
                                   min_value=0.0, max_value=500.0, 
                                   value=100.0,
                                   help="Normal range: 70-140 mg/dL")
bmi = st.number_input("BMI", 
                     min_value=10.0, max_value=60.0, 
                     value=25.0,
                     help="Normal range: 18.5-24.9")
smoking_status = st.selectbox("Smoking Status (0–3)", [0, 1, 2, 3])
pincode = st.text_input("Enter Pincode", max_chars=10)

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

