from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

csv_file_path = "Prediction.csv"
df = pd.read_csv(csv_file_path)


STORMGLASS_API_KEY = "YOUR-STORM-GLASS-API-KEY"

model = joblib.load("crop_model.pkl")
le_soil = joblib.load("soil_encoder.pkl")
le_season = joblib.load("season_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")

soil_model = tf.keras.models.load_model("soil_model.h5")
soil_class_labels = ["Alluvial", "Black", "Clayey", "Laterite", "Red", "Sandy"]

@app.route('/')
def home():
    return render_template('index.html')

# Route for searching crops in CSV
def search_crop(crop_name):
    result = df[df.iloc[:, 0].str.lower() == crop_name.lower().strip()]
    return result.iloc[0].to_dict() if not result.empty else None

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    crop_name = request.form.get('crop_name', '').strip().lower()
    data = search_crop(crop_name)
    if data:
        return render_template('index.html', data=data)
    return render_template('index.html', error="Crop not found!")


@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        data = request.json
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude and Longitude are required"}), 400

        url = f"https://api.stormglass.io/v2/weather/point?lat={latitude}&lng={longitude}&params=airTemperature,humidity,precipitation&source=sg"
        headers = {"Authorization": STORMGLASS_API_KEY}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({"error": "Failed to fetch weather data"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        rainfall = float(request.form.get("Rainfall", 0))
        temperature = float(request.form.get("Temperature", 0))
        soil_type = request.form.get("SoilType", "")
        season = request.form.get("Season", "")

        
        if not all([rainfall, temperature, soil_type, season]):
            return render_template('index.html', error="All fields are required!")

       
        soil_type_encoded = le_soil.transform([soil_type])[0]
        season_encoded = le_season.transform([season])[0]

     
        input_data = [[rainfall, temperature, soil_type_encoded, season_encoded]]
        predictions = model.predict_proba(input_data)

        
        top_indices = predictions.argsort()[0][-3:][::-1]
        top_crops = le_crop.inverse_transform(top_indices)
        result_text = ", ".join(top_crops)

        return render_template('index.html', result=f"Recommended Crops: {result_text}")
    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

 
    prediction = soil_model.predict(img_array)
    predicted_class = soil_class_labels[np.argmax(prediction)]

    return jsonify({"soil_type": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

