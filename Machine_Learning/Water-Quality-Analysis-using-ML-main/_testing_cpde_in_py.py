import pickle
import pandas as pd
import numpy as np


with open('naive_bayes_model.pkl', 'rb') as model_file:
    nb_model_loaded = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler_loaded = pickle.load(scaler_file)


example_data = {
    'ph': 1.2,  
    'Hardness': 150,
    'Solids': 350,
    'Chloramines': 3.0,
    'Sulfate': 250,
    'Conductivity': 400,
    'Organic_carbon': 5.0,
    'Trihalomethanes': 50,
    'Turbidity': 2.0
}


example_df = pd.DataFrame([example_data])


example_scaled = scaler_loaded.transform(example_df)


example_prediction = nb_model_loaded.predict(example_scaled)

potability = "Potable" if example_prediction[0] == 1 else "Not Potable"
print(f"\nExample Water Sample Prediction: {potability}")
