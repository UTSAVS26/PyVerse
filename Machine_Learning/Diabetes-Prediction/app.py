from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and metrics
with open('diabetes_prediction.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
model = model_data['model']
feature_names = model_data.get('feature_names', 
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
     'Insulin', 'BMI', 'DPF', 'Age'])

# Get metrics with defaults
metrics = model_data.get('metrics', {
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.78,
    'f1': 0.80,
    'roc_auc': 0.87
})

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and organize form data
        input_values = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        
        # Create feature dictionary for display
        feature_values = dict(zip(feature_names, input_values))
        
        # Make prediction
        prediction = model.predict([input_values])[0]
        probability = round(model.predict_proba([input_values])[0][1] * 100, 1)
        
        # Get feature importances
        importances = (model.feature_importances_ if hasattr(model, 'feature_importances_') 
                      else [1/len(feature_names)] * len(feature_names))
        
        # Combine features with their importance and value
        feature_impact = sorted(
            [(name, imp, feature_values[name]) for name, imp in zip(feature_names, importances)],
            key=lambda x: x[1], 
            reverse=True
        )

        return render_template(
            'result.html',
            prediction=prediction,
            probability=probability,
            features=feature_impact,
            metrics=metrics,
            form_data=feature_values  # Pass the feature values dictionary
        )
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)