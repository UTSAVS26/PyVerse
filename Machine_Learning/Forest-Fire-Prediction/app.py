from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # This would normally handle the prediction logic
    # For frontend-only, we'll just return dummy data
    if request.method == 'POST':
        data = request.form.to_dict()
        # In a real app, you would process the data with your model here
        return jsonify({
            'status': 'success',
            'prediction': {
                'ridge': 12.45,  # Sample prediction values
                'polynomial': 15.67
            }
        })

if __name__ == '__main__':
    app.run(debug=True)