from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = Flask(__name__, static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def send_file(path):
    return send_from_directory('.', path)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data['image'].split(',')[1]
    quality = int(data['quality'])

    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Save processed image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    processed_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({'processedImage': f"data:image/jpeg;base64,{processed_image}"})

if __name__ == '__main__':
    app.run(debug=True)