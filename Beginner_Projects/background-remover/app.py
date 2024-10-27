from flask import Flask, request, send_file, render_template, jsonify
from rembg import remove
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Received POST request")  
        if 'file' not in request.files:
            print("No file part")  
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            print("No selected file")  
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            print(f"Processing file: {file.filename}")  
            input_image = Image.open(file.stream)
            
            output_image = remove(input_image)
            
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            print("Image processed successfully")  
            return jsonify({"image": img_base64})
        else:
            print("File not allowed")  
            return jsonify({"error": "File not allowed"}), 400
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)