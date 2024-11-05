from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from transformers import pipeline
from gtts import gTTS

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16 MB

# Create uploads and audio directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Initialize the image-to-text pipeline
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ''
    image_url = ''
    audio_url = ''
    
    if request.method == 'POST' and 'photo' in request.files:
        # Process the uploaded photo
        photo = request.files['photo']
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)

        # Convert the image to RGB and process
        image = Image.open(filepath).convert('RGB')

        # Generate caption
        captions = image_to_text(image)
        caption = captions[0]['generated_text'] 

        # Set image URL for display
        image_url = url_for('static', filename=f'uploads/{filename}')

        # Convert caption to audio using gtts
        if caption:
            tts = gTTS(text=caption, lang='en')
            audio_filename = f"{filename.rsplit('.', 1)[0]}.mp3"  # Same name but with .mp3 extension
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_filepath)
            audio_url = url_for('static', filename=f'audio/{audio_filename}')

    return render_template('index.html', caption=caption, image_url=image_url, audio_url=audio_url)

if __name__ == '__main__':
    app.run(debug=True)