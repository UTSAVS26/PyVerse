# Background Remover 

This is a Flask-based web application that allows users to upload images and remove their backgrounds using the `rembg` library. It supports common image formats like PNG, JPG, and JPEG.

## Features
- **Flask Backend**: A lightweight web server built with Flask to handle file uploads and image processing requests.
- **rembg Library**: Utilizes the powerful `rembg` tool for background removal.
- **Pillow (PIL)**: Used for image handling and manipulation.
- **Base64 Image Encoding**: After processing, the app returns the image in Base64 format, making it easy to embed in HTML or share over APIs.
- **Frontend Interface**: A minimal HTML, CSS, JS webpage where users can upload images for background removal.


## Installation

### Prerequisites
- Python 3.x
- `pip` (Python package installer)

### Steps

1. **Clone the repository:**
   ```bash
   cd background-removal-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   # .\venv\Scripts\activate  # For Windows
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app:**
   ```bash
   python app.py
   ```

5. **Access the app** in your browser.

## Usage

1. Use the upload form on the page to select an image (PNG, JPG, or JPEG).
2. After submitting the image, the app will process the image and remove the background.
3. The result will be returned as a Base64-encoded string, which you can use in web applications.


