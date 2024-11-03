# Image Resizer and Filter Tool 🌄

## Overview

The **Image Resizer and Filter Tool** is a web-based application that allows users to upload images, resize them, apply various filters, and download the processed images. Built with HTML, CSS, and JavaScript, this tool provides an intuitive interface for manipulating images directly in the browser.🖼️

## Features 🩵


- Image upload
- Real-time preview
- Image filters (Grayscale, Sepia, Invert)
- Image resizing
- Text overlay with customizable font size, color, and position
- Adjustable compression quality
- Download processed image
  
## Technologies Used 👨🏻‍💻

- Frontend:
  - HTML5
  - CSS3
  - JavaScript (ES6+)
- Backend:
  - Python 3.7+
  - Flask
  - Pillow (PIL Fork)


## Installation ⬇

To run this project locally, follow these steps:

1. **Clone the Repository**:
```
git clone https://github.com/yourusername/image-resizer-filter-tool.git
```
2.**Navigate to the Project Directory**:
   ```
cd image-resizer-filter-tool
   ```
3.**Install the required Python packages:**
```
pip install flask flask-cors Pillow
```
4.**Run the Flask server:**
```
python app.py
```


4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Upload an image using the "Choose an image" button.
2. Select a filter from the dropdown menu (None, Grayscale, Sepia, Invert).
3. Adjust the resize percentage if needed.
4. Add text to the image if desired, and customize its appearance and position.
5. Set the compression quality (1-100).
6. The preview will update in real-time as you make changes.
7. Click the "Process Image" button to finalize the changes.
8. Click the "Download Processed Image" link to save the result.

## Project Structure 

- `index.html`: The main HTML file containing the structure of the web page.
- `style.css`: CSS file for styling the web page.
- `script.js`: JavaScript file handling client-side logic and interactions.
- `app.py`: Python Flask server handling image processing requests.
