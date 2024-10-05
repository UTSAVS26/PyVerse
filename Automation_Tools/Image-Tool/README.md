# Image Tool

## Overview
This Python-based Image tool, built using **Streamlit**, allows users to perform various image manipulation tasks such as resizing, cropping, format conversion, and enhancement. With an intuitive, web-based interface, users can make adjustments with real-time previews. This tool is designed for single-image processing, providing a simple yet powerful platform for common image editing tasks.

## Features
- **Image Resizing:**
  - Resize images by specifying dimensions (height and width) or by percentage.
  - Real-time preview of resized images.
  - Estimate and display the file size based on new dimensions.

- **Image Cropping:**
  - Interactive cropping tool using `streamlit-cropper`, allowing users to adjust crop boxes and aspect ratios.
  - Manual input for specific crop dimensions.
  - Live preview of the cropped image.

- **Image Format Conversion:**
  - Convert images between popular formats such as JPEG, PNG, BMP, and GIF.
  - Adjust compression quality for formats like JPEG.
  - Preview the image in the selected format before conversion.

- **Image Enhancement:**
  - Adjust brightness, contrast, sharpness, and color using `Pillow`'s `ImageEnhance`.
  - Live preview of the enhancements before applying them.

- **Drag-and-Drop Support:**
  - Drag and drop images directly into the interface for quick uploads.

- **Save and Download:**
  - Download the edited image in the selected format directly from the interface.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/UTSAVS26/PyVerse/tree/main/Automation_Tools/Image-Tool.git
   cd Image-Tool
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies:**
   Install the required dependencies using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   Once dependencies are installed, run the tool with:
   ```bash
   streamlit run Image_Tool.py
   ```


## Requirements

- Python 3.x
- Required libraries (installed via `requirements.txt`):
  - `Streamlit` (for the web-based GUI)
  - `Pillow` (for image processing)
  - `streamlit-cropper` (for cropping functionality)
