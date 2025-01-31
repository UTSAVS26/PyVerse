# AI-Powered Grad-CAM Visualization for Brain Tumor MRI Classification

This project demonstrates the use of the Grad-CAM (Gradient-weighted Class Activation Mapping) technique to visualize and interpret the decision-making process of a deep learning model for classifying brain tumor MRI images.

![grad-cam](https://github.com/user-attachments/assets/376f4bbd-6dc4-441b-8da0-4be2ea08ff0b)

---

## Features
- **Deep Learning Model:** Uses a ResNet50 architecture pre-trained on ImageNet, fine-tuned for 4-class classification: Glioma, Meningioma, No Tumor, and Pituitary tumors.
- **Grad-CAM Visualization:** Highlights the regions in the MRI image that influenced the model’s prediction.
- **Custom Layers:** Includes additional dense and dropout layers for enhanced performance.
- **Visualization Options:**
  - Original Image
  - Grad-CAM Heatmap
  - Superimposed Heatmap on Original Image
  - Prediction Confidence Scores
- **Save Results:** Automatically saves visualizations for further analysis.

---

## Prerequisites

### Libraries
Ensure the following libraries are installed:
- TensorFlow
- NumPy
- Matplotlib
- Pillow (PIL)

### Dataset
This project assumes the presence of a dataset structured for brain tumor classification. Example path: `brain-tumor-mri-dataset/Testing/glioma/Te-gl_0037.jpg`.

---

## How It Works

### Model Architecture
The model uses ResNet50 as the base, with additional layers:
1. **Global Average Pooling Layer**
2. **Dense Layer** with 256 units and ReLU activation
3. **Dropout Layer** with a dropout rate of 0.5
4. **Output Layer** with softmax activation for 4 classes

### Grad-CAM
Grad-CAM highlights the significant regions of an image that contributed to a specific class prediction. It uses:
- The gradients of the target class with respect to the output of the last convolutional layer.
- A weighted map of activations to generate a heatmap.

---

## Usage

### 1. Clone and Setup
```bash
git clone https://github.com/<your-repo>/ai-ml-techniques.git
cd ai-ml-techniques
```

### 2. Run the Script

#### Example Usage
Update the `image_path` in the script:
```python
image_path = "C:/path/to/your/image.jpg"
```
Run the script:
```bash
python gradcam_visualization.py
```

### 3. Results
- Visualizations are saved in the `output/` directory:
  - `original_image.png`: Original MRI image.
  - `heatmap.png`: Grad-CAM heatmap.
  - `superimposed_image.png`: Superimposed heatmap on the original image.

---

## Code Overview

### Grad-CAM Function
```python
def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    ...
```
Generates a Grad-CAM heatmap and retrieves prediction probabilities.

### Visualization Function
```python
def apply_gradcam(image_path, model, target_size=(128, 128), opacity=0.4):
    ...
```
Applies Grad-CAM and displays visualizations, including a bar chart of prediction confidences.

### Save Results Function
```python
def save_results(original_img, heatmap, superimposed_img, output_dir="output/"):
    ...
```
Saves generated images to the specified directory.

---

## Project Goals
This project is part of a larger collection of AI and ML techniques aimed at advancing medical imaging and interpretability. By visualizing model predictions, we aim to:
- Enhance trust in AI models by providing insights into decision-making.
- Improve diagnostic support for medical professionals.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- TensorFlow for providing robust deep learning tools.
- Researchers and contributors to Grad-CAM for their groundbreaking work.
