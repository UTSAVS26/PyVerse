## Dataset

### Source
The dataset is available on Roboflow Universe:
- Dataset Link: [AI Leprosy Detection Dataset](https://universe.roboflow.com/intelligent-systems-1b35z/ai-leprosy-bbdnr)
- Format: COCO JSON
- Classes: Binary classification (Leprosy/Non-Leprosy)

### Dataset Structure
The dataset is split into:
- Training set
- Validation set
- Test set

Each set contains:
- RGB images
- COCO format annotations (_annotations.coco.json)

### Accessing the Dataset
1. Visit the [dataset page](https://universe.roboflow.com/intelligent-systems-1b35z/ai-leprosy-bbdnr)
2. Create a Roboflow account if needed
3. Download the dataset in COCO format
4. Place the downl# Leprosy Detection System

## Overview
This project implements an automated system for detecting leprosy using machine learning and image processing techniques. The system aims to assist healthcare professionals in early diagnosis of leprosy by analyzing skin lesion images.

## Features
- Automated analysis of skin lesion images
- Support for multiple image formats (JPG, PNG)
- Pre-processing pipeline for image enhancement
- Deep learning model for lesion classification
- User-friendly interface for healthcare professionals
- Detailed report generation

## Hardware Requirements

### Minimum Requirements
- 2x NVIDIA Tesla T4 GPUs (or equivalent)
- 16GB+ GPU memory
- 32GB RAM recommended
- 50GB available storage space

### Development Setup
The model was developed and tested on:
- NVIDIA Tesla T4 GPUs (2x)
- CUDA 11.x
- PyTorch with CUDA support

Note: Training time may vary significantly with different hardware configurations. The model is optimized for multi-GPU training using DataParallel.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/leprosy-detection.git
cd leprosy-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/train.py
```

### Testing/Inference
The model can be used for inference using the provided testing script:

```bash
python src/test.py
```

Key features of the testing module:
- Supports batch processing of multiple images
- Displays predictions with confidence scores
- Visualizes results using matplotlib
- Handles both CPU and GPU inference

#### Testing Configuration
```python
# Example configuration
model_path = 'best_custom_vit_mo.pth'
num_classes = 2
class_names = ['Leprosy', 'No Lep']

# Image preprocessing parameters
image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

#### Custom Inference
```python
from model import CustomViT, load_model
from utils import preprocess_image, predict

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('best_custom_vit_mo.pth', num_classes=2, device=device)

# Process single image
image_tensor = preprocess_image('path/to/image.jpg', mean, std)
category_id, probability = predict(model, image_tensor, device)
```

## Dataset
The project uses a custom dataset format with COCO-style annotations:
- Training, validation, and test sets are provided separately
- Images are annotated with binary labels (Leprosy/Non-Leprosy)
- Dataset is loaded using a custom `LeprosyDataset` class extending `torch.utils.data.Dataset`

## Project Structure
```
leprosy-detection/
├── src/
│   ├── train.py           # Training script
│   ├── test.py           # Inference script
│ 
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   ├── valid/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   └── test/
│       ├── images/
│       └── _annotations.coco.json
├── models/              # Saved model checkpoints
├── results/            # Training results and visualizations
├── docs/
└── requirements.txt
```

## Model Architecture
The system implements a Custom Vision Transformer (ViT) architecture specifically designed for leprosy detection:

### Key Components
- **Patch Embedding**: Converts input images (224x224) into patches (16x16) and projects them to the embedding dimension (768)
- **Transformer Blocks**: 12 layers of transformer blocks with:
  - Multi-head self-attention (12 heads)
  - Layer normalization
  - MLP with GELU activation
  - Dropout for regularization
- **Classification Head**: Final layer for binary classification (Leprosy vs Non-Leprosy)

### Training Details
- Batch Size: 32
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Cross Entropy Loss
- Training Duration: 20 epochs
- Data Augmentation: Resize, Normalization (ImageNet stats)
- Model Selection: Best model saved based on validation accuracy

## Performance Metrics
The model's performance is comprehensively evaluated using various metrics:
- Training and validation metrics tracked per epoch
- Confusion matrices generated for detailed error analysis
- Final evaluation on test set includes:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1 Score
  - Loss values

### Visualization
- Training history plots showing:
  - Loss curves (training and validation)
  - Accuracy progression
  - Precision, Recall, and F1 score trends
- Confusion matrices for each epoch and final test results
- All visualizations saved automatically with timestamps

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- World Health Organization (WHO) for providing clinical guidelines
- Contributing healthcare institutions for providing validated datasets
- Research partners and medical professionals for expert guidance

## Contact
- Project Maintainer: [Mohak]
- Email: [mohakgupta0981@gmail.com]
- Project Link: https://github.com/lukiod/Levit

## Disclaimer
This tool is designed to assist healthcare professionals and should not be used as the sole basis for diagnosis. Always consult qualified medical professionals for proper diagnosis and treatment.