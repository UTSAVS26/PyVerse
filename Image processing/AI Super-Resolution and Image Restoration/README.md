# AI-Powered Image Restoration and Enhancement Tool

## Project Description
This project aims to develop an **AI-Powered Image Restoration Tool** that revitalizes low-quality historical photos by upscaling, denoising, and adding realistic color. Using advanced deep learning techniques, it transforms degraded images into vibrant, high-quality visuals while preserving their historical context. This tool is perfect for heritage conservation, family archiving, and historical documentation.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - OpenCV: For image processing tasks
  - Pillow: For handling image files
  - PyTorch: For implementing deep learning models
  - torchvision: For image transformations and model utilities
- **Models**:
  - SRCNN (Super-Resolution Convolutional Neural Network): For image upscaling
  - DnCNN: For image denoising
  - Pre-trained colorization models (e.g., U-Net): For adding color to grayscale images

 ## Datasets
To train or fine-tune the SRCNN model, you can use the following datasets:

1. **DIV2K**: A high-quality dataset for super-resolution with 800 training images.
   - [Download DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

2. **Flickr2K**: Contains 2,656 high-resolution images, useful as a complement to DIV2K.
   - [Download Flickr2K Dataset](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

3. **BSD300** and **BSD500**: Classical image processing datasets.
   - [Download BSD300 Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz)
   - [Download BSD500 Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS500.tgz)

4. **Set5 and Set14**: Small datasets often used for testing super-resolution models.
   - [Download Set5 & Set14 Datasets](https://github.com/jbhuang0604/SelfExSR/tree/master/data)

## Pre-trained SRCNN Model
To skip the training process, you can use a pre-trained SRCNN model:
- [Download SRCNN Pretrained Model](https://github.com/leftthomas/SRGAN/blob/master/model/srresnet.pth)


  # Connect with Me

- **GitHub**: [Peart-Guy](https://github.com/Peart-Guy)
- **LinkedIn**: [Ankan Mukhopadhyay](https://www.linkedin.com/in/ankan-mukhopadhyaypeartguy/)

