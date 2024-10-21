## ESRGAN_Model ![Repo Stars](https://img.shields.io/github/stars/dino65-dev/lox?style=social) ![Forks](https://img.shields.io/github/forks/dino65-dev/lox?style=social) ![Watchers](https://img.shields.io/github/watchers/dino65-dev/lox?style=social)
 This is ESRGAN_Model Finetuned with 4k video frames.
## ESRGAN (Enhanced SRGAN) Fine Tuned Using [:rocket: [[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)] Model.

# Quick Test
### Dependencies
- Python 3
- [Pytorch >= 1.0](https://pytorch.org/)  (CUDA version >= 7.5 if installing with CUDA.[More Details](https://pytorch.org/get-started/previous-versions/))
- Python packages: ``` pip install numpy opencv-python ```
## Test Models
1. Clone this github repo.
```
git clone https://github.com/dino65-dev/ESRGAN_Model.git
cd ESRGAN_Model
```
2. Place your own low-resolution images in ```./LR``` folder. (There are two sample images - baboon and comic).
3. I've downloaded and fine tuned the model. But you can download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the models in ```./models```. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).
4. Run test. We provide ESRGAN model and RRDB_PSNR model and you can config in the ```test.py```.
```
 python test.py
```







