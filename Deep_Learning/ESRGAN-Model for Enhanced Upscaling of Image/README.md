## ESRGAN_Model ![Repo Stars](https://img.shields.io/github/stars/dino65-dev/lox?style=social) ![Forks](https://img.shields.io/github/forks/dino65-dev/lox?style=social) ![Watchers](https://img.shields.io/github/watchers/dino65-dev/lox?style=social)
 This is ESRGAN_Model Finetuned with 4k video frames.

# Quick Test
### Dependencies
- Python 3
- [Pytorch >= 1.0](https://pytorch.org/)  (CUDA version >= 7.5 if installing with CUDA.[More Details](https://pytorch.org/get-started/previous-versions/))
- Python packages: ``` pip install numpy opencv-python ```
2. Go to ``` ./figures``` folder to see how and well it works. 
3. Place your own low-resolution images in ```./LR``` folder.
4. I've downloaded and fine tuned the model. But you can download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the models in ```./models```. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).
5. Run test. We provide ESRGAN model and RRDB_PSNR model and you can config in the ```test.py```.
```
 python test.py 
```







