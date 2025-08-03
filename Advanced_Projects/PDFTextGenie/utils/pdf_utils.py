from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance

def preprocess_image(img, preprocess_opts=None):
    if not preprocess_opts:
        return img
    # Binarization
    if preprocess_opts.get('binarize'):
        img = img.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
    # Denoise (simple median filter)
    if preprocess_opts.get('denoise'):
        img = img.filter(ImageFilter.MedianFilter(size=3))
    # Contrast
    if preprocess_opts.get('contrast'):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(preprocess_opts.get('contrast_factor', 1.5))
    return img

def pdf_to_images(pdf_path, dpi=300, preprocess_opts=None):
    """
    Convert PDF to list of PIL Images. Optionally preprocess each image.
    preprocess_opts: dict, e.g. {'binarize': True, 'denoise': True, 'contrast': True, 'contrast_factor': 2.0}
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    if preprocess_opts:
        images = [preprocess_image(img, preprocess_opts) for img in images]
    return images 