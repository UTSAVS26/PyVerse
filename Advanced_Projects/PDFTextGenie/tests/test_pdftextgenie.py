import os
import pytest
from utils.pdf_utils import pdf_to_images, preprocess_image
from ocr.llama_ocr import ocr_images
from ocr.postprocess import clean_text
from PIL import Image

# Helper: create a blank image for OCR tests
def make_blank_image():
    return Image.new('RGB', (200, 100), color='white')

def test_preprocess_image_binarize():
    img = make_blank_image()
    processed = preprocess_image(img, {'binarize': True})
    assert processed.mode == '1'

def test_preprocess_image_denoise():
    img = make_blank_image()
    processed = preprocess_image(img, {'denoise': True})
    assert processed.size == img.size

def test_preprocess_image_contrast():
    img = make_blank_image()
    processed = preprocess_image(img, {'contrast': True, 'contrast_factor': 2.0})
    assert processed.size == img.size

def test_clean_text_hyphens():
    text = 'exam-\nple text.'
    cleaned = clean_text(text, fix_hyphens=True)
    assert 'example' in cleaned

def test_clean_text_merge_paragraphs():
    text = 'This is a line.\nThis is another.'
    cleaned = clean_text(text, merge_paragraphs=True)
    assert '\n' not in cleaned or cleaned.count('\n') < 2

def test_ocr_images_blank(monkeypatch):
    # Patch EasyOCREngine to avoid actual OCR
    from ocr.llama_ocr import EasyOCREngine
    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass
        def recognize(self, images):
            return 'dummy text'
    monkeypatch.setattr('ocr.llama_ocr.EasyOCREngine', DummyEngine)
    img = make_blank_image()
    result = ocr_images([img], model='easyocr')
    assert 'dummy text' in result

def test_pdf_to_images_invalid():
    with pytest.raises((FileNotFoundError, ValueError)):
        pdf_to_images('nonexistent.pdf')