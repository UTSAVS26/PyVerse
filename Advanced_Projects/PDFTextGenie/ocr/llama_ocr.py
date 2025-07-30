import easyocr
import numpy as np

# TrOCR imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

class EasyOCREngine:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize(self, images, preserve_layout=False):
        text = ''
        for img in images:
            try:
                if hasattr(img, 'convert'):
                    img = img.convert('RGB')
                img_np = np.array(img)
                if preserve_layout:
                    # Use detail=1 to get bounding boxes and sort by y
                    results = self.reader.readtext(img_np, detail=1, paragraph=False)
                    # results: list of (bbox, text, conf)
                    # Sort by top-left y, then x
                    lines = sorted(results, key=lambda r: (min([pt[1] for pt in r[0]]), min([pt[0] for pt in r[0]])))
                    for bbox, line_text, conf in lines:
                        x = min(pt[0] for pt in bbox)
                        indent = int(x // 30)  # 30 pixels per indent level
                        output_line = ' ' * (indent * 4) + line_text
                        text += output_line + '\n'
                else:
                    results = self.reader.readtext(img_np, detail=0, paragraph=True)
                    text += '\n'.join(results) + '\n'
            except Exception as e:
                text += f'\n[OCR ERROR: {e}]\n'
        return text

class TrOCREngine:
    def __init__(self, gpu=True):
        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)

    def recognize(self, images):
        text = ''
        for img in images:
            try:
                if hasattr(img, 'convert'):
                    img = img.convert('RGB')
                pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text += result + '\n'
            except Exception as e:
                text += f'\n[OCR ERROR: {e}]\n'
        return text

def ocr_images(images, model='easyocr', languages=['en'], gpu=False, preserve_layout=False):
    if model == 'easyocr':
        engine = EasyOCREngine(languages=languages, gpu=gpu)
        return engine.recognize(images, preserve_layout=preserve_layout)
    elif model == 'trocr':
        engine = TrOCREngine(gpu=gpu)
        return engine.recognize(images)
    else:
        raise NotImplementedError(f'OCR model {model} not implemented yet.') 