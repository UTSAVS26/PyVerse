from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("Arihant-Bhandari/feedflash-t5")
tokenizer = AutoTokenizer.from_pretrained("Arihant-Bhandari/feedflash-t5")

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=412)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)