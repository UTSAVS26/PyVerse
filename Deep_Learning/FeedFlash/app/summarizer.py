from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Global variables for lazy loading
_model = None
_tokenizer = None
MODEL_NAME = "Arihant-Bhandari/feedflash-flan-t5"  # Updated model

def _load_model():
    """Lazy load the model and tokenizer."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        try:
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}") from e
    return _model, _tokenizer

def summarize_text(text):
    """Summarize the input text using a pre-trained T5 model.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the input text is not a non-empty string.
        RuntimeError: If summarization fails due to model or tokenizer errors.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    if len(text.strip()) == 0:
        return "No content to summarize."

    try:
        model, tokenizer = _load_model()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=600  # Increased input token limit
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,      # Increased output token limit
                num_beams=4,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}") from e