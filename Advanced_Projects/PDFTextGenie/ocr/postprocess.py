import re

def ollama_format_question_paper(ocr_text, model='llama2', filename=None):
    import requests
    subject_line = f"Subject: {filename}" if filename else "Subject:"
    prompt = (
        "Institute: ABV-IIITM Gwalior\n"
        f"{subject_line}\n\n"
        "the above lines should be at the very beginning of the output"
        "You are an expert at formatting academic question papers. "
        "Given the raw OCR output of a question paper, rewrite it as a clean, well-formatted document.\n"
        "- Fix any OCR errors, spelling mistakes, and restore all numbering and section headers.\n"
        "- Clearly separate questions and sub-questions.\n"
        "- Use bold or ALL CAPS for section headers (e.g., PART A, PART B).\n"
        "- Use proper indentation for sub-questions.\n"
        "- Remove any extraneous line breaks, artifacts, or page numbers.\n"
        "- Output only the clean, well-formatted question paper, nothing else.\n"
        "- Preserve marks/points if present.\n"
        "- Use clear formatting for instructions and special notes.\n\n"
        f"{ocr_text}"
    )
    print(f"[Ollama] Sending to model: {model}")
    print(f"[Ollama] Prompt (first 200 chars): {prompt[:200]}")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]

def clean_text(text, fix_hyphens=True, merge_paragraphs=True, preserve_layout=False, spellcheck=False, column_detection=False, ai_correction=False, ollama_ai=False, ollama_model='llama2', filename=None):
    # Remove extra newlines
    if preserve_layout:
        text = re.sub(r'\n{3,}', '\n\n', text)
    else:
        text = re.sub(r'\n{2,}', '\n', text)
    if fix_hyphens:
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    if merge_paragraphs and not preserve_layout:
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Spellcheck/grammar correction
    if spellcheck:
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            def correct_word(word):
                return spell.correction(word) if word.isalpha() else word
            text = ' '.join([correct_word(w) for w in text.split()])
        except ImportError:
            pass
        try:
            import language_tool_python
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(text)
            text = language_tool_python.utils.correct(text, matches)
        except ImportError:
            pass
    # Column detection (placeholder)
    if column_detection:
        # TODO: Implement column detection using bounding boxes
        pass
    # Ollama AI correction (preferred if enabled)
    if ollama_ai:
        try:
            return ollama_format_question_paper(text, model=ollama_model, filename=filename)
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            pass
    # AI correction (local, using flan-t5-large and improved prompt)
    if ai_correction:
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            nlp = pipeline('text2text-generation', model='google/flan-t5-large', device=device)
            subject_line = f"Subject: {filename}" if filename else "Subject:"
            prompt = (
                "Institute: ABV-IIITM Gwalior\n"
                f"{subject_line}\n\n"
                "You are an expert at formatting academic question papers. Given the raw OCR output of a question paper, rewrite it as a clean, well-formatted document.\n"
                "- Fix any OCR errors, spelling mistakes, and restore all numbering and section headers.\n"
                "- Clearly separate questions and sub-questions.\n"
                "- Use bold or ALL CAPS for section headers (e.g., PART A, PART B).\n"
                "- Use proper indentation for sub-questions.\n"
                "- Remove any extraneous line breaks, artifacts, or page numbers.\n"
                "- Output only the clean, well-formatted question paper, nothing else.\n"
                "- Preserve marks/points if present.\n"
                "- Use clear formatting for instructions and special notes.\n\n"
            )
            max_chunk = 800
            lines = text.split('\n')
            formatted = []
            for i in range(0, len(lines), 20):
                chunk = '\n'.join(lines[i:i+20])
                if chunk.strip():
                    result = nlp(prompt + chunk, max_length=max_chunk)[0]['generated_text']
                    formatted.append(result)
            text = '\n'.join(formatted)
        except ImportError:
            pass
        except Exception:
            pass
    return text.strip() 