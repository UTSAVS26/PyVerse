# 📄 Resume Scanner (Python)

Extract Name, Email, Phone, and Skills from PDF resumes for HR filtering.

## 🚀 How to Run

1. Install dependencies:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Add resume in `data/sample_resume.pdf`.

3. Run:
```
python resume_scanner.py
```

4. Output in `output/extracted_data.json`.

## 📦 Built with
- PyPDF2
- spaCy
- regex

