# 🧠 FlashGenie: AI Flashcard Generator from PDFs

## 📌 Project Overview

**FlashGenie** is an intelligent flashcard generator that takes any **user-provided PDF** (e.g., textbooks, lecture notes), processes the text, and creates **concise, educational flashcards** using NLP techniques like **keyword extraction**, **summarization**, and **question generation**.

## ✅ Key Features

- Upload any text-based PDF (notes, books, etc.)
- Automatic extraction of **key concepts**
- Generates **question-answer** flashcards using NLP
- Supports multiple question types:
  - Factual ("What is...")
  - Conceptual ("Why does...")
  - Fill-in-the-blanks
- Export to **CSV**, **Anki**, or **printable PDF**

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd VoiceMoodMirror
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

### Usage

#### Web Interface (Recommended)
```bash
streamlit run ui/app.py
```

#### Command Line
```bash
python -m flashgenie.main --input path/to/document.pdf --output flashcards.csv
```

## 📂 Project Structure

```
flashgenie/
│
├── pdf_parser/
│   └── extract_text.py          # PDF to plain text using PyMuPDF / PDFMiner
│
├── preprocessing/
│   ├── clean_text.py            # Remove references, footnotes, etc.
│   └── chunker.py               # Split into sentences/sections for processing
│
├── nlp/
│   ├── keyword_extractor.py     # Uses RAKE / spaCy / KeyBERT
│   ├── summarizer.py            # (Optional) Sentence summarizer
│   ├── question_generator.py    # T5 / BART / Rule-based QG
│   └── answer_selector.py       # Identify precise answers from source text
│
├── flashcard/
│   ├── flashcard_formatter.py   # Convert Q/A pairs to card formats
│   └── export.py                # Export to Anki CSV / PDF
│
├── ui/
│   └── app.py                   # Streamlit interface
│
├── examples/
│   └── demo_notebook.ipynb
│
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_preprocessing.py
│   ├── test_nlp.py
│   └── test_flashcard.py
│
├── requirements.txt
├── README.md
└── main.py
```

## 🧠 How It Works

1. **Extract Text** → Preprocess
2. **Detect Keywords** → Context windows
3. **Generate Questions** → T5 model
4. **Detect Answers** → Extract or summarize
5. **Export Flashcards** → Use, revise, learn!

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=flashgenie
```

## 🎮 Features in Action

Upload `Biology_Chapter3.pdf` and get output like:

```
Q: What is the function of the mitochondria?
A: Powerhouse of the cell; generates ATP through respiration.

Q: Fill in the blank: DNA is composed of nucleotides containing ____, phosphate, and a sugar.
A: Bases
```

## 🧪 Enhancements & Stretch Goals

- ✍️ User rating for Q/A quality → self-improvement loop
- 🧑‍🎓 Difficulty tagging (easy, medium, hard)
- 🔤 Multilingual PDF + Flashcard support
- 🧠 Use embeddings for semantic de-duplication
- 📲 Deploy as a web/mobile app with login
- 🎧 Optional TTS readout for auditory learners

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on GitHub.
