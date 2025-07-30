# 📄 PDFTextGenie

> **Smart, Modern, and Extensible PDF-to-Text Converter with Advanced OCR and GUI**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green?logo=qt)](https://riverbankcomputing.com/software/pyqt/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

---

## 🚀 Project Vision

**PDFTextGenie** is a next-generation, user-friendly Python application for extracting clean, accurate text from scanned or image-based PDFs. It leverages state-of-the-art OCR (Optical Character Recognition) engines and advanced post-processing to deliver high-fidelity text output, even from noisy or complex documents. Designed for researchers, students, archivists, and anyone who needs reliable text extraction, PDFTextGenie offers batch processing, smart formatting, and a modern, cross-platform GUI.

---

## ✨ Key Features

- **Batch PDF Input:** Select and queue multiple PDFs for conversion.
- **Advanced OCR:** Uses EasyOCR (with modular backend for future engines like TrOCR, Tesseract, LLaMA OCR).
- **Image Preprocessing:** Binarization, denoising, and contrast enhancement for better OCR accuracy.
- **Smart Text Cleanup:** Removes extra newlines, fixes hyphenated line breaks, merges paragraphs.
- **Modern GUI:** Built with PyQt5, featuring drag & drop, live status, output preview, and persistent settings.
- **Error Handling:** User-friendly feedback for failed conversions, with detailed error dialogs.
- **Persistent Settings:** Remembers your last-used options and output folder.
- **Extensible Architecture:** Modular codebase for easy addition of new OCR engines or output formats.
- **Offline & Cross-Platform:** No API keys or internet required. Works on Windows, Linux, and macOS.

---

## 🧩 Architecture Overview

```
pdftextgenie/
├── gui/
│   └── main_window.py          # PyQt5 GUI
├── ocr/
│   ├── llama_ocr.py            # OCR backend (EasyOCR, extensible)
│   └── postprocess.py          # Text cleanup utilities
├── utils/
│   └── pdf_utils.py            # PDF to image conversion & preprocessing
├── output/
│   └── (Generated .txt files)
├── tests/
│   └── test_pdftextgenie.py    # Automated tests
├── app.py                      # Main launcher
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/-------
cd pdftextgenie
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** You may need additional system packages for `pdf2image` (e.g., poppler). See [pdf2image docs](https://github.com/Belval/pdf2image#installing-poppler-on-windows) for details.

---

## 🏁 Quick Start

```bash
python app.py
```

- Use the GUI to select PDFs, set options, and start conversion.
- Extracted text files will appear in your chosen output folder.

---

## 🖥️ GUI Usage Guide

- **Add PDFs:** Click 'Add PDFs' or drag & drop files into the list.
- **Remove/Clear:** Remove selected or clear all PDFs from the list.
- **Choose Output Folder:** Set where extracted text files will be saved.
- **Options:**
  - **Fix hyphens:** Reconstruct words split across lines.
  - **Merge paragraphs:** Combine lines into paragraphs.
  - **DPI:** Set image resolution for PDF-to-image conversion.
- **Image Preprocessing:**
  - **Binarize:** Convert images to black & white for better OCR.
  - **Denoise:** Apply median filter to reduce noise.
  - **Enhance Contrast:** Boost contrast (set factor).
- **Start Conversion:** Begin batch processing. Progress and errors are shown live.
- **Preview:** See a snippet of the extracted text and any errors.
- **Help/About:** Access app info from the menu bar.

---

## 🧠 Advanced Options & Extensibility

- **OCR Backend:**
  - Currently uses EasyOCR. The code is modular—add new engines (Tesseract, TrOCR, etc.) in `ocr/llama_ocr.py`.
- **Image Preprocessing:**
  - Easily extend `utils/pdf_utils.py` to add more filters or preprocessing steps.
- **Output Formats:**
  - Add DOCX, HTML, or other formats by extending the output logic in the GUI and backend.
- **Settings:**
  - Persistent via `QSettings`. Add more options as needed.

---

## 🧪 Testing

- Automated tests are provided in `tests/test_pdftextgenie.py`.
- Run all tests:
  ```bash
  python -m pytest tests/
  ```
- Tests cover:
  - Image preprocessing (binarize, denoise, contrast)
  - Text cleanup (hyphens, paragraphs)
  - OCR backend interface (mocked)
  - Error handling for invalid files

---

## 🛠️ Troubleshooting

- **pdf2image errors:**
  - Ensure [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) is installed and on your PATH (Windows).
- **OCR accuracy issues:**
  - Try enabling preprocessing options or increasing DPI.
- **GUI not launching:**
  - Check Python and PyQt5 installation.
- **Other issues:**
  - Run tests and check error dialogs for details.

---

## 🤝 Contributing

Contributions are welcome! To propose a feature, fix a bug, or add a new OCR backend:

1. Fork the repo and create a new branch.
2. Add your changes and tests.
3. Open a pull request with a clear description.

**Ideas for contribution:**

- Add new OCR engines (Tesseract, TrOCR, LLaMA OCR)
- Multilingual support
- Export to DOCX/HTML
- Table/column structure preservation
- GUI enhancements and accessibility

---

## 👤 Author

- **Author:** [@SK8-infi](https://github.com/SK8-infi)

---
