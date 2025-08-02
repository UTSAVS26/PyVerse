# PDF Intelligence Extractor

📄 An intelligent document parser that extracts structured data from unstructured PDFs like resumes, invoices, reports, and research papers — with support for tables, totals, metadata, and key sections.

## 🚀 Project Overview

PDF Intelligence Extractor is a Python-based tool designed to analyze and extract key information from a wide variety of PDF files using a hybrid of rule-based (regex) and AI-assisted techniques. It supports deep parsing of layouts, tables, and sections, making it ideal for automating workflows involving:

- Resume parsing
- Invoice analytics
- Research paper summarization
- Report metadata extraction

The tool provides clean structured outputs in JSON or CSV formats, ready for downstream automation or database ingestion.

## 🧠 Key Features

### 🗂️ PDF Layout Awareness
- Parse text, headers, footers, and sections intelligently
- Detect and extract tabular data
- Read multi-column layouts

### 📊 Smart Table Extractor
- Uses pdfplumber or PyMuPDF for accurate table boundary detection
- Handles rotated pages, multi-page tables

### 🤖 AI + Regex Hybrid Extraction
- Regex patterns for known formats (e.g., invoices, phone numbers)
- Optional AI (transformer-based) to label sections semantically (e.g., "Education", "Experience", "Summary")

### 💾 Output Formats
- JSON for structured key-value mappings
- CSV for table data
- Plain text or Markdown for summaries

### 📌 Pluggable Parsers
- Easily extend support for new formats (academic papers, receipts, contracts)

## 🛠️ Tech Stack

| Purpose | Library |
|---------|---------|
| PDF parsing | pdfplumber, PyMuPDF |
| AI/NLP extraction | transformers, spacy |
| Regex processing | re, regex, dateutil |
| Table handling | pandas |
| Exporting | json, csv, markdown |

## 📂 Project Structure

```
pdf_intelligence_extractor/
├── extractors/
│   ├── __init__.py
│   ├── base_parser.py
│   ├── invoice_parser.py
│   ├── resume_parser.py
│   └── research_parser.py
├── utils/
│   ├── __init__.py
│   ├── table_utils.py
│   ├── file_utils.py
│   └── ai_helpers.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── outputs/
│   └── sample_output.json
├── examples/
│   ├── resume.pdf
│   └── invoice.pdf
├── tests/
│   ├── __init__.py
│   ├── test_resume_parser.py
│   ├── test_invoice_parser.py
│   └── test_research_parser.py
├── README.md
└── requirements.txt
```

## 📈 Use Cases

### 📄 Resume Parsing
Extract candidate name, email, phone, education, skills

### 💰 Invoice Automation
Extract totals, dates, sender/recipient, tax fields, line items

### 🧪 Research Paper Analysis
Extract authors, affiliations, abstract, references

### 📊 Business Report Metadata
Auto-label key sections, detect anomalies in totals

## 📦 Installation

```bash
git clone https://github.com/yourname/pdf-intelligence-extractor.git
cd pdf-intelligence-extractor
pip install -r requirements.txt
```

## 🧪 Example Usage

```bash
# Parse a resume PDF and export to JSON
python cli/main.py --file resume.pdf --type resume --out parsed_resume.json

# Parse all invoices in a folder
python cli/main.py --dir ./invoices/ --type invoice --out ./outputs/

# Auto-detect document type and extract
python cli/main.py --file unknown.pdf --detect-type --out out.json
```

## 📤 Sample Output

```json
{
  "document_type": "resume",
  "name": "John Doe",
  "email": "john.doe@email.com",
  "skills": ["Python", "NLP", "Data Analysis"],
  "education": [
    {
      "degree": "B.Tech in CSE",
      "institute": "XYZ University",
      "year": "2021"
    }
  ],
  "experience": [
    {
      "company": "TechCorp",
      "role": "Data Scientist",
      "duration": "Jan 2022 - Present"
    }
  ]
}
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=pdf_intelligence_extractor
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details. 