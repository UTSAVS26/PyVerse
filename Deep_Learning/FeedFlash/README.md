# FeedFlash – Scheduled News Summarizer (Part 1 & Latest Updates)

## AIM

To build an automated news summarization tool that extracts and condenses real news articles into short, high-quality summaries using a transformer-based model, with robust scheduling, free-tier compatibility, and a modern web interface for daily news browsing.

---

## DATASET LINK

[Kaggle News Summary Dataset](https://www.kaggle.com/datasets/sbhatti/news-summarization)

---

## NOTEBOOK & DEMO LINKS

- [Kaggle Notebook](https://www.kaggle.com/code/arihantbhandari04/feedflash-model)
- [HuggingFace Space](https://huggingface.co/spaces/Arihant-Bhandari/FeedFlash)

---

## DESCRIPTION

FeedFlash addresses information overload by creating a concise, readable digest of important daily news stories. Designed for readers, students, and professionals, it keeps you informed without the need to read entire articles.

- **Headline Fetching:** Uses NewsAPI to pull 30 real-world articles per session (now upgraded to 30 total: 20 international, 10 Indian, no genre restriction).
- **Content Extraction:** Employs `newspaper3k` for robust article scraping and cleaning.
- **Summarization:** Utilizes a PEFT/LoRA-finetuned, 4-bit quantized and dequantized Flan-T5-Base model, supporting up to 600 input tokens and 300 output tokens for richer, context-aware summaries (~90 words).
- **Storage:** Summaries and logs are saved in JSON files (`summaries.json`, `fetch_log.json`) for easy access and integration.
- **User Interface:** Gradio web interface lets users browse headlines, read summaries, and access original articles, with auto-scroll to summary and robust error handling.
- **Automation:** GitHub Actions scheduler fetches and summarizes news every 4 hours (10 AM–10 PM), syncing results to the Hugging Face Space.
- **Free-Tier Compatibility:** All scripts and workflows are designed to run on free cloud and notebook resources.

---

## EXPLANATION

### Features

- **Automated News Fetching:** Scheduled collection of 30 articles (20 international, 10 Indian) using NewsAPI.
- **Full Article Extraction:** Cleans and parses full article text with `newspaper3k`.
- **Transformer-Based Summarization:** Summaries generated with a Flan-T5-Base model, fine-tuned via PEFT/LoRA, quantized to 4-bit for efficient training, then dequantized for fast CPU/ONNX inference.
- **JSON Storage:** Outputs are saved in `summaries.json` and `fetch_log.json` for persistence and downstream use.
- **Modern Gradio UI:** Interactive, mobile-friendly interface for headline selection, summary viewing, and link access, with auto-scroll and robust error handling.
- **Scheduled Automation:** GitHub Actions workflow runs every 4 hours, updating summaries and logs, and syncing to Hugging Face Space.
- **Error Handling:** Graceful handling of missing or malformed data, API errors, and extraction/summarization failures.

### Implementation Steps

1. Selected and explored a relevant news summarization dataset.
2. Fine-tuned a Flan-T5-Base model (PEFT/LoRA, 4-bit quantized, dequantized for inference) on Kaggle.
3. Scripted automated article fetching with NewsAPI and robust content extraction with `newspaper3k`.
4. Implemented summarization logic and error handling.
5. Saved all outputs in JSON format.
6. Built a Gradio UI for user-friendly access to summaries, with mobile enhancements.
7. Set up a GitHub Actions scheduler for regular updates and automatic syncing to Hugging Face Spaces.

---

## LIBRARIES USED

- `requests`
- `newspaper3k`
- `transformers`
- `gradio`
- `json`
- `huggingface-hub`
- `newsapi-python`
- `torch`
- `sentencepiece`
- `pandas`
- `lxml`

---

## SCREENSHOTS

*Add your own screenshots here:*

- ![Project Folder Structure](https://github.com/Arihant-Bhandari/PyVerse/blob/FeedFlash-2.0/Deep_Learning/FeedFlash/images/directory-structure.png)

- ![Website](https://github.com/Arihant-Bhandari/PyVerse/blob/FeedFlash-2.0/Deep_Learning/FeedFlash/images/hf-space.png)

- ![Summary 1](https://github.com/Arihant-Bhandari/PyVerse/blob/FeedFlash-2.0/Deep_Learning/FeedFlash/images/summary-1.png)

- ![Summary 2](https://github.com/Arihant-Bhandari/PyVerse/blob/FeedFlash-2.0/Deep_Learning/FeedFlash/images/summary-2.png)

- ![Summary 3](https://github.com/Arihant-Bhandari/PyVerse/blob/FeedFlash-2.0/Deep_Learning/FeedFlash/images/summary-3.png)

---

## MODEL PERFORMANCE

| Model                        | Loss  | Training Steps | Notes                                                   |
|------------------------------|-------|---------------|---------------------------------------------------------|
| Flan-T5-Base (LoRA, 4-bit)   | 1.95  | 14,517        | Quantized 4-bit, dequantized for CPU, 600→300 tokens    |

---

## CONCLUSION

### Key Learnings

- Practical experience with transformer-based summarization and quantization.
- Integration of external APIs and web scraping for real-world data.
- Construction of an end-to-end ML pipeline with a user-facing UI and scheduled automation.

### Use Cases

- Daily news summarization for students, readers, and professionals.
- Compact news feeds for low-data or accessibility-focused environments.

### Real-World Integration

- Deploy the Gradio app on Hugging Face Spaces or a Flask server.
- Schedule the fetch-and-summarize script to run throughout the day for fresh content.
- Optionally add persistent storage and user feedback logging.

---

## FEATURES PLANNED BUT NOT YET IMPLEMENTED

- User feedback logging (thumbs-up/thumbs-down)
- Multi-language or regional news support

---

## ABOUT THE AUTHOR

**Arihant Bhandari**

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arihant-bhandari/)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Arihant-Bhandari)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Arihant-Bhandari)