# FeedFlash – Scheduled News Summarizer (Part 1)

## AIM

To build a manual news summarization tool that extracts and condenses real news articles into short summaries using a transformer-based model.

## DATASET LINK

[Kaggle News Summary Dataset](https://www.kaggle.com/datasets/sbhatti/news-summarization)

## MY NOTEBOOK LINK

[Kaggle Notebook](https://www.kaggle.com/code/arihantbhandari04/feedflash-model)

[HuggingFace Space](https://huggingface.co/spaces/Arihant-Bhandari/FeedFlash)

## DESCRIPTION

The FeedFlash project aims to solve the problem of overwhelming news content by creating a concise, readable digest of important daily stories. It is necessary for readers, students, and professionals who want to stay informed without reading entire articles.

The system uses NewsAPI to fetch headlines and `newspaper3k` to extract full article content. A transformer model (e.g., `T5-base`) is used to generate \~60-word summaries.

I began by exploring news summarization datasets, then proceeded to fine-tune a summarization model on Kaggle using T5. I built modular scripts for fetching articles, generating summaries, and serving them via a Gradio UI.

**Additional Resources Used:**

* Hugging Face Transformers documentation
* Gradio documentation
* Blogs on news summarization
* Papers: BART (Lewis et al., 2019), T5 (Raffel et al., 2020)

## EXPLANATION

### DETAILS OF THE DIFFERENT FEATURES

* **Article Fetching**: Uses NewsAPI to pull 30 real-world articles.
* **Content Extraction**: `newspaper3k` scrapes and cleans article text.
* **Summarization**: Transformer model generates short summaries.
* **Storage**: Results are saved to `summaries.json`.
* **UI**: Gradio interface to view headlines, summaries, and links.

### WHAT I HAVE DONE

1. Selected a summarization dataset.
2. Fine-tuned a transformer summarization model.
3. Created a script to fetch real articles using NewsAPI.
4. Built a text extraction module using `newspaper3k`.
5. Wrote summarization logic using Hugging Face pipelines.
6. Stored all outputs in JSON format.
7. Developed a Gradio UI to view results.

### PROJECT TRADE-OFFS AND SOLUTIONS

1. **Trade-off 1**: Accuracy vs. speed of summarization.

   * **Solution**: Used `T5-base` for balance between model size and performance.
2. **Trade-off 2**: Article variety vs. API rate limits.

   * **Solution**: Limited pulls to 30 articles per session, reduced API usage.

### LIBRARIES NEEDED

* requests
* newspaper3k
* transformers
* gradio
* json

### SCREENSHOTS

*To be attached:*

* Project folder structure
* Gradio UI screenshot
* Summaries JSON preview

### MODELS USED AND THEIR ACCURACIES

| Model                  | Loss | 
| ---------------------- | ---------------- | 
| T5-base (fine-tuned) | 1.59   |

## CONCLUSION

### WHAT YOU HAVE LEARNED

* Gained hands-on experience with transformer summarizers
* Learned API integration and article scraping techniques
* Built an end-to-end ML pipeline with UI

### USE CASES OF THIS MODEL

1. Daily news summarization for students and readers.
2. Compact content feeds for low-data or accessibility-focused platforms.

### HOW TO INTEGRATE THIS MODEL IN REAL WORLD

1. Deploy Gradio app via Hugging Face Spaces or Flask.
2. Automate the fetch-and-summarize script to run daily.
3. Optional: add storage + user feedback with database or logging.

### FEATURES PLANNED BUT NOT IMPLEMENTED

* Automatic scheduled runs using `apscheduler`
* Daily reset of summary and log data
* Thumbs-up/thumbs-down user feedback logging

### YOUR NAME

*Arihant Bhandari*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/arihant-bhandari/)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Arihant-Bhandari)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Arihant-Bhandari)