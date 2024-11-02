# Fake News Detection

This project is a machine learning model that detects fake news articles based on their content. It uses Natural Language Processing (NLP) and text classification techniques to classify news as either real or fake.

## Dataset
The dataset used is the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. 

The dataset used consists of two CSV files:
- `Fake.csv` contains fake news articles.
- `True.csv` contains true news articles.

Both files should be placed in the `dataset` directory and include the following columns:
- **title**: The title of the news article.
- **text**: The content of the news article.
- **subject**: The subject/category of the news article.
- **date**: The publication date of the news article.

## Features

- **TF-IDF Vectorization**: Converts text data to a format suitable for machine learning.
- **Naive Bayes Classification**: A probabilistic classifier ideal for text-based tasks.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn
