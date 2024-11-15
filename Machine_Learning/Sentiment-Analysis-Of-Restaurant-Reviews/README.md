# Sentiment Analysis of Restaurant Reviews

## Overview

This project focuses on sentiment analysis using machine learning techniques. The goal is to classify the sentiment of textual data into positive, negative, or neutral categories. By leveraging various machine learning algorithms, this project aims to provide insights into customer opinions expressed in restaurant reviews.

## Technologies Used

- **Python**: The primary programming language used for data manipulation, model building, and evaluation.
- **NLTK (Natural Language Toolkit)**: A powerful library for working with human language data (text). It provides tools for text processing, including tokenization, stemming, and removing stopwords.
- **Scikit-learn**: A robust library for machine learning in Python, which provides tools for model training, evaluation, and feature extraction.
- **Jupyter Notebook**: An interactive environment for developing and presenting data science projects. It allows for easy experimentation and visualization of results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Tusharb331/Sentiment-Analysis-Of-Restaurant-Reviews.git
   cd Sentiment-Analysis-Of-Restaurant-Reviews

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the Jupyter notebook Restaurant_Review_Sentiment_Analysis.ipynb to see the step-by-step analysis, preprocessing of data, model training, and evaluation.

## Description

### Data Cleaning

To ensure the accuracy and effectiveness of sentiment analysis, the following preprocessing steps are applied to the reviews:

1. **Removing Special Characters**: Eliminating punctuation and symbols that do not contribute to the sentiment.
2. **Converting to Lowercase**: Standardizing the text to lowercase to ensure uniformity during analysis.
3. **Removing Stopwords**: Filtering out common words (e.g., "and", "the", "is") that do not carry significant meaning in sentiment analysis.
4. **Stemming**: Reducing words to their base or root form (e.g., "running" to "run") to consolidate similar meanings.

### Feature Extraction

The **Bag of Words** model is utilized for feature extraction, transforming text data into numerical feature vectors using **CountVectorizer**. This technique creates a matrix where each row represents a review, and each column corresponds to a unique word from the corpus. The values in the matrix represent the frequency of each word in the respective review.

### Model Training

Three different machine learning models are trained to classify the sentiment of the reviews:

1. **Multinomial Naive Bayes**: A probabilistic model that assumes the presence of a particular feature (word) in a class (sentiment) is independent of the presence of any other feature.
2. **Bernoulli Naive Bayes**: Similar to Multinomial Naive Bayes, but it assumes binary features (words present or not).
3. **Logistic Regression**: A regression model used for binary classification that predicts the probability of a binary outcome based on one or more predictor variables.

### Model Evaluation

Each model is evaluated on various performance metrics to assess its effectiveness:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the all actual positives.

### Results

The performance of each model is summarized below:

- **Multinomial Naive Bayes**: 
  - Accuracy: 76.5%
  - Precision: 0.78
  - Recall: 0.78
- **Bernoulli Naive Bayes**: 
  - Accuracy: 76.5%
  - Precision: 0.79
  - Recall: 0.76
- **Logistic Regression**: 
  - Accuracy: 75.0%
  - Precision: 0.82
  - Recall: 0.68

### Prediction

You can predict the sentiment (positive or negative) of your review messages using the trained models. 

### Example

Here's a simple example of how to use the sentiment analysis functionality in your code:

```python
from sentiment_analysis import predict_review

msg = 'The food is really good here.'
prediction = predict_review(msg)
print(f"Sentiment: {'Positive' if prediction else 'Negative'} Review")

