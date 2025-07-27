# 📧 Spam Email Detector — Streamlit App

## 🎯 Goal

The main objective of this project is to build a lightweight yet accurate **spam email detector** using **machine learning** and **Natural Language Processing (NLP)** techniques. The project features a user-friendly **Streamlit web app** that predicts whether a given email message is **spam or not**, helping users avoid unwanted or malicious content.

---

## 🧵 Dataset

- 📂 **Dataset Name**: Synthetic SMS Spam Dataset
- 🛠️ **Created By**: Shirsha Nag (Manually curated)
- 🧪 **Description**:  
  This custom dataset was created to simulate real-world email and SMS messages. It includes a balanced mix of spam and legitimate (ham) messages to train the model effectively.

- ✉️ **Message Types Included**:
  - Promotional offers and advertisements
  - Scam/phishing messages
  - Normal personal or professional messages

- 📄 **Structure**:
  - `text`: Message content
  - `spam`: Label (1 = Spam, 0 = Not Spam)

> ✅ This self-created dataset was used to train and test the spam classification model implemented in this project.

---

## 🧾 Description

This project implements a **Logistic Regression** classifier embedded within a **Streamlit web app**. The app takes user input, classifies it as spam or not spam, and displays the confidence level of the prediction. The model uses **TF-IDF vectorization** to convert text into numerical features suitable for machine learning.

---

## 🧮 What I Had Done!

1. Designed and compiled a synthetic dataset of labeled spam and ham messages.
2. Performed data preprocessing using `TfidfVectorizer` to convert text into feature vectors.
3. Split the dataset into training and testing sets.
4. Built a `Pipeline` combining the vectorizer and a Logistic Regression model.
5. Trained the model on the training data and evaluated it on the test set.
6. Developed a clean and interactive user interface using **Streamlit**.
7. Added confidence scores and basic input validation for better UX.

---

## 🚀 Models Implemented

| Model                | Reason for Choosing                                      |
|----------------------|-----------------------------------------------------------|
| **Logistic Regression** | Efficient, interpretable, and ideal for binary classification tasks like spam detection. |

> *(Future scope includes experimenting with Naive Bayes, SVM, or ensemble methods.)*

---

## 📚 Libraries Needed

- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- *(Optional for EDA)*: `matplotlib`, `seaborn`, `nltk`, `wordcloud`

Install dependencies using:

```bash
pip install pandas numpy scikit-learn streamlit
