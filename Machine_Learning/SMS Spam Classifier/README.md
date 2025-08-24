# 📩 SMS Spam Classifier using Naive Bayes


This project builds a **text classification model** to detect whether an SMS message is **spam** or **ham (not spam)** using **Naive Bayes** and **TF-IDF vectorization**. It is a classic NLP use case and demonstrates how probabilistic classifiers can be used to flag unwanted messages.

---

## 📌 Objective

To classify SMS messages as spam or ham using machine learning, and evaluate model performance through classification metrics and confusion matrix analysis.

---

## 📂 Dataset

- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Samples**: 5,572 messages
- **Target Variable**: 
  - `spam`: Unwanted messages
  - `ham`: Legitimate messages
- **Format**:
  - `label`: spam/ham
  - `message`: the text content of the SMS

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked class distribution (spam vs ham)
- Visualized frequent spam/ham words using word clouds
- Observed that spam messages tend to use promotional terms, numbers, and URLs

---

## ⚙️ Preprocessing

- Removed punctuation, stopwords, and converted text to lowercase
- Encoded labels: `ham → 0`, `spam → 1`
- Applied **TF-IDF Vectorizer** to transform messages into numerical features
- Split data into **training and test sets** with stratification

---

## 🧠 Model Used

- **Multinomial Naive Bayes**: Ideal for discrete features like word counts or TF-IDF scores
- Trained using scikit-learn’s `MultinomialNB`

---

## 📈 Model Evaluation

| Metric        | Ham (0) | Spam (1) | Interpretation                            |
|---------------|---------|----------|-------------------------------------------|
| Precision     | 0.95    | 1.00     | No ham misclassified as spam              |
| Recall        | 1.00    | 0.62     | 38% of spam messages were missed          |
| F1-Score      | 0.97    | 0.76     | Spam recall could be improved             |
| **Accuracy**  | –       | –        | **95.16% overall**                         |

**Confusion Matrix**:

```
[[903   0]
 [ 50  81]]
```

- ✅ True Ham: 903
- ❌ Missed Spam: 50
- ✅ Correct Spam: 81

---

## 🔍 Sample Prediction

```python
predict_message("Congratulations! You've won a free vacation! Reply NOW to claim.")
# Output: 'Spam'
```

The model takes any SMS input and returns a prediction: **Spam** or **Ham**.

---

## 💻 How to Run

```bash
# 1. Clone or download the repo
# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud

# 3. Run the Jupyter Notebook
# 4. Use predict_message() function to test your own texts
```

---

## 🚀 Future Enhancements

- Improve recall for spam class by tuning alpha or using other classifiers
- Experiment with n-grams or additional features
- Deploy with Streamlit or Flask for interactive UI
- Use deep learning (LSTM, BERT) for more advanced performance

---

## 👤 Author

- **GitHub**: [archangel2006](https://github.com/archangel2006)  
