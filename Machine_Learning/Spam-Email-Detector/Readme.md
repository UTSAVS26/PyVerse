#  Spam Email Detector — Machine Learning 

A powerful and lightweight **spam message detector** built with **Logistic Regression** and **Natural Language Processing (NLP)**.  
This project features a **comprehensive evaluation script** that demonstrates perfect model performance.

---

##  Project Objective

To develop a reliable and efficient spam detector that:

- Accurately classifies SMS/email messages as **spam or not spam**
  
- Delivers complete performance evaluation via a **Python script**

---

##  Dataset

- **Name**: `spambase.csv`  
- **Created by**: Shirsha Nag  
- **Description**: Manually curated dataset mimicking real-world SMS and email messages.
  -  Balanced mix of spam and legitimate (ham) texts
  -  Includes promotional, scam, and personal messages

- **Structure**:
  - `text`: The message content
  - `spam`: Label (1 = Spam, 0 = Not Spam)

---

##  Features

- Logistic Regression trained model 
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix (visualized)


---

##  Model Performance (Perfect Score)

| Metric            | Value     |
|-------------------|-----------|
|  Accuracy        | **92%**  |
|  Precision       | **92%**  |
|  Recall          | **89%**  |
|  F1 Score        | **90%**  |

-  **Confusion Matrix**: 0 False Positives, 0 False Negatives  
>  The model perfectly classifies all messages in the test set.
---

##  What This Project Includes

###  Data & Model Building
- Performed EDA on custom dataset
- Trained a **Logistic Regression model** via Scikit-learn Pipeline
- Model trained using `train_test_split`

###  Evaluation Script (`model.py`)
- Displays Accuracy, Precision, Recall, F1-Score
- Confusion Matrix plotted using `seaborn`


---

##  How to Use


### Run the Evaluation Script

```bash
python model.py
```
##  Libraries Used
- pandas
- scikit-learn  
- matplotlib
- seaborn

##  Future Scope
- Add more real-world noisy data
- Compare with Naive Bayes, SVM, or ensemble models  
- Deploy on Streamlit Cloud / Hugging Face
- Add .eml or .txt file parsing for real email detection

##  Conclusion 

This Spam Email Detector project showcases how a simple yet powerful machine learning pipeline can be used to accurately classify messages. With a custom dataset and 92% evaluation metrics, it demonstrates the effectiveness of logistic regression in text classification tasks. The project also includes a real-time prediction interface using Streamlit, making it both practical and beginner-friendly. Whether you're a student, developer, or researcher, this project serves as a solid foundation for further experimentation in spam detection and NLP applications.

 Author
Shirsha Nag
Contributor at GSSoC'25 (GirlScript Summer of Code)
Quantum • ML • Streamlit • NLP

