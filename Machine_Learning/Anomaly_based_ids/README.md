# Anomaly-Based Intrusion Detection System

### 🎯 **Goal**

The primary objective of this project is to develop an anomaly-based Intrusion Detection System (IDS) that identifies deviations from normal network behavior, potentially signaling an intrusion. This system uses machine learning techniques to classify network traffic as normal or anomalous, thus improving network security by detecting unusual or malicious activities.

### 🧵 **Dataset**

The dataset used is a standard network traffic dataset containing various types of network events, with features such as:

- Duration of connection
- Protocol type (TCP, UDP, ICMP)
- Source and destination IP addresses and ports
- Number of bytes transferred
- Flag status
- Various other network-related attributes
  
You can access the dataset from a relevant network traffic source, such as the https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection

### 🧾 **Description**

In this project, we implemented an anomaly detection system for network intrusion detection. The system uses machine learning algorithms to classify network traffic and detect abnormal behavior indicative of a potential attack.

The project is divided into the following stages:

- Data Preprocessing: Cleaning and preparing the dataset for training and testing.
- Feature Engineering: Selecting and transforming relevant features for model training.
- Model Training: Using classification algorithms such as Random Forest, Support Vector Machines (SVM), and others to train the IDS.
- Evaluation: Evaluating the model's performance using metrics like accuracy, precision, recall, and F1-score.
- Anomaly Detection: Detecting anomalies in the network traffic and classifying them as potential intrusions.

### 🧮 **What I had done!**

- Data Collection:

   - Loaded and explored the dataset using pandas to understand the structure and distribution of network traffic.
   - Handled missing values, anomalies, and outliers in the dataset.

- Feature Engineering:

   - Applied feature scaling and normalization techniques to improve model performance.
   - Selected features critical for anomaly detection.

- Model Implementation:

   - Implemented and trained models such as Random Forest, Support Vector Machines (SVM), and Decision Trees.
   - Tuned hyperparameters to optimize the performance of each model.

- Anomaly Detection:

   - Used the trained models to detect anomalous traffic in real-time and classify it as normal or suspicious.

- Model Evaluation:

  - Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.
  - Generated confusion matrices and ROC curves to visualize the performance of the models.

### 🚀 **Models Implemented**

- Random Forest: Used for its robustness in handling large feature sets and non-linear data. It's an ensemble learning method for classification that operates by constructing a multitude of decision trees.
- Support Vector Machine (SVM): Chosen for its ability to handle high-dimensional spaces effectively, particularly useful for detecting anomalies in complex datasets.

#### Why These Models?
- Random Forest: Offers high accuracy and reduces the risk of overfitting. Its ensemble approach enhances its ability to detect subtle anomalies in the network traffic.
- SVM: Efficient in high-dimensional spaces, making it suitable for detecting anomalies where the normal and malicious classes are not linearly separable.

### 📚 **Libraries Needed**

- pandas – For data manipulation and analysis.
- numpy – For numerical computations.
- scikit-learn – For machine learning algorithms and model evaluation.
- matplotlib & seaborn – For data visualization.
- imbalanced-learn – For handling imbalanced datasets.

Install them using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

### 📈 **Performance of the Models based on the Accuracy Scores**

- Accuracy: The accuracy of the trained models in classifying normal and anomalous traffic.
- Precision and Recall: Used to measure the model's performance in identifying true positives and false positives.
- F1-Score: Balances precision and recall, providing a more comprehensive measure of model performance.

### 📢 **Conclusion**

This project successfully implemented an anomaly-based Intrusion Detection System using machine learning techniques. The system effectively detected anomalous network traffic, allowing for early identification of potential security breaches.

Best-fit Model: The Random Forest model provided the highest accuracy and precision, making it the most effective model for detecting anomalies in this context. The system can be further enhanced with real-time network traffic analysis for continuous monitoring and detection.

### ✒️ **Your Signature**

Sharayu Anuse
