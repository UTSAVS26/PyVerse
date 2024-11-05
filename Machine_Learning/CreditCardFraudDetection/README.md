  # Hackathon_Project
Fraud transaction detection using machine learning
kaggle dataset file link is here:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



## Project Aims

This project aims to convey several key points:

1. **Practical Application of Machine Learning**: 
   Demonstrates a real-world application of machine learning in financial security, specifically in detecting credit card fraud.

2. **Handling Imbalanced Datasets**: 
   Showcases how to deal with imbalanced datasets, a common challenge in fraud detection. The code uses undersampling of the majority class (legitimate transactions) to balance the dataset.

3. **Basic Machine Learning Workflow**: 
   Illustrates the fundamental steps in a machine learning project:
   - Data loading and preprocessing
   - Splitting data into features and target
   - Dividing data into training and testing sets
   - Model selection, training, and evaluation

4. **Use of Popular Data Science Libraries**: 
   Demonstrates the use of common Python libraries for data science and machine learning:
   - pandas for data manipulation
   - scikit-learn for machine learning tasks
   - numpy for numerical operations

5. **Simple Model Implementation**: 
   Uses Logistic Regression, a straightforward and interpretable model, as a starting point for fraud detection.

6. **Model Evaluation**: 
   Shows how to evaluate a model's performance using accuracy scores on both training and test data.

7. **Reproducibility**: 
   Includes a link to the dataset and provides the code, emphasizing reproducibility in data science.

8. **Importance of Fraud Detection**: 
    Highlights the significance of fraud detection in the financial sector, addressing a real-world problem that affects many people and businesses.


### Data Preprocessing

1. **Data Loading**:
   - The dataset is loaded from 'creditcard.csv' using pandas:
     ```python
     data = pd.read_csv("creditcard.csv")
     ```

2. **Class Separation**:
   - Legitimate and fraudulent transactions are separated:
     ```python
     legit = data[data.Class == 0]
     fraud = data[data['Class'] == 1]
     ```
   - This separation allows for analysis of class imbalance.

3. **Feature and Target Separation**:
   - Features (X) and target variable (y) are split:
     ```python
     x = data.drop('Class', axis=1)
     y = data['Class']
     ```

4. **Handling Class Imbalance**:
   - Undersampling of the majority class (legitimate transactions) is performed:
     ```python
     legit_s = legit.sample(n=len(fraud), random_state=2)
     data = pd.concat([legit_s, fraud], axis=0)
     ```
   - This creates a balanced dataset for training.

### Model Training

1. **Train-Test Split**:
   - Data is split into training and testing sets:
     ```python
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
     ```
   - 20% of data is reserved for testing.
   - Stratification ensures that the class distribution is maintained in both sets.

2. **Model Selection**:
   - Logistic Regression is chosen as the classification algorithm:
     ```python
     model = LogisticRegression()
     ```
   - This is a good baseline model for binary classification tasks.

3. **Model Training**:
   - The model is trained on the training data:
     ```python
     model.fit(x_train, y_train)
     ```

### Model Evaluation

1. **Accuracy Calculation**:
   - The model's performance is evaluated using accuracy scores:
     ```python
     train_acc = accuracy_score(model.predict(x_train), y_train)
     test_acc = accuracy_score(model.predict(x_test), y_test)
     ```
   - Both training and testing accuracies are calculated to assess overfitting.


### Conclusion

This implementation provides a solid foundation for credit card fraud detection. The use of undersampling to balance the dataset and Logistic Regression as the classification algorithm offers a good starting point. The separate calculation of training and testing accuracies allows for basic assessment of model generalization.This project serves as an introductory example of applying machine learning to a critical financial problem, demonstrating how relatively simple techniques can be used to approach complex real-world issues. It provides a starting point for understanding and implementing fraud detection systems.


Thank You !