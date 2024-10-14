# Credit Card Fraud Detection System

**Project Overview**:  
This project aims to detect fraudulent credit card transactions using machine learning algorithms. The dataset, sourced from Kaggle, contains transaction data labeled as fraudulent or non-fraudulent. The goal is to build a model that can accurately classify transactions as fraudulent or not, despite the significant class imbalance (very few fraud cases compared to legitimate transactions).

**Key Features**:  
- **Dataset**: `creditcard.csv` from the Kaggle `mlg-ulb/creditcardfraud` dataset, containing credit card transaction details and fraud labels.
- **Algorithms Used**: Random Forest, AdaBoost, CatBoost, SVM, LightGBM, and XGBoost.
- **Evaluation Metrics**: The primary metric used is the Gini coefficient, with cross-validation using K-Folds to ensure robustness.
  
**Tools & Libraries**:
- **Data Handling**: Pandas, NumPy.
- **Visualization**: Matplotlib, Seaborn, Plotly.
- **Machine Learning**: Scikit-learn (RandomForestClassifier, AdaBoostClassifier, SVM), LightGBM, CatBoost, XGBoost.
  
**Steps**:

1. **Data Preparation**:
    - **Download & Load Dataset**: The dataset is downloaded from Kaggle, unzipped, and read using Pandas.
    - **Initial Exploration**: Quick examination of data dimensions and summary statistics using `data_df.head()` and `data_df.describe()`.
    - **Handling Missing Data**: No missing values were found in the dataset.

2. **Data Imbalance**:  
    The dataset is heavily imbalanced, with a much larger number of legitimate transactions (Class 0) compared to fraudulent ones (Class 1). A bar chart is plotted to visualize the imbalance.

3. **Exploratory Data Analysis (EDA)**:
    - **Time and Amount Analysis**: Visualization of transaction time distribution and transaction amounts for both fraud and non-fraud classes using density plots and line plots.
    - **Transaction Statistics by Hour**: Transactions are analyzed by the hour, calculating minimum, maximum, sum, mean, and median transaction amounts for fraud and non-fraud transactions.
    - **Boxplots**: Boxplots are used to compare the amount distributions between fraudulent and non-fraudulent transactions.

4. **Correlation Analysis**:  
   A heatmap is generated to visualize the correlation between features in the dataset, helping identify relationships between features that could be important for classification.

5. **Machine Learning Models**:
    - **Random Forest**: Configured with 100 estimators, Gini metric, and 4 parallel jobs.
    - **AdaBoost**: Boosting approach to improve performance.
    - **CatBoost**: Handling categorical variables efficiently.
    - **LightGBM & XGBoost**: Gradient boosting algorithms optimized for large datasets.
    - **SVM**: Support vector machine classifier.

6. **Model Training & Validation**:
    - **Train-Validation Split**: The dataset is split into training, validation, and test sets using an 80/20 split.
    - **Cross-Validation**: 5-fold cross-validation is performed to ensure the model generalizes well to unseen data.
    - **Evaluation**: ROC-AUC score is the key performance metric used to evaluate models.

7. **Visualization**:
    - Scatter plots, line plots, density plots, and correlation heatmaps are extensively used to understand the relationships between features and their impact on fraud detection.

**Conclusion**:  
This project builds a robust system for detecting fraudulent credit card transactions using a variety of machine learning algorithms. Extensive data exploration, handling of class imbalance, and model validation techniques are used to ensure accuracy and reliability of fraud detection models.
