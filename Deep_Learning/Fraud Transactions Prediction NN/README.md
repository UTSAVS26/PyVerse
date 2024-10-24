# Deep Learning

This project aims to develop a robust machine learning model to detect fraudulent credit card transactions. The model leverages various techniques, including data preprocessing, feature scaling, resampling methods, and advanced machine learning algorithms such as Neural Networks, Random Forests, and XGBoost. The goal is to accurately identify fraudulent transactions while minimizing false positives.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `matplotlib`: Data visualization
  - `seaborn`: Statistical data visualization
  - `scikit-learn`: Machine learning algorithms and tools
  - `tensorflow` and `keras`: Deep learning framework
  - `imblearn`: Handling imbalanced datasets
  - `shap`: Model interpretation using SHAP values
  - `missingno`: Visualizing missing data

## Dataset Description
The dataset used in this project is sourced from Kaggle and contains transactions made by credit card holders between September 2013 and October 2014. It consists of **284,807 transactions**, out of which only **492 transactions** are marked as fraudulent (0.172%). The dataset includes the following features:
- `Time`: Time elapsed since the first transaction
- `V1` to `V28`: Anonymized features resulting from PCA transformation
- `Amount`: Transaction amount
- `Class`: Target variable (1 for fraud, 0 for non-fraud)

## Installation Instructions
To set up the project environment, follow these steps:

1. **Fork the repository**:
    - Click the "Fork" button at the top right of this repository.


2. **Deep Learning dir**:
    - Go to the main Deep Learning folder
    ```
     cd Deep Learning
    ```

3. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Exploration and Preprocessing**:
    - Load the dataset and perform exploratory data analysis (EDA) to understand the data distribution and identify any missing values or outliers.
    - Preprocess the data by handling missing values and scaling features.

2. **Model Training**:
    - Train various machine learning models, including Logistic Regression, Random Forest, and Neural Networks.
    - Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

3. **Model Evaluation**:
    - Evaluate the models using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.

4. **Model Interpretation**:
    - Use SHAP values to interpret the model's predictions and understand the impact of different features on the output.

5. **Run the Jupyter Notebooks**:
    - Open the Jupyter notebooks in the `notebooks` directory to explore the code and results interactively.