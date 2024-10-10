# PROJECT TITLE
Red Wine Quality Prediction using Machine Learning Models

## GOAL
**Aim:** To predict the quality of red wine based on its physicochemical properties using various machine learning models.

## DATASET
**Dataset:** Red Wine Quality - UCI Machine Learning Repository

## DESCRIPTION
This project aims to predict the quality of red wine based on its physicochemical attributes. The dataset consists of several features such as alcohol, pH, sulphates, and residual sugar, which influence the quality of the wine. The target variable is the wine's quality, which is rated on a scale from 3 to 8. Various machine learning models are applied to build a prediction model for wine quality.

## WHAT I HAD DONE
### 1. Configured Kaggle API:
- Set up Kaggle API to download the dataset from the UCI repository.
- Loaded the dataset into the working environment.

### 2. Performed Exploratory Data Analysis (EDA):
- Loaded the dataset and viewed the top 5 rows.
- Extracted statistical information about the features using the `.describe()` method.
- Explored correlations between different features and the target variable (quality).
- Visualized the distribution of wine quality ratings using pie charts and the distribution of alcohol content using histograms.

### 3. Preprocessed the Data:
- Scaled the numerical features using `StandardScaler` to standardize the data.
- Handled the class imbalance in the target variable using SMOTE (Synthetic Minority Over-sampling Technique).

### 4. Defined Features and Quality Levels:
- Analyzed the dataset based on different ranges of quality (3-4, 5-6, and 7-8) to understand feature behavior in each category.

### 5. Built Multiple Machine Learning Models:
- **Models implemented:**
  - Logistic Regression (LR)
  - Naive Bayes (NB)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (DT)
  - Support Vector Machine (SVM with RBF and linear kernel)
  - Linear Discriminant Analysis (LDA)

### 6. Evaluated the Model:
- Used 5-fold Stratified Cross-Validation to evaluate model performance.
- Calculated accuracy for each model to compare their success rates.

## MODELS USED
- **Logistic Regression (LR):** A linear model that predicts categorical outcomes based on input features.
- **Naive Bayes (NB):** A probabilistic model based on Bayes’ theorem.
- **K-Nearest Neighbors (KNN):** A non-parametric model that classifies based on the majority vote of the nearest neighbors.
- **Decision Tree (DT):** A tree-based model that splits the data on feature values to predict the target variable.
- **Support Vector Machine (SVM):** A model that finds the optimal hyperplane to classify data points in high-dimensional spaces.
- **Linear Discriminant Analysis (LDA):** A linear model that reduces dimensionality while preserving as much class-discriminatory information as possible.

## LIBRARIES NEEDED
- **Numpy:** For numerical computations.
- **Pandas:** For data manipulation and preprocessing.
- **Matplotlib:** For basic data visualization.
- **Seaborn:** For advanced data visualization.
- **Scikit-Learn:** For building and evaluating machine learning models.
- **Imbalanced-learn:** For handling imbalanced datasets (SMOTE).

## INSIGHTS
Through this project, you will gain insights into:
- How to explore and preprocess data, including scaling features and handling class imbalance.
- Building and comparing the performance of various machine learning models such as Logistic Regression, Naive Bayes, K-Nearest Neighbors, Decision Tree, and Support Vector Machines.
- Using cross-validation to evaluate model performance.
- Visualizing the results and understanding the distribution of wine quality based on its features.

