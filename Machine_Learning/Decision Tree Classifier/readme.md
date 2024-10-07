
# PROJECT TITLE

Customer Purchase Prediction using Decision Tree Classifier

## GOAL

**Aim**: To predict whether a customer will purchase a product or service based on demographic and behavioral data using a Decision Tree Classifier.

## DATASET

[Bank Marketing Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## DESCRIPTION

This is a classification problem where we need to predict if a customer will make a purchase. The dataset consists of demographic and behavioral features of customers, which are used to build a Decision Tree Classifier model.

## WHAT I HAD DONE

1. Performed **exploratory data analysis (EDA)** on the dataset.
    - Loaded the dataset and viewed the top 5 rows.
    - Calculated statistical data for features in the dataset.
    - Analyzed the correlation between the features.
    - Performed **data visualization** using libraries like `matplotlib` and `seaborn`.
  
2. Preprocessed the data:
    - Handled missing values.
    - Encoded categorical variables into numerical formats.
    - Split the data into training and testing sets.
    
3. Built the **Decision Tree Classifier** model:
    - Used **Gini** and **Entropy** as criteria to create two different decision tree models.
    - Optimized the model by tuning hyperparameters such as `max_depth` and `min_samples_split`.

4. Evaluated the model using the following metrics:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**

5. Compared the performance of the two Decision Tree models (one using Gini and the other using Entropy criteria) based on evaluation metrics.

## MODELS USED

- **Decision Tree Classifier (Gini criterion)**: A decision tree built using Gini impurity as the split criterion.
- **Decision Tree Classifier (Entropy criterion)**: A decision tree built using information gain as the split criterion.

## LIBRARIES NEEDED

- **Numpy**: For numerical computations.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For data visualization.
- **Seaborn**: For advanced data visualization.
- **Scikit-Learn**: For building and evaluating the machine learning models.

## INSIGHTS
By this project, you will gain insights into:

- How to preprocess data, handle missing values, and encode categorical variables.
- Building and tuning Decision Tree models using different criteria (Gini and Entropy).
- Evaluating classification models with accuracy, precision, recall, and F1-score.
- Visualizing the results using data visualization libraries.