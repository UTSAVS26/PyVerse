# Diabetes Prediction Model with Enhanced Evaluation and Explainability

# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve,
                             classification_report)
import shap

# Loading the dataset
df = pd.read_csv('kaggle_diabetes.csv')

# Data Exploration
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
    df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Visualizing the distribution of features
plt.figure(figsize=(12, 8))
for i, col in enumerate(df_copy.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.histplot(df_copy[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Splitting the data into features and target variable
X = df_copy.drop(columns='Outcome')
y = df_copy['Outcome']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Hyperparameter tuning using GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                          param_grid=param_grid, 
                          cv=5,
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters found
best_params = grid_search.best_params_
print(f"\nBest hyperparameters: {best_params}")

# Building Random Forest Model with best hyperparameters
classifier = RandomForestClassifier(**best_params, random_state=42)
classifier.fit(X_train, y_train)

# Cross-validation scores
print("\nCross-validation scores:")
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

# Evaluating on test set
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Feature Importance
feature_importance = classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# SHAP Explainability
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# Feature importance plot
plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.close()

# Sample individual prediction explanation
sample_idx = 0
plt.figure()
shap.force_plot(explainer.expected_value[1], 
                shap_values[1][sample_idx,:], 
                X_test.iloc[sample_idx,:], 
                matplotlib=True, 
                show=False)
plt.tight_layout()
plt.savefig('shap_force_plot.png')
plt.close()

# Saving the model and metrics
print("\nSaving model and artifacts...")
model_artifacts = {
    'model': classifier,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc_auc': roc_auc,
    'best_params': best_params,
    'feature_names': list(X.columns)
}

with open('diabetes_prediction.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\nAll artifacts saved successfully!")
print("Model training and evaluation complete.")