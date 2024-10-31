import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(model_name):
    """
    Load a saved model from the trained_models directory
    """
    filename = f'trained_models/{model_name}_model.pkl'
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Successfully loaded {model_name} model")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_data(data_path):
    """
    Prepare the wine quality data for testing, ensuring feature names match training data
    """
    try:
        # Read the data
        df = pd.read_csv(data_path)
        
        # Convert type to dummy variables and keep all columns except 'quality'
        df = pd.concat([df, pd.get_dummies(df['type'])], axis=1)
        df = df.drop('type', axis=1)
        
        # Remove null values
        df = df.dropna()
        
        # Create binary quality target
        df['quality_binary'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)
        
        # Select features in the same order as during training
        feature_columns = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "red", "white"
        ]
        
        # Prepare features and target
        X = df[feature_columns]  # Only select the features used in training
        y = df['quality_binary']
        
        print("Data preparation successful")
        print(f"Features included: {', '.join(X.columns)}")
        return X, y
    
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None, None

def evaluate_model(model, X, y, model_name):
    """
    Evaluate a model's performance
    """
    try:
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, y_pred
    
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None, None

def test_all_models(data_path):
    """
    Test all saved models and compare their performance
    """
    # List of model names
    model_names = [
        'logistic_regression',
        'decision_tree',
        'random_forest',
        'knn',
        'svc',
        'gradient_boosting'
    ]
    
    # Prepare data
    X, y = prepare_data(data_path)
    if X is None or y is None:
        return
    
    # Store results
    results = []
    
    # Test each model
    for model_name in model_names:
        model = load_model(model_name)
        if model is not None:
            accuracy, _ = evaluate_model(model, X, y, model_name)
            if accuracy is not None:
                results.append({'Model': model_name, 'Accuracy': accuracy * 100})
    
    # Create comparison plot
    if results:
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', data=results_df)
        plt.xticks(rotation=45)
        plt.title('Model Comparison on Test Data')
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run the test script
    """
    print("Starting model testing...")
    
    # Specify the path to your test data
    data_path = 'winequalityN.csv'  # Update this path as needed
    
    # Test all models
    test_all_models(data_path)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()