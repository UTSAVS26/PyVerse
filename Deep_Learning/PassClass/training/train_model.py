"""
Model Training Script

This script trains the password strength classifier and saves the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tfidf_classifier import TFIDFPasswordClassifier
from utils.metrics import plot_confusion_matrix, plot_classification_report


def load_dataset(filepath: str = 'data/password_dataset.csv') -> pd.DataFrame:
    """
    Load the password dataset.
    
    Args:
        filepath: Path to the dataset CSV file
        
    Returns:
        DataFrame with password data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with {len(df)} samples")
        return df
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}")
        print("Please run data/generate_passwords.py first to generate the dataset.")
        return None


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for training by using the actual labels from the labeler.
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        DataFrame ready for training
    """
    # Use actual_label for training (the labeler's assessment)
    training_df = df.copy()
    training_df['label'] = training_df['actual_label']
    
    # Remove rows where target and actual labels don't match (optional)
    # This ensures we train on consistent data
    consistent_df = training_df[training_df['label_match'] == True]
    
    print(f"Using {len(consistent_df)} consistent samples for training")
    print(f"Label distribution:")
    print(consistent_df['label'].value_counts())
    
    return consistent_df


def train_multiple_models(df: pd.DataFrame) -> dict:
    """
    Train multiple models and compare their performance.
    
    Args:
        df: Training dataset
        
    Returns:
        Dictionary with trained models and their results
    """
    models = {}
    results = {}
    
    model_types = ['random_forest', 'logistic_regression', 'svm']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*50}")
        
        # Create and train classifier
        classifier = TFIDFPasswordClassifier(model_type=model_type)
        training_results = classifier.train(df)
        
        models[model_type] = classifier
        results[model_type] = training_results
        
        # Save the model
        model_path = f'models/{model_type}_password_classifier.pkl'
        classifier.save_model(model_path)
        
        print(f"Model saved to: {model_path}")
    
    return models, results


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare performance of different models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model_type, result in results.items():
        comparison_data.append({
            'Model': model_type.replace('_', ' ').title(),
            'Accuracy': result['accuracy'],
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = 'training/model_comparison.png'):
    """
    Plot model comparison results.
    
    Args:
        comparison_df: DataFrame with model comparison data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(comparison_df['Model'], comparison_df['Accuracy'], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Cross-validation comparison
    bars2 = ax2.bar(comparison_df['Model'], comparison_df['CV Mean'], 
                    yerr=comparison_df['CV Std'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Cross-Validation Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('CV Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Model comparison plot saved to: {save_path}")


def evaluate_best_model(models: dict, results: dict, df: pd.DataFrame):
    """
    Evaluate the best performing model in detail.
    
    Args:
        models: Dictionary of trained models
        results: Dictionary of model results
        df: Training dataset
    """
    # Find best model
    best_model_type = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_type]
    best_results = results[best_model_type]
    
    print(f"\n{'='*50}")
    print(f"EVALUATING BEST MODEL: {best_model_type.upper()}")
    print(f"{'='*50}")
    
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"Cross-validation score: {best_results['cv_mean']:.4f} (+/- {best_results['cv_std'] * 2:.4f})")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(best_results['y_test'], best_results['y_pred']))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        best_results['confusion_matrix'],
        classes=['weak', 'medium', 'strong'],
        title=f'Confusion Matrix - {best_model_type.title()}',
        save_path=f'training/{best_model_type}_confusion_matrix.png'
    )
    
    # Test on some examples
    test_examples = [
        "abc",           # weak
        "password123",   # weak
        "Hello2023!",    # medium
        "G7^s9L!zB1m",  # strong
        "qwerty",       # weak
        "MyPass@123",   # medium
        "tR#8$!XmPq@",  # strong
        "123456789",    # weak
        "SecurePass1!", # medium
        "K9#mN2$pL7@",  # strong
    ]
    
    print(f"\nTesting on example passwords:")
    print("-" * 50)
    
    correct_predictions = 0
    for password in test_examples:
        result = best_model.predict_with_confidence(password)
        is_correct = result['prediction_match']
        if is_correct:
            correct_predictions += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {password:15} -> {result['predicted_label']:8} "
              f"(confidence: {result['confidence']:.3f})")
    
    print(f"\nExample accuracy: {correct_predictions}/{len(test_examples)} ({correct_predictions/len(test_examples):.1%})")
    
    return best_model, best_model_type


def main():
    """Main training function."""
    print("🔐 PassClass Model Training")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Prepare training data
    training_df = prepare_training_data(df)
    
    if len(training_df) == 0:
        print("No consistent data found for training.")
        return
    
    # Train multiple models
    models, results = train_multiple_models(training_df)
    
    # Compare models
    comparison_df = compare_models(results)
    print(f"\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plot_model_comparison(comparison_df)
    
    # Evaluate best model
    best_model, best_model_type = evaluate_best_model(models, results, training_df)
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED!")
    print(f"Best model: {best_model_type}")
    print(f"Best accuracy: {results[best_model_type]['accuracy']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 