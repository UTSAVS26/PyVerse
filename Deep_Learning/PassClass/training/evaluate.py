"""
Model Evaluation Script

This script evaluates the trained password strength classifier with detailed metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import pickle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tfidf_classifier import TFIDFPasswordClassifier
from utils.metrics import plot_confusion_matrix, plot_classification_report


def load_trained_model(model_path: str) -> TFIDFPasswordClassifier:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded classifier
    """
    try:
        classifier = TFIDFPasswordClassifier()
        classifier.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return classifier
    except FileNotFoundError:
        print(f"Model not found at: {model_path}")
        return None


def load_test_data(filepath: str = 'data/password_dataset.csv') -> pd.DataFrame:
    """
    Load test data for evaluation.
    
    Args:
        filepath: Path to the dataset
        
    Returns:
        DataFrame with test data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded test data with {len(df)} samples")
        return df
    except FileNotFoundError:
        print(f"Test data not found at: {filepath}")
        return None


def evaluate_model_performance(classifier: TFIDFPasswordClassifier, test_passwords: list, test_labels: list) -> dict:
    """
    Evaluate model performance on test data.
    
    Args:
        classifier: Trained classifier
        test_passwords: List of test passwords
        test_labels: List of true labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    predictions = []
    probabilities = []
    
    for password in test_passwords:
        pred = classifier.predict(password)
        proba = classifier.predict_proba(password)
        predictions.append(pred)
        probabilities.append(proba)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    
    # Classification report
    report = classification_report(test_labels, predictions, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, average=None, labels=['weak', 'medium', 'strong']
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, label in enumerate(['weak', 'medium', 'strong']):
        class_metrics[label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Calculate ROC AUC for each class (one-vs-rest)
    roc_auc_scores = {}
    for i, label in enumerate(['weak', 'medium', 'strong']):
        # Convert to one-vs-rest format
        y_true_binary = [1 if l == label else 0 for l in test_labels]
        y_score_binary = [proba[label] for proba in probabilities]
        
        try:
            auc = roc_auc_score(y_true_binary, y_score_binary)
            roc_auc_scores[label] = auc
        except ValueError:
            roc_auc_scores[label] = 0.0
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'roc_auc_scores': roc_auc_scores,
        'predictions': predictions,
        'probabilities': probabilities,
        'test_labels': test_labels,
        'test_passwords': test_passwords
    }
    
    return results


def plot_evaluation_results(results: dict, save_dir: str = 'training/evaluation'):
    """
    Create evaluation plots and save them.
    
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        classes=['weak', 'medium', 'strong'],
        title='Password Strength Classification - Confusion Matrix',
        save_path=f'{save_dir}/confusion_matrix.png'
    )
    
    # 2. Per-class metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(results['class_metrics'].keys())
    metrics = ['precision', 'recall', 'f1']
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results['class_metrics'][cls][metric] for cls in classes]
        ax.bar(x + i * width, values, width, label=metric.title())
    
    ax.set_xlabel('Password Strength Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        values = [results['class_metrics'][cls][metric] for cls in classes]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. ROC AUC scores
    fig, ax = plt.subplots(figsize=(8, 6))
    
    classes = list(results['roc_auc_scores'].keys())
    auc_scores = list(results['roc_auc_scores'].values())
    
    bars = ax.bar(classes, auc_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('ROC AUC Score')
    ax.set_title('ROC AUC Scores by Class')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_auc_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Prediction confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidences = [max(proba.values()) for proba in results['probabilities']]
    predictions = results['predictions']
    
    # Create box plot by prediction class
    confidence_by_class = {}
    for i, pred in enumerate(predictions):
        if pred not in confidence_by_class:
            confidence_by_class[pred] = []
        confidence_by_class[pred].append(confidences[i])
    
    data = [confidence_by_class[cls] for cls in ['weak', 'medium', 'strong']]
    labels = ['weak', 'medium', 'strong']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Prediction Confidence')
    ax.set_title('Prediction Confidence Distribution by Class')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation plots saved to: {save_dir}")


def generate_evaluation_report(results: dict, save_path: str = 'training/evaluation_report.txt'):
    """
    Generate a detailed evaluation report.
    
    Args:
        results: Evaluation results
        save_path: Path to save the report
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("🔐 PassClass Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall accuracy
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Performance:\n")
        f.write("-" * 30 + "\n")
        for cls, metrics in results['class_metrics'].items():
            f.write(f"{cls.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n\n")
        
        # ROC AUC scores
        f.write("ROC AUC Scores:\n")
        f.write("-" * 20 + "\n")
        for cls, auc in results['roc_auc_scores'].items():
            f.write(f"{cls.upper()}: {auc:.4f}\n")
        f.write(f"\nMean ROC AUC: {np.mean(list(results['roc_auc_scores'].values())):.4f}\n\n")
        
        # Detailed classification report
        f.write("Detailed Classification Report:\n")
        f.write("-" * 35 + "\n")
        f.write(classification_report(results['test_labels'], results['predictions']))
        
        # Confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 20 + "\n")
        f.write("Predicted →\n")
        f.write("Actual ↓\n")
        f.write("           weak  medium  strong\n")
        
        classes = ['weak', 'medium', 'strong']
        cm = results['confusion_matrix']
        
        for i, cls in enumerate(classes):
            f.write(f"{cls:8} {cm[i][0]:6} {cm[i][1]:7} {cm[i][2]:7}\n")
    
    print(f"Evaluation report saved to: {save_path}")


def test_model_on_examples(classifier: TFIDFPasswordClassifier):
    """
    Test the model on specific example passwords.
    
    Args:
        classifier: Trained classifier
    """
    test_examples = [
        ("abc", "weak"),
        ("password123", "weak"),
        ("Hello2023!", "medium"),
        ("G7^s9L!zB1m", "strong"),
        ("qwerty", "weak"),
        ("MyPass@123", "medium"),
        ("tR#8$!XmPq@", "strong"),
        ("123456789", "weak"),
        ("SecurePass1!", "medium"),
        ("K9#mN2$pL7@", "strong"),
        ("admin", "weak"),
        ("user123", "weak"),
        ("Test@2023", "medium"),
        ("Complex#Pass1", "strong"),
        ("simple", "weak"),
        ("MixedCase123", "medium"),
        ("UltraSecure#2023!", "strong"),
    ]
    
    print("\nTesting Model on Example Passwords:")
    print("=" * 60)
    print(f"{'Password':<20} {'Predicted':<10} {'Actual':<10} {'Confidence':<12} {'Status'}")
    print("-" * 60)
    
    correct = 0
    total = len(test_examples)
    
    for password, expected in test_examples:
        result = classifier.predict_with_confidence(password)
        predicted = result['predicted_label']
        confidence = result['confidence']
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{password:<20} {predicted:<10} {expected:<10} {confidence:<12.3f} {status}")
    
    accuracy = correct / total
    print("-" * 60)
    print(f"Example Accuracy: {correct}/{total} ({accuracy:.1%})")


def main():
    """Main evaluation function."""
    print("🔐 PassClass Model Evaluation")
    print("=" * 50)
    
    # Load model (try different model types)
    model_types = ['random_forest', 'logistic_regression', 'svm']
    classifier = None
    
    for model_type in model_types:
        model_path = f'models/{model_type}_password_classifier.pkl'
        classifier = load_trained_model(model_path)
        if classifier is not None:
            print(f"Using {model_type} model for evaluation")
            break
    
    if classifier is None:
        print("No trained model found. Please run training/train_model.py first.")
        return
    
    # Load test data
    df = load_test_data()
    if df is None:
        print("No test data found. Please run data/generate_passwords.py first.")
        return
    
    # Prepare test data
    test_passwords = df['password'].tolist()
    test_labels = df['actual_label'].tolist()
    
    # Evaluate model
    print(f"\nEvaluating model on {len(test_passwords)} test samples...")
    results = evaluate_model_performance(classifier, test_passwords, test_labels)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Mean ROC AUC: {np.mean(list(results['roc_auc_scores'].values())):.4f}")
    
    # Generate plots
    plot_evaluation_results(results)
    
    # Generate report
    generate_evaluation_report(results)
    
    # Test on examples
    test_model_on_examples(classifier)
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 