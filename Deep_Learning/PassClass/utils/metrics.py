"""
Metrics and Visualization Utilities

This module provides utility functions for plotting and analyzing model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                         title: str = 'Confusion Matrix',
                         save_path: str = None, normalize: bool = False):
    """
    Plot confusion matrix with custom styling.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names
        title: Plot title
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_classification_report(report: Dict[str, Any], 
                             title: str = 'Classification Report',
                             save_path: str = None):
    """
    Plot classification report as a heatmap.
    
    Args:
        report: Classification report dictionary
        title: Plot title
        save_path: Path to save the plot
    """
    # Extract metrics for plotting
    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []
    
    for class_name in ['weak', 'medium', 'strong']:
        if class_name in report:
            classes.append(class_name)
            precision.append(report[class_name]['precision'])
            recall.append(report[class_name]['recall'])
            f1_score.append(report[class_name]['f1-score'])
            support.append(report[class_name]['support'])
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    }, index=classes)
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar=True, square=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Metric', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance: Dict[str, float], 
                           top_n: int = 20,
                           title: str = 'Feature Importance',
                           save_path: str = None):
    """
    Plot feature importance for Random Forest model.
    
    Args:
        feature_importance: Dictionary of feature importance scores
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save the plot
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importance = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(features)), importance, 
                    color='skyblue', edgecolor='navy')
    
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', ha='left', va='center', fontsize=10)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_confidence(confidences: List[float], 
                             predictions: List[str],
                             title: str = 'Prediction Confidence Distribution',
                             save_path: str = None):
    """
    Plot distribution of prediction confidence scores.
    
    Args:
        confidences: List of confidence scores
        predictions: List of predicted labels
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall confidence distribution
    ax1.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Confidence by class
    confidence_by_class = {}
    for conf, pred in zip(confidences, predictions):
        if pred not in confidence_by_class:
            confidence_by_class[pred] = []
        confidence_by_class[pred].append(conf)
    
    # Box plot by class
    data = [confidence_by_class[cls] for cls in ['weak', 'medium', 'strong']]
    labels = ['weak', 'medium', 'strong']
    
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Confidence by Predicted Class', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(comparison_data: List[Dict[str, Any]],
                         title: str = 'Model Comparison',
                         save_path: str = None):
    """
    Plot comparison of different models.
    
    Args:
        comparison_data: List of dictionaries with model metrics
        title: Plot title
        save_path: Path to save the plot
    """
    df = pd.DataFrame(comparison_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(df['Model'], df['Accuracy'], 
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
    if 'CV Mean' in df.columns:
        bars2 = ax2.bar(df['Model'], df['CV Mean'], 
                        yerr=df['CV Std'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Cross-Validation Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CV Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_password_length_distribution(passwords: List[str], 
                                   labels: List[str],
                                   title: str = 'Password Length Distribution by Class',
                                   save_path: str = None):
    """
    Plot distribution of password lengths by class.
    
    Args:
        passwords: List of passwords
        labels: List of corresponding labels
        title: Plot title
        save_path: Path to save the plot
    """
    # Calculate lengths
    lengths = [len(pwd) for pwd in passwords]
    
    # Create DataFrame
    df = pd.DataFrame({
        'length': lengths,
        'label': labels
    })
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall length distribution
    ax1.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Password Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Password Length Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Length distribution by class
    for label in ['weak', 'medium', 'strong']:
        class_lengths = df[df['label'] == label]['length']
        ax2.hist(class_lengths, bins=15, alpha=0.6, label=label.title())
    
    ax2.set_xlabel('Password Length', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Length Distribution by Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_character_analysis(passwords: List[str], 
                          labels: List[str],
                          title: str = 'Character Analysis by Class',
                          save_path: str = None):
    """
    Plot character type analysis by class.
    
    Args:
        passwords: List of passwords
        labels: List of corresponding labels
        title: Plot title
        save_path: Path to save the plot
    """
    import string
    
    # Analyze character types
    analysis_data = []
    
    for pwd, label in zip(passwords, labels):
        analysis = {
            'label': label,
            'length': len(pwd),
            'uppercase': sum(1 for c in pwd if c.isupper()),
            'lowercase': sum(1 for c in pwd if c.islower()),
            'digits': sum(1 for c in pwd if c.isdigit()),
            'special': sum(1 for c in pwd if c in string.punctuation)
        }
        analysis_data.append(analysis)
    
    df = pd.DataFrame(analysis_data)
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different character types
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    char_types = ['uppercase', 'lowercase', 'digits', 'special']
    titles = ['Uppercase Letters', 'Lowercase Letters', 'Digits', 'Special Characters']
    
    for i, (char_type, title) in enumerate(zip(char_types, titles)):
        ax = axes[i//2, i%2]
        
        # Box plot by class
        data = [df[df['label'] == cls][char_type] for cls in ['weak', 'medium', 'strong']]
        labels = ['weak', 'medium', 'strong']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(f'Number of {title}', fontsize=12)
        ax.set_title(f'{title} by Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_performance_summary(results: Dict[str, Any]) -> str:
    """
    Create a text summary of model performance.
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        Formatted performance summary
    """
    summary = []
    summary.append("🔐 PassClass Model Performance Summary")
    summary.append("=" * 50)
    summary.append("")
    
    # Overall metrics
    summary.append(f"Overall Accuracy: {results['accuracy']:.4f}")
    summary.append("")
    
    # Per-class metrics
    summary.append("Per-Class Performance:")
    summary.append("-" * 30)
    # Per-class metrics
    summary.append("Per-Class Performance:")
    summary.append("-" * 30)
    if 'class_metrics' in results:
        for cls, metrics in results['class_metrics'].items():
            summary.append(f"{cls.upper()}:")
            summary.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            summary.append(f"  Recall: {metrics.get('recall', 0):.4f}")
            summary.append(f"  F1-Score: {metrics.get('f1', 0):.4f}")
            summary.append(f"  Support: {metrics.get('support', 0)}")
            summary.append("")
    
    # ROC AUC scores
    if 'roc_auc_scores' in results:
        summary.append("ROC AUC Scores:")
        summary.append("-" * 20)
        for cls, auc in results['roc_auc_scores'].items():
            summary.append(f"{cls.upper()}: {auc:.4f}")
        summary.append(f"Mean ROC AUC: {np.mean(list(results['roc_auc_scores'].values())):.4f}")
        summary.append("")
    
    return "\n".join(summary)


if __name__ == "__main__":
    # Test the metrics functions
    print("Testing metrics utilities...")
    
    # Sample data for testing
    sample_cm = np.array([[50, 10, 5], [8, 45, 7], [3, 8, 44]])
    classes = ['weak', 'medium', 'strong']
    
    # Test confusion matrix plotting
    plot_confusion_matrix(sample_cm, classes, "Sample Confusion Matrix")
    
    print("Metrics utilities test completed!") 