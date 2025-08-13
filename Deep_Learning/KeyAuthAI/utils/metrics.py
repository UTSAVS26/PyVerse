"""
Metrics Module for KeyAuthAI

This module provides evaluation metrics for keystroke dynamics authentication:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate) 
- EER (Equal Error Rate)
- ROC curve plotting
- Performance evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


class KeystrokeMetrics:
    """Calculates and visualizes keystroke dynamics authentication metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics = {}
    
    def calculate_far_frr(self, genuine_scores: List[float], 
                          impostor_scores: List[float], 
                          threshold: float) -> Dict[str, float]:
        """
        Calculate FAR and FRR for a given threshold.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            threshold: Authentication threshold
            
        Returns:
            Dictionary with FAR and FRR values
        """
        if not genuine_scores or not impostor_scores:
            return {'far': 0.0, 'frr': 0.0}
        
        # Calculate FAR (False Acceptance Rate)
        # FAR = impostors accepted / total impostors
        impostors_accepted = sum(1 for score in impostor_scores if score >= threshold)
        far = impostors_accepted / len(impostor_scores) if impostor_scores else 0.0
        
        # Calculate FRR (False Rejection Rate)
        # FRR = genuine users rejected / total genuine users
        genuine_rejected = sum(1 for score in genuine_scores if score < threshold)
        frr = genuine_rejected / len(genuine_scores) if genuine_scores else 0.0
        
        return {
            'far': far,
            'frr': frr,
            'threshold': threshold,
            'genuine_count': len(genuine_scores),
            'impostor_count': len(impostor_scores)
        }
    
    def calculate_eer(self, genuine_scores: List[float], 
                     impostor_scores: List[float]) -> Dict[str, float]:
        """
        Calculate EER (Equal Error Rate) and optimal threshold.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            
        Returns:
            Dictionary with EER and optimal threshold
        """
        if not genuine_scores or not impostor_scores:
            return {'eer': 0.0, 'optimal_threshold': 0.0}
        
        # Combine all scores and sort
        all_scores = genuine_scores + impostor_scores
        thresholds = np.unique(all_scores)
        
        # Calculate FAR and FRR for each threshold
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            metrics = self.calculate_far_frr(genuine_scores, impostor_scores, threshold)
            far_values.append(metrics['far'])
            frr_values.append(metrics['frr'])
        
        # Find EER (where FAR = FRR)
        eer = 0.0
        optimal_threshold = 0.0
        
        for i, (far, frr) in enumerate(zip(far_values, frr_values)):
            if abs(far - frr) < 0.01:  # Within 1% tolerance
                eer = (far + frr) / 2
                optimal_threshold = thresholds[i]
                break
        
        # If no exact match, find closest
        if eer == 0.0:
            differences = [abs(far - frr) for far, frr in zip(far_values, frr_values)]
            min_idx = np.argmin(differences)
            eer = (far_values[min_idx] + frr_values[min_idx]) / 2
            optimal_threshold = thresholds[min_idx]
        
        return {
            'eer': eer,
            'optimal_threshold': optimal_threshold,
            'thresholds': thresholds.tolist(),
            'far_values': far_values,
            'frr_values': frr_values
        }
    
    def calculate_roc_metrics(self, genuine_scores: List[float], 
                            impostor_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate ROC curve metrics.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            
        Returns:
            Dictionary with ROC metrics
        """
        if not genuine_scores or not impostor_scores:
            return {'auc': 0.0, 'fpr': [], 'tpr': [], 'thresholds': []}
        
        # Prepare labels
        y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
        y_scores = genuine_scores + impostor_scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def evaluate_model_performance(self, genuine_scores: List[float], 
                                 impostor_scores: List[float]) -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            
        Returns:
            Dictionary with comprehensive metrics
        """
        if not genuine_scores or not impostor_scores:
            return {'error': 'Insufficient data for evaluation'}
        
        # Basic statistics
        genuine_mean = np.mean(genuine_scores)
        genuine_std = np.std(genuine_scores)
        impostor_mean = np.mean(impostor_scores)
        impostor_std = np.std(impostor_scores)
        
        # Calculate EER
        eer_metrics = self.calculate_eer(genuine_scores, impostor_scores)
        
        # Calculate ROC metrics
        roc_metrics = self.calculate_roc_metrics(genuine_scores, impostor_scores)
        
        # Calculate FAR/FRR at optimal threshold
        optimal_metrics = self.calculate_far_frr(
            genuine_scores, impostor_scores, eer_metrics['optimal_threshold']
        )
        
        # Calculate additional metrics
        # Calculate additional metrics
        denominator = np.sqrt((genuine_std**2 + impostor_std**2) / 2)
        if denominator == 0:
            # If both stds are zero, all scores are identical:
            # infinite separability only if means differ, else zero
            d_prime = float('inf') if genuine_mean != impostor_mean else 0.0
        else:
            d_prime = abs(genuine_mean - impostor_mean) / denominator
        
        return {
            'genuine_stats': {
                'mean': genuine_mean,
                'std': genuine_std,
                'min': np.min(genuine_scores),
                'max': np.max(genuine_scores)
            },
            'impostor_stats': {
                'mean': impostor_mean,
                'std': impostor_std,
                'min': np.min(impostor_scores),
                'max': np.max(impostor_scores)
            },
            'eer': eer_metrics['eer'],
            'optimal_threshold': eer_metrics['optimal_threshold'],
            'far_at_eer': optimal_metrics['far'],
            'frr_at_eer': optimal_metrics['frr'],
            'roc_auc': roc_metrics['auc'],
            'd_prime': d_prime,
            'total_genuine': len(genuine_scores),
            'total_impostor': len(impostor_scores)
        }
    
    def plot_roc_curve(self, genuine_scores: List[float], 
                      impostor_scores: List[float], 
                      title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve for keystroke dynamics authentication.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not genuine_scores or not impostor_scores:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient data for ROC curve', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate ROC curve
        roc_metrics = self.calculate_roc_metrics(genuine_scores, impostor_scores)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(roc_metrics['fpr'], roc_metrics['tpr'], 
               color='blue', linewidth=2, 
               label=f'ROC Curve (AUC = {roc_metrics["auc"]:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', 
               label='Random Classifier')
        
        # Customize plot
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('True Positive Rate (1 - FRR)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return fig
    
    def plot_score_distributions(self, genuine_scores: List[float], 
                               impostor_scores: List[float], 
                               title: str = "Score Distributions") -> plt.Figure:
        """
        Plot score distributions for genuine and impostor samples.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(genuine_scores, bins=20, alpha=0.7, label='Genuine', 
                color='green', density=True)
        ax1.hist(impostor_scores, bins=20, alpha=0.7, label='Impostor', 
                color='red', density=True)
        ax1.set_xlabel('Authentication Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data = [genuine_scores, impostor_scores]
        labels = ['Genuine', 'Impostor']
        ax2.boxplot(data, labels=labels)
        ax2.set_ylabel('Authentication Score')
        ax2.set_title('Score Statistics')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_far_frr_curve(self, genuine_scores: List[float], 
                          impostor_scores: List[float], 
                          title: str = "FAR/FRR Curve") -> plt.Figure:
        """
        Plot FAR/FRR curve showing the relationship between thresholds and error rates.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not genuine_scores or not impostor_scores:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient data for FAR/FRR curve', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate EER metrics
        eer_metrics = self.calculate_eer(genuine_scores, impostor_scores)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot FAR and FRR curves
        ax.plot(eer_metrics['thresholds'], eer_metrics['far_values'], 
               color='red', linewidth=2, label='FAR')
        ax.plot(eer_metrics['thresholds'], eer_metrics['frr_values'], 
               color='blue', linewidth=2, label='FRR')
        
        # Mark EER point
        ax.axhline(y=eer_metrics['eer'], color='green', linestyle='--', 
                  label=f'EER = {eer_metrics["eer"]:.3f}')
        ax.axvline(x=eer_metrics['optimal_threshold'], color='green', linestyle='--', 
                  label=f'Optimal Threshold = {eer_metrics["optimal_threshold"]:.3f}')
        
        # Customize plot
        ax.set_xlabel('Authentication Threshold')
        ax.set_ylabel('Error Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_performance_report(self, genuine_scores: List[float], 
                                  impostor_scores: List[float], 
                                  username: str = "User") -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            genuine_scores: Scores from legitimate user sessions
            impostor_scores: Scores from impostor sessions
            username: Username for the report
            
        Returns:
            Formatted performance report string
        """
        if not genuine_scores or not impostor_scores:
            return "Error: Insufficient data for performance evaluation"
        
        # Calculate metrics
        performance = self.evaluate_model_performance(genuine_scores, impostor_scores)
        
        # Generate report
        report = f"""
KeyAuthAI Performance Report
===========================
User: {username}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Information:
-------------------
Genuine samples: {performance['total_genuine']}
Impostor samples: {performance['total_impostor']}

Score Statistics:
----------------
Genuine scores:
  Mean: {performance['genuine_stats']['mean']:.4f}
  Std:  {performance['genuine_stats']['std']:.4f}
  Min:  {performance['genuine_stats']['min']:.4f}
  Max:  {performance['genuine_stats']['max']:.4f}

Impostor scores:
  Mean: {performance['impostor_stats']['mean']:.4f}
  Std:  {performance['impostor_stats']['std']:.4f}
  Min:  {performance['impostor_stats']['min']:.4f}
  Max:  {performance['impostor_stats']['max']:.4f}

Performance Metrics:
-------------------
EER (Equal Error Rate): {performance['eer']:.4f}
Optimal Threshold: {performance['optimal_threshold']:.4f}
FAR at EER: {performance['far_at_eer']:.4f}
FRR at EER: {performance['frr_at_eer']:.4f}
ROC AUC: {performance['roc_auc']:.4f}
d-prime: {performance['d_prime']:.4f}

Interpretation:
--------------
- EER: Lower is better (ideal: < 0.05)
- ROC AUC: Higher is better (ideal: > 0.95)
- d-prime: Higher is better (ideal: > 3.0)
- FAR: Lower is better (ideal: < 0.01)
- FRR: Lower is better (ideal: < 0.01)
"""
        
        return report


def calculate_far_frr(genuine_scores: List[float], 
                     impostor_scores: List[float], 
                     threshold: float) -> Dict[str, float]:
    """
    Convenience function to calculate FAR and FRR.
    
    Args:
        genuine_scores: Scores from legitimate user sessions
        impostor_scores: Scores from impostor sessions
        threshold: Authentication threshold
        
    Returns:
        Dictionary with FAR and FRR values
    """
    metrics = KeystrokeMetrics()
    return metrics.calculate_far_frr(genuine_scores, impostor_scores, threshold)


def calculate_eer(genuine_scores: List[float], 
                 impostor_scores: List[float]) -> Dict[str, float]:
    """
    Convenience function to calculate EER.
    
    Args:
        genuine_scores: Scores from legitimate user sessions
        impostor_scores: Scores from impostor sessions
        
    Returns:
        Dictionary with EER and optimal threshold
    """
    metrics = KeystrokeMetrics()
    return metrics.calculate_eer(genuine_scores, impostor_scores)


if __name__ == "__main__":
    # Example usage
    print("KeyAuthAI Metrics Module")
    print("=" * 30)
    
    # Sample data
    genuine_scores = [0.8, 0.85, 0.9, 0.75, 0.88, 0.92, 0.87, 0.83, 0.89, 0.91]
    impostor_scores = [0.3, 0.25, 0.4, 0.35, 0.2, 0.45, 0.3, 0.25, 0.35, 0.4]
    
    metrics = KeystrokeMetrics()
    
    # Calculate EER
    eer_result = metrics.calculate_eer(genuine_scores, impostor_scores)
    print(f"EER: {eer_result['eer']:.4f}")
    print(f"Optimal Threshold: {eer_result['optimal_threshold']:.4f}")
    
    # Calculate performance metrics
    performance = metrics.evaluate_model_performance(genuine_scores, impostor_scores)
    print(f"ROC AUC: {performance['roc_auc']:.4f}")
    print(f"d-prime: {performance['d_prime']:.4f}")
    
    # Generate report
    report = metrics.generate_performance_report(genuine_scores, impostor_scores, "test_user")
    print(report) 