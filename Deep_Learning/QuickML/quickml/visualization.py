"""
Visualization utilities for QuickML.
Handles plotting and creating visualizations for model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Handles all visualization tasks for QuickML including model performance,
    feature importance, and data exploration.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the visualizer.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metric: str = 'Mean Score', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a bar plot comparing model performance.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by the specified metric
        df_sorted = comparison_df.sort_values(metric, ascending=True)
        
        # Create bar plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center')
        
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Model')
        ax.set_title(f'Model Performance Comparison - {metric.capitalize()}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                               top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a feature importance plot.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores.flatten() if importance_scores.ndim > 1 else importance_scores
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(importance_df['feature'], importance_df['importance'])
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center')
        
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Features')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create ROC curve plot.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != 'classification':
            raise ValueError("ROC curve only available for classification tasks")
        
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create prediction vs actual plot for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.task_type != 'regression':
            raise ValueError("Prediction vs actual plot only available for regression tasks")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Prediction vs Actual Values')
        ax.grid(True, alpha=0.3)
        
        # Add R² value
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_data_distribution(self, df: pd.DataFrame, target_column: str,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create data distribution plots.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Target distribution
        if self.task_type == 'classification':
            df[target_column].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Target Distribution')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Count')
        else:
            df[target_column].hist(bins=30, ax=axes[0, 0])
            axes[0, 0].set_title('Target Distribution')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
        
        # Missing values
        missing_data = df.isnull().sum()
        missing_columns = missing_data[missing_data > 0]
        if len(missing_columns) > 0:
            missing_columns.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values')
            axes[0, 1].set_xlabel('Column')
            axes[0, 1].set_ylabel('Missing Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Missing Values')
        
        # Numerical columns distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            df[numerical_cols[0]].hist(bins=30, ax=axes[1, 0])
            axes[1, 0].set_title(f'Distribution of {numerical_cols[0]}')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
        
        # Categorical columns distribution
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df[categorical_cols[0]].value_counts().head(10).plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title(f'Top 10 Values in {categorical_cols[0]}')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_model_comparison(self, comparison_df: pd.DataFrame, 
                                          metric: str = 'Mean Score') -> go.Figure:
        """
        Create an interactive model comparison plot using Plotly.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            
        Returns:
            Plotly figure
        """
        # Sort by the specified metric
        df_sorted = comparison_df.sort_values(metric, ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                y=df_sorted['Model'],
                x=df_sorted[metric],
                orientation='h',
                text=[f'{val:.4f}' for val in df_sorted[metric]],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=f'Model Performance Comparison - {metric.capitalize()}',
            xaxis_title=metric.capitalize(),
            yaxis_title='Model',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_feature_importance(self, feature_names: List[str], 
                                            importance_scores: np.ndarray,
                                            top_n: int = 10) -> go.Figure:
        """
        Create an interactive feature importance plot using Plotly.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['feature'],
                x=importance_df['importance'],
                orientation='h',
                text=[f'{val:.4f}' for val in importance_df['importance']],
                textposition='auto',
                marker_color='lightcoral'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Feature Importance',
            yaxis_title='Features',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def save_all_plots(self, plots: Dict[str, plt.Figure], 
                      output_dir: str = 'plots', dpi: int = 300) -> None:
        """
        Save all plots to files.
        
        Args:
            plots: Dictionary of plot name to matplotlib figure
            output_dir: Output directory
            dpi: DPI for saved images
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in plots.items():
            filename = f"{output_dir}/{name.lower().replace(' ', '_')}.png"
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Saved {filename}")
        
        plt.close('all')
