import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_plot_style():
    """Set consistent style for all plots"""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

def plot_time_series(data, columns, titles):
    """Plot time series for given columns"""
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4))
    for ax, col, title in zip(axes, columns, titles):
        ax.plot(data[col])
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    plt.tight_layout()
    return fig

# In src/visualization.py
def plot_correlation_matrix(data):
    """Plot correlation heatmap"""
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    return plt.gcf()

def plot_feature_importance(importance_df, title):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    return plt.gcf()