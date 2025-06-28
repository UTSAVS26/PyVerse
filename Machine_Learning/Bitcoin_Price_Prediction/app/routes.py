from flask import render_template, request, flash, redirect, url_for
from flask import Blueprint
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
from datetime import datetime
from src.data_processing import load_data, preprocess_data, prepare_training_data
from src.models import train_model, get_feature_importance_df, save_model
from src.visualization import set_plot_style, plot_time_series, plot_correlation_matrix, plot_feature_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from .forms import ModelForm

bp = Blueprint('main', __name__)

# Load and preprocess data once when the app starts
data = load_data('data/bitcoin_dataset.csv')
data_clean = preprocess_data(data)
X_train, X_test, y_train, y_test, feature_names, scaler = prepare_training_data(data_clean)

def save_plot_to_base64(fig):
    """Save matplotlib figure to base64 encoded string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/eda')
def eda():
    set_plot_style()
    
    # Time Series Plot (using Year if available)
    ts_cols = ['btc_market_price', 'btc_trade_volume', 'btc_blocks_size']
    if 'Year' in data_clean.columns:
        ts_fig = plt.figure(figsize=(15, 5))
        for col in ts_cols:
            plt.plot(data_clean['Year'], data_clean[col], label=col)
        plt.legend()
    else:
        ts_fig = plot_time_series(data_clean, ts_cols, ts_cols)
    
    ts_img = save_plot_to_base64(ts_fig)
    plt.close(ts_fig)
    
    # Correlation Matrix
    corr_fig = plot_correlation_matrix(data_clean)
    corr_img = save_plot_to_base64(corr_fig)
    plt.close(corr_fig)
    
    return render_template('eda.html', 
                         ts_plot=ts_img,
                         corr_plot=corr_img)

@bp.route('/model', methods=['GET', 'POST'])
def model():
    form = ModelForm()
    model_files = os.listdir('models') if os.path.exists('models') else []
    
    if form.validate_on_submit():
        model_type = form.model_type.data
        alpha = form.alpha.data
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
            model_name = "Linear Regression"
        elif model_type == 'ridge':
            model = Ridge(alpha=alpha)
            model_name = "Ridge Regression"
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
            model_name = "Lasso Regression"
        elif model_type == 'svr':
            model = SVR(kernel='linear')
            model_name = "Support Vector Regression"
        
        results = train_model(model, X_train, X_test, y_train, y_test)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
        save_model(results['model'], f"models/{filename}")
        
        # Feature importance
        fi_img = None
        if results['feature_importance'] is not None:
            importance_df = get_feature_importance_df(results['model'], feature_names)
            fi_fig = plot_feature_importance(importance_df, f"{model_name} Feature Importance")
            fi_img = save_plot_to_base64(fi_fig)
            plt.close(fi_fig)
        
        flash(f"{model_name} trained successfully! Train R²: {results['train_score']:.4f}, Test R²: {results['test_score']:.4f}", 'success')
        return render_template('model.html',
                             form=form,
                             model_name=model_name,
                             train_score=results['train_score'],
                             test_score=results['test_score'],
                             fi_plot=fi_img,
                             model_files=model_files,
                             show_results=True)
    
    return render_template('model.html', 
                         form=form, 
                         model_files=model_files,
                         show_results=False)