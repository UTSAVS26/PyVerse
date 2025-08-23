"""
Streamlit web interface for QuickML AutoML Engine.
Provides a beautiful and interactive interface for uploading data and viewing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings

from quickml import QuickML

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="QuickML - Mini AutoML Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 QuickML</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Mini AutoML Engine - Upload any CSV and get the best model automatically!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload your CSV file",
            type=['csv'],
            help="Upload any CSV file with your dataset"
        )
        
        # Target column selection
        target_column = st.text_input(
            "🎯 Target Column (optional)",
            help="Leave empty for auto-detection (last column)"
        )
        
        # Task type selection
        task_type = st.selectbox(
            "📊 Task Type",
            ["Auto-detect", "Classification", "Regression"],
            help="Let QuickML auto-detect or specify manually"
        )
        
        # Cross-validation folds
        cv_folds = st.slider(
            "🔄 Cross-validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
        
        # Run button
        run_analysis = st.button(
            "🚀 Run QuickML Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                
                # Data info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Info:**")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                with col2:
                    st.write("**Missing Values:**")
                    missing_data = df.isnull().sum()
                    st.dataframe(missing_data[missing_data > 0].to_frame('Missing Count'))
            
            # Run analysis
            if run_analysis:
                with st.spinner("🚀 Running QuickML AutoML Pipeline..."):
                    # Initialize QuickML
                    quickml = QuickML(
                        target_column=target_column if target_column else None,
                        task_type=task_type.lower() if task_type != "Auto-detect" else None
                    )
                    
                    # Fit the model
                    results = quickml.fit(df, cv_folds=cv_folds)
                    
                    # Store in session state
                    st.session_state.quickml = quickml
                    st.session_state.results = results
                    st.session_state.df = df
                
                # Display results
                display_results(quickml, results, df)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    else:
        # Welcome message
        st.markdown("""
        <div class="info-message">
            <h3>🎯 How to use QuickML:</h3>
            <ol>
                <li><strong>Upload your CSV file</strong> - Any dataset with features and target</li>
                <li><strong>Configure settings</strong> - Target column, task type, CV folds</li>
                <li><strong>Run analysis</strong> - QuickML will automatically:
                    <ul>
                        <li>Detect target column and task type</li>
                        <li>Handle missing values and encoding</li>
                        <li>Train multiple models (5 algorithms)</li>
                        <li>Select the best performing model</li>
                        <li>Generate visualizations and insights</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Example datasets
        st.subheader("📚 Example Datasets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Classification Examples:**")
            st.markdown("- Iris dataset")
            st.markdown("- Breast cancer")
            st.markdown("- Customer churn")
        
        with col2:
            st.markdown("**Regression Examples:**")
            st.markdown("- House prices")
            st.markdown("- Stock prices")
            st.markdown("- Sales forecasting")
        
        with col3:
            st.markdown("**Features:**")
            st.markdown("- Auto preprocessing")
            st.markdown("- Multiple algorithms")
            st.markdown("- Model comparison")
            st.markdown("- Feature importance")


def display_results(quickml, results, df):
    """Display QuickML results in a beautiful format."""
    
    st.markdown("## 🎉 Analysis Complete!")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best Model",
            results['best_model_name'],
            help="The best performing model"
        )
    
    with col2:
        st.metric(
            "Score",
            f"{results['best_score']:.4f}",
            help="Cross-validation score"
        )
    
    with col3:
        st.metric(
            "Task Type",
            results['task_type'].title(),
            help="Classification or Regression"
        )
    
    with col4:
        st.metric(
            "Features",
            results['transformed_shape'][1],
            help="Number of features after preprocessing"
        )
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Comparison", 
        "🎯 Model Performance", 
        "📈 Feature Importance",
        "📋 Data Analysis",
        "💾 Download Model"
    ])
    
    with tab1:
        display_model_comparison(quickml)
    
    with tab2:
        display_model_performance(quickml, results, df)
    
    with tab3:
        display_feature_importance(quickml)
    
    with tab4:
        display_data_analysis(df, results)
    
    with tab5:
        display_model_download(quickml)


def display_model_comparison(quickml):
    """Display model comparison results."""
    st.subheader("📊 Model Performance Comparison")
    
    # Get model summary
    model_summary = quickml.get_model_summary()
    
    if not model_summary.empty:
        # Display table
        st.dataframe(model_summary, use_container_width=True)
        
        # Create interactive plot
        fig = go.Figure(data=[
            go.Bar(
                y=model_summary['Model'],
                x=model_summary['Mean Score'].astype(float),
                orientation='h',
                text=[f'{val:.4f}' for val in model_summary['Mean Score'].astype(float)],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Mean Score",
            yaxis_title="Model",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model comparison data available.")


def display_model_performance(quickml, results, df):
    """Display detailed model performance metrics."""
    st.subheader("🎯 Model Performance Details")
    
    # Evaluation metrics
    eval_metrics = results['evaluation_metrics']
    
    if eval_metrics:
        # Display metrics in columns
        metrics_cols = st.columns(3)
        
        for i, (metric, value) in enumerate(eval_metrics.items()):
            if metric not in ['test_size', 'train_size']:
                col_idx = i % 3
                with metrics_cols[col_idx]:
                    if isinstance(value, float):
                        st.metric(metric.title(), f"{value:.4f}")
                    else:
                        st.metric(metric.title(), str(value))
        
        # Model-specific visualizations
        if results['task_type'] == 'classification':
            # Confusion matrix
            st.subheader("📊 Confusion Matrix")
            y_pred = quickml.predict(df)
            
            # Create confusion matrix plot
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(df[results['target_column']].values, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC curve if available
            try:
                y_proba = quickml.predict_proba(df)
                st.subheader("📈 ROC Curve")
                
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(df[results['target_column']].values, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC curve (AUC = {roc_auc:.2f})',
                    line=dict(color='darkorange', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except:
                st.info("ROC curve not available for this model.")
        
        else:  # Regression
            # Prediction vs actual plot
            st.subheader("📈 Prediction vs Actual")
            y_pred = quickml.predict(df)
            
            fig = px.scatter(
                x=df[results['target_column']].values,
                y=y_pred,
                title="Prediction vs Actual Values",
                labels={'x': 'Actual Values', 'y': 'Predicted Values'}
            )
            
            # Add perfect prediction line
            min_val = min(df[results['target_column']].min(), y_pred.min())
            max_val = max(df[results['target_column']].max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)


def display_feature_importance(quickml):
    """Display feature importance analysis."""
    st.subheader("📈 Feature Importance")
    
    feature_importance = quickml.get_feature_importance()
    
    if feature_importance is not None:
        feature_names = quickml.feature_names
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        # Show top features
        top_n = min(10, len(importance_df))
        top_features = importance_df.tail(top_n)
        
        # Display table
        st.dataframe(top_features, use_container_width=True)
        
        # Create plot
        fig = go.Figure(data=[
            go.Bar(
                y=top_features['Feature'],
                x=top_features['Importance'],
                orientation='h',
                text=[f'{val:.4f}' for val in top_features['Importance']],
                textposition='auto',
                marker_color='lightcoral'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")


def display_data_analysis(df, results):
    """Display data analysis and insights."""
    st.subheader("📋 Data Analysis")
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Overview:**")
        st.write(f"- Shape: {df.shape}")
        st.write(f"- Target column: {results['target_column']}")
        st.write(f"- Task type: {results['task_type']}")
        st.write(f"- Missing values: {df.isnull().sum().sum()}")
    
    with col2:
        st.write("**Target Distribution:**")
        target_counts = df[results['target_column']].value_counts()
        st.dataframe(target_counts.to_frame('Count'))
    
    # Data visualizations
    st.subheader("📊 Data Visualizations")
    
    # Target distribution
    if results['task_type'] == 'classification':
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            title="Target Distribution",
            labels={'x': 'Class', 'y': 'Count'}
        )
    else:
        fig = px.histogram(
            df, 
            x=results['target_column'],
            title="Target Distribution",
            nbins=30
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.subheader("🔍 Missing Values Analysis")
        missing_df = missing_data[missing_data > 0].to_frame('Missing Count')
        st.dataframe(missing_df)
        
        fig = px.bar(
            x=missing_df.index,
            y=missing_df['Missing Count'],
            title="Missing Values by Column"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_model_download(quickml):
    """Display model download options."""
    st.subheader("💾 Download Model")
    
    st.markdown("""
    Download your trained model for later use. The model file includes:
    - Trained model
    - Preprocessing pipeline
    - Feature names
    - Model metadata
    """)
    
    # Create download button
    if st.button("📥 Download Model", type="primary"):
        # Save model to bytes
        import joblib
        import io
        
        model_package = {
            'model': quickml.best_model,
            'preprocessor': quickml.preprocessor,
            'model_name': quickml.best_model_name,
            'task_type': quickml.task_type,
            'target_column': quickml.target_column,
            'feature_names': quickml.feature_names,
            'results': quickml.results
        }
        
        buffer = io.BytesIO()
        joblib.dump(model_package, buffer)
        buffer.seek(0)
        
        # Create download link
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="quickml_model.pkl">Download Model File</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Usage instructions
    st.markdown("""
    ### 🔧 How to use the downloaded model:
    
    ```python
    import joblib
    from quickml import QuickML
    
    # Load the model
    model_package = joblib.load('quickml_model.pkl')
    
    # Create QuickML instance
    quickml = QuickML()
    quickml.load_model('quickml_model.pkl')
    
    # Make predictions on new data
    new_data = pd.read_csv('new_data.csv')
    predictions = quickml.predict(new_data)
    ```
    """)


if __name__ == "__main__":
    main()
