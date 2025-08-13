"""
PassClass Streamlit Web App

Interactive web application for password strength testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling.labeler import PasswordLabeler
from models.tfidf_classifier import TFIDFPasswordClassifier


def load_model():
    """Load the trained model."""
    model_types = ['random_forest', 'logistic_regression', 'svm']
    
    for model_type in model_types:
def load_model():
    """Load the trained model."""
    model_types = ['random_forest', 'logistic_regression', 'svm']
    errors = []

    for model_type in model_types:
        model_path = f'models/{model_type}_password_classifier.pkl'
        try:
            classifier = TFIDFPasswordClassifier()
            classifier.load_model(model_path)
            return classifier, model_type
        except FileNotFoundError as e:
            errors.append(f"{model_type}: {str(e)}")
            continue

    print(f"Failed to load any model. Attempted: {', '.join(errors)}")
    return None, None
    
    return None, None


def get_strength_color(label):
    """Get color for strength label."""
    colors = {
        'weak': '#FF6B6B',
        'medium': '#FFA726',
        'strong': '#66BB6A'
    }
    return colors.get(label, '#757575')


def get_strength_icon(label):
    """Get icon for strength label."""
    icons = {
        'weak': '🔴',
        'medium': '🟡',
        'strong': '🟢'
    }
    return icons.get(label, '⚪')


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PassClass - Password Strength Classifier",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .strength-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔐 PassClass</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">AI-Powered Password Strength Classifier</h2>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        classifier, model_type = load_model()
    
    if classifier is None:
        st.error("❌ No trained model found! Please run the training script first.")
        st.info("To train the model, run: `python training/train_model.py`")
        return
    
    st.success(f"✅ Model loaded successfully ({model_type.replace('_', ' ').title()})")
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Password Strength Analysis")
        
        # Password input
        password = st.text_input(
            "Enter a password to analyze:",
            placeholder="Type your password here...",
            type="password",
            help="Enter any password to see its strength analysis"
        )
        
        if password:
            # Get predictions
            with st.spinner("Analyzing password..."):
                # Model prediction
# At the top of your module (assuming PasswordLabeler is already imported)
@st.cache_resource
def get_labeler():
    """Get or create a cached PasswordLabeler instance."""
    return PasswordLabeler()

# … later in your code …

if password:
    # Get predictions
    with st.spinner("Analyzing password..."):
        # Model prediction
        model_result = classifier.predict_with_confidence(password)
        
        # Labeler analysis
        labeler = get_labeler()
        labeler_analysis = labeler.get_detailed_analysis(password)
                # Labeler analysis
                labeler = PasswordLabeler()
                labeler_analysis = labeler.get_detailed_analysis(password)
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            
            # Create columns for predictions
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                st.markdown("#### 🤖 ML Model Prediction")
                predicted_label = model_result['predicted_label']
                confidence = model_result['confidence']
                
                color = get_strength_color(predicted_label)
                icon = get_strength_icon(predicted_label)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {color}; margin-bottom: 0.5rem;">
                        {icon} {predicted_label.upper()}
                    </h3>
                    <p style="font-size: 1.2rem; margin: 0;">
                        Confidence: <strong>{confidence:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with pred_col2:
                st.markdown("#### 🧠 Rule-Based Analysis")
                labeler_label = labeler_analysis['label']
                
                color = get_strength_color(labeler_label)
                icon = get_strength_icon(labeler_label)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {color}; margin-bottom: 0.5rem;">
                        {icon} {labeler_label.upper()}
                    </h3>
                    <p style="font-size: 1.2rem; margin: 0;">
                        Rule-based assessment
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with pred_col3:
                st.markdown("#### ✅ Agreement")
                agreement = model_result['prediction_match']
                agreement_text = "✅ Match" if agreement else "❌ Disagree"
                agreement_color = "#4CAF50" if agreement else "#F44336"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {agreement_color}; margin-bottom: 0.5rem;">
                        {agreement_text}
                    </h3>
                    <p style="font-size: 1.2rem; margin: 0;">
                        Model vs Rules
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis
            st.markdown("### 📈 Detailed Analysis")
            
            # Password characteristics
            score = labeler_analysis['score']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Length", score['length'])
            
            with col2:
                st.metric("Uppercase", score['uppercase'])
            
            with col3:
                st.metric("Digits", score['digits'])
            
            with col4:
                st.metric("Special Chars", score['special'])
            
            # Issues and strengths
            col1, col2 = st.columns(2)
            
            with col1:
                if labeler_analysis['issues']:
                    st.markdown("#### ⚠️ Issues Found")
                    for issue in labeler_analysis['issues']:
                        st.markdown(f"• {issue}")
                else:
                    st.success("✅ No issues found!")
            
            with col2:
                if labeler_analysis['strengths']:
                    st.markdown("#### ✅ Strengths")
                    for strength in labeler_analysis['strengths']:
                        st.markdown(f"• {strength}")
                else:
                    st.warning("⚠️ No strengths identified")
            
            # Model probabilities
            st.markdown("### 🎯 Model Confidence Breakdown")
            
            proba = model_result['probabilities']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weak_prob = proba.get('weak', 0)
                st.metric("Weak Probability", f"{weak_prob:.1%}")
                st.progress(weak_prob)
            
            with col2:
                medium_prob = proba.get('medium', 0)
                st.metric("Medium Probability", f"{medium_prob:.1%}")
                st.progress(medium_prob)
            
            with col3:
                strong_prob = proba.get('strong', 0)
                st.metric("Strong Probability", f"{strong_prob:.1%}")
                st.progress(strong_prob)
    
    with col2:
        st.subheader("📚 Information")
        
        st.markdown("""
        **How it works:**
        
        This app uses a machine learning model trained on synthetic password data to predict password strength.
        
        **Strength Levels:**
        - 🔴 **Weak**: Easily guessable passwords
        - 🟡 **Medium**: Moderate security
        - 🟢 **Strong**: High security
        
        **Features:**
        - ML-based prediction
        - Rule-based analysis
        - Detailed breakdown
        - Confidence scores
        """)
        
        # Sample passwords
        st.markdown("### 🧪 Test Examples")
        
        sample_passwords = [
            ("abc", "weak"),
            ("password123", "weak"),
            ("Hello2023!", "medium"),
            ("G7^s9L!zB1m", "strong"),
            ("qwerty", "weak"),
            ("MyPass@123", "medium"),
            ("tR#8$!XmPq@", "strong"),
        ]
        
        for pwd, expected in sample_passwords:
            if st.button(f"Test: {pwd}", key=pwd):
     # Password input
     default_password = st.session_state.get('test_password', '')
     password = st.text_input(
         "Enter a password to analyze:",
         value=default_password,
         placeholder="Type your password here..." if not default_password else "",
         type="password",
         help="Enter any password to see its strength analysis"
     )
     
     # Clear test password after use
     if 'test_password' in st.session_state:
         del st.session_state['test_password']
        
        # Model info
        st.markdown("### 🤖 Model Info")
        st.info(f"""
        **Model Type:** {model_type.replace('_', ' ').title()}
        
        **Features:** TF-IDF character n-grams
        
        **Training:** Synthetic password dataset
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔐 PassClass - AI-Powered Password Strength Classifier</p>
        <p>Built with ❤️ using Streamlit and scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 