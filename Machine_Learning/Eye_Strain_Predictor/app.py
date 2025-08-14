"""
Eye Strain Predictor Web Application

A Streamlit-based web application that predicts digital eye strain risk
based on user screen usage habits and lifestyle factors.

Author: AI Assistant
Date: 2025
License: MIT
"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

# Page configuration
st.set_page_config(
    page_title="👁️ Eye Strain Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page configuration
st.set_page_config(
    page_title="👁️ Eye Strain Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model() -> Dict[str, Any]:
    """
    Load the trained machine learning model and associated components.
    
    Returns:
        Dict[str, Any]: Dictionary containing model, scaler, feature columns, 
                       accuracy, and target names
    
    Raises:
        FileNotFoundError: If model file is not found
        Exception: For any other loading errors
    """
    try:
        model_data: Dict[str, Any] = joblib.load('eye_strain_model.joblib')
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run train_model.py first!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def create_probability_chart(target_names: List[str], probabilities: np.ndarray) -> None:
    """
    Create and display a probability bar chart using matplotlib.
    
    Args:
        target_names (List[str]): List of risk level names
        probabilities (np.ndarray): Array of prediction probabilities
    """
    # Create matplotlib chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # Green, Yellow, Orange, Red
    bars = ax.bar(target_names, probabilities, color=colors)
    
    ax.set_title("Risk Level Probabilities", fontsize=16, fontweight='bold')
    ax.set_xlabel("Risk Level", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add probability values on top of bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def get_recommendations(prediction: int) -> Tuple[str, List[str]]:
    """
    Get personalized recommendations based on eye strain risk level.
    
    Args:
        prediction (int): Predicted risk level (0-3)
    
    Returns:
        Tuple[str, List[str]]: Status message and list of recommendations
    """
    recommendations_map = {
        0: (
            "✅ Great! You have healthy screen habits. Keep up the good work!",
            [
                "Continue taking regular breaks",
                "Maintain good posture and screen distance",
                "Keep up your current eye care routine"
            ]
        ),
        1: (
            "⚠️ You have mild eye strain risk. Consider some improvements:",
            [
                "Increase break frequency to every hour",
                "Adjust screen brightness to match surroundings",
                "Try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds",
                "Consider using a blue light filter"
            ]
        ),
        2: (
            "🟠 Moderate eye strain risk detected. Take action now:",
            [
                "Reduce daily screen time if possible",
                "Take breaks every 30-45 minutes",
                "Increase screen distance to at least 60cm",
                "Improve room lighting to reduce glare",
                "Practice eye exercises regularly",
                "Consider computer glasses with blue light protection"
            ]
        ),
        3: (
            "🔴 High eye strain risk! Immediate changes needed:",
            [
                "Significantly reduce screen time",
                "Take breaks every 20-30 minutes",
                "Consult an eye care professional",
                "Use artificial tears if eyes feel dry",
                "Ensure proper room lighting",
                "Consider a screen break of several hours daily",
                "Practice blinking exercises",
                "Use a humidifier to prevent dry eyes"
            ]
        )
    }
    
    return recommendations_map.get(prediction, recommendations_map[1])


# Load model
model_data: Dict[str, Any] = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_columns: List[str] = model_data['feature_columns']
target_names: List[str] = model_data['target_names']

def main() -> None:
    """
    Main application function that runs the Streamlit interface.
    """
    # Main title
    st.title("👁️ Eye Strain Predictor")
    st.markdown("### Predict your risk of digital eye strain based on your screen usage habits")
    st.markdown("---")

    # Create input form
    with st.form("eye_strain_form"):
        st.subheader("📝 Please fill in your information")
        
        # Input form in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**👤 Personal Information**")
            age: int = st.slider("Age", 16, 65, 25, help="Your current age")
            sleep_quality: int = st.select_slider(
                "Sleep Quality", 
                options=[1, 2, 3, 4, 5], 
                value=3,
                format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x-1],
                help="Rate your overall sleep quality"
            )
            
            previous_eye_problems: str = st.selectbox(
                "Previous Eye Problems", 
                ["No", "Yes"],
                help="Do you have any existing eye conditions?"
            )

        with col2:
            st.markdown("**📱 Screen Usage**")
            screen_time_hours: float = st.slider(
                "Daily Screen Time (hours)", 
                0.5, 16.0, 8.0, 0.5,
                help="Total hours spent looking at screens per day"
            )
            screen_brightness_percent: int = st.slider(
                "Screen Brightness (%)", 
                10, 100, 70,
                help="Average screen brightness setting"
            )
            screen_distance_cm: int = st.slider(
                "Screen Distance (cm)", 
                20, 100, 50,
                help="Average distance from your eyes to the screen"
            )

        # More inputs
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🏠 Environment**")
            room_lighting: str = st.selectbox(
                "Room Lighting", 
                ["Poor", "Adequate"],
                help="Quality of ambient lighting in your workspace"
            )
            
            blink_rate_per_min: int = st.slider(
                "Blink Rate (per minute)", 
                5, 25, 15,
                help="Estimated number of times you blink per minute"
            )

        with col4:
            st.markdown("**💡 Habits**")
            break_frequency_per_hour: float = st.slider(
                "Breaks per Hour", 
                0.0, 6.0, 2.0, 0.5,
                help="Number of screen breaks you take per hour"
            )
            
            blue_light_filter: str = st.selectbox(
                "Blue Light Filter", 
                ["No", "Yes"],
                help="Do you use blue light filtering?"
            )
            
            eye_exercises: str = st.selectbox(
                "Eye Exercises", 
                ["No", "Yes"],
                help="Do you regularly do eye exercises?"
            )

        # Submit button
        submitted: bool = st.form_submit_button("🔍 Predict Eye Strain Risk", use_container_width=True)
        
        if submitted:
            # Process inputs
            processed_data = process_user_input(
                age, screen_time_hours, screen_brightness_percent, screen_distance_cm,
                room_lighting, blink_rate_per_min, break_frequency_per_hour,
                sleep_quality, blue_light_filter, eye_exercises, previous_eye_problems
            )
            
            # Make prediction
            prediction, probabilities = make_prediction(processed_data)
            
            # Display results
            display_results(prediction, probabilities, target_names)


def process_user_input(
    age: int, screen_time_hours: float, screen_brightness_percent: int,
    screen_distance_cm: int, room_lighting: str, blink_rate_per_min: int,
    break_frequency_per_hour: float, sleep_quality: int, blue_light_filter: str,
    eye_exercises: str, previous_eye_problems: str
) -> np.ndarray:
    """
    Process user inputs and convert them to the format expected by the model.
    
    Args:
        age, screen_time_hours, etc.: User input values
    
    Returns:
        np.ndarray: Processed input array ready for prediction
    """
    # Convert categorical variables to numerical
    room_lighting_val = 1 if room_lighting == "Adequate" else 0
    blue_light_filter_val = 1 if blue_light_filter == "Yes" else 0
    eye_exercises_val = 1 if eye_exercises == "Yes" else 0
    previous_eye_problems_val = 1 if previous_eye_problems == "Yes" else 0
    
    # Create input array in the correct order
    user_input = np.array([[
        age, screen_time_hours, screen_brightness_percent, screen_distance_cm,
        room_lighting_val, blink_rate_per_min, break_frequency_per_hour,
        sleep_quality, blue_light_filter_val, eye_exercises_val, previous_eye_problems_val
    ]])
    
    return user_input


def make_prediction(user_input: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Make prediction using the trained model.
    
    Args:
        user_input (np.ndarray): Processed user input
    
    Returns:
        Tuple[int, np.ndarray]: Prediction and probabilities
    """
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)[0]
    probabilities = model.predict_proba(user_input_scaled)[0]
    
    return prediction, probabilities


def display_results(prediction: int, probabilities: np.ndarray, target_names: List[str]) -> None:
    """
    Display prediction results and recommendations.
    
    Args:
        prediction (int): Predicted risk level
        probabilities (np.ndarray): Prediction probabilities
        target_names (List[str]): Risk level names
    """
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Risk level colors
    risk_colors = {
        "None": "🟢",
        "Mild": "🟡", 
        "Moderate": "🟠",
        "Severe": "🔴"
    }
    
    risk_level = target_names[prediction]
    
    # Display main result
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"### {risk_colors[risk_level]} {risk_level} Eye Strain Risk")
        
        # Display probability chart
        create_probability_chart(target_names, probabilities)
    
    # Display recommendations
    st.markdown("---")
    st.subheader("💡 Personalized Recommendations")
    
    status_message, recommendations = get_recommendations(prediction)
    
    if prediction == 0:  # None
        st.success(status_message)
    elif prediction == 1:  # Mild
        st.warning(status_message)
    elif prediction == 2:  # Moderate
        st.warning(status_message)
    else:  # Severe
        st.error(status_message)
    
    # Display recommendations as a list
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Additional information
    with st.expander("ℹ️ About This Prediction"):
        st.markdown("""
        **How it works:**
        - This model uses machine learning to analyze your screen usage patterns
        - It considers factors like screen time, environment, and personal habits
        - Predictions are based on established research about digital eye strain
        
        **Accuracy:** {:.1f}%
        
        **Disclaimer:** This tool is for educational purposes only. 
        For serious eye concerns, please consult with an eye care professional.
        """.format(model_data['accuracy'] * 100))


if __name__ == "__main__":
    main()
