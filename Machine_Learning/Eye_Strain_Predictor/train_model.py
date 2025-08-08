"""
Eye Strain Predictor - Model Training Module

This module trains a Random Forest classifier to predict digital eye strain risk
based on user screen usage habits and lifestyle factors.

Features:
- Data loading and preprocessing
- Model training with Random Forest
- Performance evaluation and visualization
- Model persistence with joblib

Author: AI Assistant
Date: 2025
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data() -> Optional[pd.DataFrame]:
    """
    Load and prepare the eye strain dataset for training.
    
    Returns:
        Optional[pd.DataFrame]: Loaded dataset or None if file not found
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    try:
        df = pd.read_csv('eye_strain_dataset.csv')
        print("✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print("❌ Dataset not found. Please run generate_dataset.py first!")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def train_eye_strain_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train the eye strain prediction model using Random Forest.
    
    Args:
        df (pd.DataFrame): Input dataset containing features and target
        
    Returns:
        Dict[str, Any]: Dictionary containing trained model and metadata
    """
    
    # Prepare features and target
    feature_columns = [
        'age', 'screen_time_hours', 'screen_brightness_percent', 'screen_distance_cm',
        'room_lighting', 'blink_rate_per_min', 'break_frequency_per_hour', 
        'sleep_quality', 'blue_light_filter', 'eye_exercises', 'previous_eye_problems'
    ]
    
    X = df[feature_columns]
    y = df['eye_strain_level']
    
    print(f"\nFeatures: {feature_columns}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    print("\n🔄 Training Random Forest model...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['None', 'Mild', 'Moderate', 'Severe']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Feature Importance:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for Eye Strain Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['None', 'Mild', 'Moderate', 'Severe'],
                yticklabels=['None', 'Mild', 'Moderate', 'Severe'])
    plt.title('Confusion Matrix - Eye Strain Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'target_names': ['None', 'Mild', 'Moderate', 'Severe']
    }
    
    joblib.dump(model_data, 'eye_strain_model.joblib')
    print(f"\n💾 Model saved as 'eye_strain_model.joblib'")
    print(f"📊 Plots saved: feature_importance.png, confusion_matrix.png")
    
    return model_data

def test_model_prediction(model_data: Dict[str, Any]) -> None:
    """
    Test the model with sample predictions to verify functionality.
    
    Args:
        model_data (Dict[str, Any]): Dictionary containing model and metadata
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns: List[str] = model_data['feature_columns']
    target_names: List[str] = model_data['target_names']
    
    print(f"\n🧪 Testing model with sample data:")
    
    # Test cases representing different user profiles
    test_cases = [
        {
            'name': "Heavy Screen User (High Risk)",
            'data': [25, 12, 90, 35, 0, 8, 0.5, 2, 0, 0, 0]  # High strain expected
        },
        {
            'name': "Healthy User (Low Risk)",
            'data': [30, 4, 50, 60, 1, 16, 3, 4, 1, 1, 0]  # Low strain expected
        },
        {
            'name': "Office Worker (Moderate Risk)",
            'data': [35, 8, 70, 45, 1, 12, 2, 3, 1, 0, 0]  # Moderate strain expected
        }
    ]
    
    for test_case in test_cases:
        data = np.array([test_case['data']])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]
        
        print(f"\n{test_case['name']}:")
        print(f"Prediction: {target_names[prediction]}")
        print(f"Probabilities: {dict(zip(target_names, [f'{p:.2f}' for p in probability]))}")


def main() -> None:
    """
    Main function to execute the training pipeline.
    """
    print("🚀 Starting Eye Strain Model Training...")
    print("=" * 50)
    
    # Load data
    df = load_and_prepare_data()
    
    if df is not None:
        # Train model
        print(f"\n📊 Training model...")
        model_data = train_eye_strain_model(df)
        
        # Test model
        print(f"\n🧪 Testing model...")
        test_model_prediction(model_data)
        
        print(f"\n✅ Training complete!")
        print(f"📁 Model saved as 'eye_strain_model.joblib'")
        print(f"📊 Visualizations saved as PNG files")
        print(f"🎯 Model accuracy: {model_data['accuracy']:.1%}")
        print(f"\n🚀 You can now run the Streamlit app with: streamlit run app.py")
    else:
        print(f"\n❌ Training failed. Please check the dataset generation.")


if __name__ == "__main__":
    main()
