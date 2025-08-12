"""
Eye Strain Predictor - Dataset Generation Module

This module generates a synthetic dataset for training the eye strain prediction model.
The dataset includes realistic relationships between screen usage patterns and eye strain risk.

Features generated:
- Personal factors: age, sleep quality, previous eye problems
- Screen usage: daily hours, brightness, distance
- Environment: room lighting, blink rate
- Habits: break frequency, blue light filter usage, eye exercises

Author: AI Assistant
Date: 2025
License: MIT
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_eye_strain_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic eye strain dataset based on realistic factors and relationships.
    
    Args:
        n_samples (int): Number of samples to generate (default: 2000)
        
    Returns:
        pd.DataFrame: Generated dataset with features and target variable
    """
    
    data = []
    
    for _ in range(n_samples):
        # Generate base features
        age = np.random.normal(25, 8)  # Age centered around 25
        age = max(16, min(65, age))  # Clamp between 16-65
        
        # Screen time (more for younger people, professionals)
        base_screen_time = np.random.exponential(6) + 2  # 2-16 hours typically
        screen_time = min(16, base_screen_time)
        
        # Screen brightness (0-100%)
        screen_brightness = np.random.uniform(20, 100)
        
        # Distance from screen (30-80 cm)
        screen_distance = np.random.normal(50, 15)
        screen_distance = max(20, min(100, screen_distance))
        
        # Room lighting (categorical: 0=dim, 1=moderate, 2=bright)
        room_lighting = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        
        # Blink rate (normal: 15-20 blinks/min, lower with screen use)
        base_blink_rate = np.random.normal(16, 3)
        # Reduce blink rate with more screen time
        blink_rate = base_blink_rate * (1 - screen_time * 0.02)
        blink_rate = max(5, min(25, blink_rate))
        
        # Break frequency (breaks per hour)
        break_frequency = np.random.exponential(2)
        break_frequency = min(10, break_frequency)
        
        # Sleep quality (1-5 scale)
        sleep_quality = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Blue light filter usage (0=no, 1=yes)
        blue_light_filter = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Eye exercises (0=no, 1=yes)
        eye_exercises = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Previous eye problems (0=no, 1=yes)
        previous_eye_problems = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Calculate eye strain level based on factors
        strain_score = 0
        
        # Screen time impact
        strain_score += screen_time * 0.3
        
        # Brightness impact (too bright or too dim is bad)
        if screen_brightness < 30 or screen_brightness > 80:
            strain_score += 1.5
        
        # Distance impact (too close is bad)
        if screen_distance < 40:
            strain_score += 2
        
        # Room lighting impact
        if room_lighting == 0:  # dim
            strain_score += 1.5
        elif room_lighting == 2:  # too bright
            strain_score += 0.5
        
        # Blink rate impact
        if blink_rate < 12:
            strain_score += 2
        
        # Break frequency impact
        strain_score -= break_frequency * 0.3
        
        # Sleep quality impact
        strain_score -= (sleep_quality - 3) * 0.5
        
        # Blue light filter reduces strain
        if blue_light_filter:
            strain_score -= 1
        
        # Eye exercises reduce strain
        if eye_exercises:
            strain_score -= 1.5
        
        # Previous problems increase susceptibility
        if previous_eye_problems:
            strain_score += 2
        
        # Age factor (older people more susceptible)
        if age > 40:
            strain_score += 1
        
        # Add some randomness
        strain_score += np.random.normal(0, 1)
        
        # Convert to categorical levels
        if strain_score <= 2:
            eye_strain_level = 0  # None
        elif strain_score <= 5:
            eye_strain_level = 1  # Mild
        elif strain_score <= 8:
            eye_strain_level = 2  # Moderate
        else:
            eye_strain_level = 3  # Severe
        
        # Append to dataset
        data.append({
            'age': round(age, 1),
            'screen_time_hours': round(screen_time, 1),
            'screen_brightness_percent': round(screen_brightness, 1),
            'screen_distance_cm': round(screen_distance, 1),
            'room_lighting': room_lighting,  # 0=dim, 1=moderate, 2=bright
            'blink_rate_per_min': round(blink_rate, 1),
            'break_frequency_per_hour': round(break_frequency, 1),
            'sleep_quality': sleep_quality,  # 1-5 scale
            'blue_light_filter': blue_light_filter,  # 0=no, 1=yes
            'eye_exercises': eye_exercises,  # 0=no, 1=yes
            'previous_eye_problems': previous_eye_problems,  # 0=no, 1=yes
            'eye_strain_level': eye_strain_level  # 0=none, 1=mild, 2=moderate, 3=severe
        })
    
    df = pd.DataFrame(data)
    
    # Add some data validation
    print("Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\nEye Strain Level Distribution:")
    print(df['eye_strain_level'].value_counts().sort_index())
    print(f"\nFeature ranges:")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"{col}: {df[col].min():.1f} - {df[col].max():.1f}")
    
    return df

def main() -> None:
    """
    Main function to generate and save the eye strain dataset.
    """
    print("🚀 Generating Eye Strain Dataset...")
    print("=" * 40)
    
    # Generate dataset
    print("📊 Creating synthetic data with realistic relationships...")
    dataset = generate_eye_strain_dataset(2000)
    
    # Save to CSV
    print(f"\n💾 Saving dataset...")
    dataset.to_csv('eye_strain_dataset.csv', index=False)
    print(f"✅ Dataset saved as 'eye_strain_dataset.csv'")
    print(f"📏 Final dataset shape: {dataset.shape}")
    print(f"\n🎯 Dataset ready for model training!")
    print(f"▶️  Next step: Run 'python train_model.py'")


if __name__ == "__main__":
    main()
