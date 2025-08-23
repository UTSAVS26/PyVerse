"""
Sample data generator for QuickML testing and demonstration.
Creates various sample datasets for classification and regression tasks.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import os

def create_classification_dataset(n_samples=1000, n_features=10, n_informative=5, 
                                n_redundant=3, n_classes=2, random_state=42):
    """
    Create a sample classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_classes: Number of classes
        random_state: Random seed
        
    Returns:
        DataFrame with classification data
    """
    np.random.seed(random_state)
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    df['category_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
    df['category_2'] = np.random.choice(['X', 'Y'], n_samples)
    
    # Add target column
    df['target'] = y
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'feature_1'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'category_1'] = np.nan
    
    return df

def create_regression_dataset(n_samples=1000, n_features=10, n_informative=5, 
                            n_targets=1, random_state=42):
    """
    Create a sample regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        n_targets: Number of target variables
        random_state: Random seed
        
    Returns:
        DataFrame with regression data
    """
    np.random.seed(random_state)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    df['category_1'] = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    df['category_2'] = np.random.choice(['Type_A', 'Type_B'], n_samples)
    
    # Add target column
    df['target'] = y.flatten()
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'feature_1'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'category_1'] = np.nan
    
    return df

def create_customer_churn_dataset(n_samples=1000, random_state=42):
    """
    Create a realistic customer churn dataset.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        DataFrame with customer churn data
    """
    np.random.seed(random_state)
    
    # Generate customer data
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'tenure': np.random.exponential(5, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (churn) based on some features
    churn_prob = (
        (df['tenure'] < 10) * 0.3 +
        (df['monthly_charges'] > 80) * 0.2 +
        (df['contract_type'] == 'Month-to-month') * 0.3 +
        (df['payment_method'] == 'Electronic check') * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['churn'] = (churn_prob > 0.5).astype(int)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_indices, 'total_charges'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    df.loc[missing_indices, 'online_security'] = np.nan
    
    return df

def create_house_prices_dataset(n_samples=1000, random_state=42):
    """
    Create a realistic house prices dataset.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        DataFrame with house prices data
    """
    np.random.seed(random_state)
    
    # Generate house data
    data = {
        'house_id': range(1, n_samples + 1),
        'square_feet': np.random.normal(2000, 500, n_samples).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'lot_size': np.random.normal(8000, 2000, n_samples).astype(int),
        'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.5, 0.25, 0.05]),
        'property_type': np.random.choice(['Single Family', 'Townhouse', 'Condo'], n_samples, p=[0.7, 0.2, 0.1]),
        'neighborhood': np.random.choice(['Downtown', 'Suburban', 'Rural'], n_samples, p=[0.3, 0.5, 0.2]),
        'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples, p=[0.2, 0.5, 0.25, 0.05]),
        'heating_type': np.random.choice(['Gas', 'Electric', 'Oil', 'Heat Pump'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'cooling_type': np.random.choice(['Central Air', 'Window Units', 'None'], n_samples, p=[0.8, 0.15, 0.05]),
        'has_pool': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'has_fireplace': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'has_basement': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (price) based on features
    base_price = 200000
    price = (
        base_price +
        df['square_feet'] * 100 +
        df['bedrooms'] * 15000 +
        df['bathrooms'] * 20000 +
        (df['year_built'] - 1950) * 1000 +
        df['lot_size'] * 5 +
        df['garage_spaces'] * 10000 +
        (df['property_type'] == 'Single Family') * 50000 +
        (df['neighborhood'] == 'Downtown') * 30000 +
        (df['condition'] == 'Excellent') * 50000 +
        (df['has_pool'] == 'Yes') * 25000 +
        (df['has_fireplace'] == 'Yes') * 15000 +
        (df['has_basement'] == 'Yes') * 20000 +
        np.random.normal(0, 20000, n_samples)
    )
    
    df['price'] = price.astype(int)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'lot_size'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    df.loc[missing_indices, 'condition'] = np.nan
    
    return df

def save_sample_datasets():
    """Create and save all sample datasets."""
    
    # Create output directory
    os.makedirs('sample_data', exist_ok=True)
    
    print("Creating sample datasets...")
    
    # Classification datasets
    print("Creating classification dataset...")
    classification_df = create_classification_dataset(n_samples=1000)
    classification_df.to_csv('sample_data/classification_data.csv', index=False)
    
    print("Creating customer churn dataset...")
    churn_df = create_customer_churn_dataset(n_samples=1000)
    churn_df.to_csv('sample_data/customer_churn.csv', index=False)
    
    # Regression datasets
    print("Creating regression dataset...")
    regression_df = create_regression_dataset(n_samples=1000)
    regression_df.to_csv('sample_data/regression_data.csv', index=False)
    
    print("Creating house prices dataset...")
    house_df = create_house_prices_dataset(n_samples=1000)
    house_df.to_csv('sample_data/house_prices.csv', index=False)
    
    print("✅ Sample datasets created successfully!")
    print("\nAvailable datasets:")
    print("- sample_data/classification_data.csv (Binary classification)")
    print("- sample_data/customer_churn.csv (Customer churn prediction)")
    print("- sample_data/regression_data.csv (Regression)")
    print("- sample_data/house_prices.csv (House price prediction)")
    
    # Print dataset info
    print("\nDataset Information:")
    for filename in ['classification_data.csv', 'customer_churn.csv', 'regression_data.csv', 'house_prices.csv']:
        filepath = f'sample_data/{filename}'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"- {filename}: {df.shape[0]} samples, {df.shape[1]} features")

if __name__ == "__main__":
    save_sample_datasets()
