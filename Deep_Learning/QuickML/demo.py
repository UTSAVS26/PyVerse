#!/usr/bin/env python3
"""
QuickML Demonstration Script
Shows how to use QuickML programmatically with different datasets.
"""

import pandas as pd
import numpy as np
from quickml import QuickML
import os

def demo_classification():
    """Demonstrate QuickML with classification data."""
    print("=" * 60)
    print("🎯 QUICKML CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Load customer churn dataset
    df = pd.read_csv('sample_data/customer_churn.csv')
    print(f"📊 Dataset: Customer Churn")
    print(f"   Shape: {df.shape}")
    print(f"   Target: churn (binary classification)")
    print(f"   Features: {len(df.columns) - 1}")
    
    # Initialize QuickML
    quickml = QuickML()
    
    # Fit the model
    print("\n🚀 Training models...")
    results = quickml.fit(df, target_column='churn', cv_folds=5)
    
    # Display results
    print(f"\n🏆 Results:")
    print(f"   Best Model: {results['best_model_name']}")
    print(f"   Score: {results['best_score']:.4f}")
    print(f"   Task Type: {results['task_type']}")
    
    # Show all model scores
    print(f"\n📈 All Model Scores:")
    for model, score in results['cv_scores'].items():
        print(f"   {model}: {score:.4f}")
    
    # Make predictions
    sample_data = df.head(5)
    predictions = quickml.predict(sample_data)
    probabilities = quickml.predict_proba(sample_data)
    
    print(f"\n🔮 Sample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"   Sample {i+1}: Predicted={pred}, Probability={prob[1]:.3f}")
    
    # Feature importance
    importance = quickml.get_feature_importance()
    if importance is not None:
        feature_names = quickml.feature_names
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Top 5 Feature Importance:")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model
    quickml.save_model('demo_classification_model.pkl')
    print(f"\n💾 Model saved to: demo_classification_model.pkl")


def demo_regression():
    """Demonstrate QuickML with regression data."""
    print("\n" + "=" * 60)
    print("📈 QUICKML REGRESSION DEMO")
    print("=" * 60)
    
    # Load house prices dataset
    df = pd.read_csv('sample_data/house_prices.csv')
    print(f"📊 Dataset: House Prices")
    print(f"   Shape: {df.shape}")
    print(f"   Target: price (regression)")
    print(f"   Features: {len(df.columns) - 1}")
    
    # Initialize QuickML
    quickml = QuickML()
    
    # Fit the model
    print("\n🚀 Training models...")
    results = quickml.fit(df, target_column='price', cv_folds=5)
    
    # Display results
    print(f"\n🏆 Results:")
    print(f"   Best Model: {results['best_model_name']}")
    print(f"   Score: {results['best_score']:.4f}")
    print(f"   Task Type: {results['task_type']}")
    
    # Show all model scores
    print(f"\n📈 All Model Scores:")
    for model, score in results['cv_scores'].items():
        print(f"   {model}: {score:.4f}")
    
    # Make predictions
    sample_data = df.head(5)
    predictions = quickml.predict(sample_data)
    
    print(f"\n🔮 Sample Predictions:")
    for i, (actual, pred) in enumerate(zip(sample_data['price'], predictions)):
        print(f"   Sample {i+1}: Actual=${actual:,.0f}, Predicted=${pred:,.0f}")
    
    # Feature importance
    importance = quickml.get_feature_importance()
    if importance is not None:
        feature_names = quickml.feature_names
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Top 5 Feature Importance:")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model
    quickml.save_model('demo_regression_model.pkl')
    print(f"\n💾 Model saved to: demo_regression_model.pkl")


def demo_model_loading():
    """Demonstrate loading and using saved models."""
    print("\n" + "=" * 60)
    print("📂 QUICKML MODEL LOADING DEMO")
    print("=" * 60)
    
    # Load the saved classification model
    if os.path.exists('demo_classification_model.pkl'):
        print("📂 Loading saved classification model...")
        quickml = QuickML()
        quickml.load_model('demo_classification_model.pkl')
        
        # Load some test data
        df = pd.read_csv('sample_data/customer_churn.csv')
        test_data = df.tail(3)
        
        # Make predictions
        predictions = quickml.predict(test_data)
        probabilities = quickml.predict_proba(test_data)
        
        print(f"🔮 Predictions on new data:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"   Sample {i+1}: Predicted={pred}, Probability={prob[1]:.3f}")
    
    # Load the saved regression model
    if os.path.exists('demo_regression_model.pkl'):
        print("\n📂 Loading saved regression model...")
        quickml = QuickML()
        quickml.load_model('demo_regression_model.pkl')
        
        # Load some test data
        df = pd.read_csv('sample_data/house_prices.csv')
        test_data = df.tail(3)
        
        # Make predictions
        predictions = quickml.predict(test_data)
        
        print(f"🔮 Predictions on new data:")
        for i, (actual, pred) in enumerate(zip(test_data['price'], predictions)):
            print(f"   Sample {i+1}: Actual=${actual:,.0f}, Predicted=${pred:,.0f}")


def main():
    """Run all demonstrations."""
    print("🚀 QuickML AutoML Engine - Demonstration")
    print("=" * 60)
    
    # Check if sample data exists
    if not os.path.exists('sample_data'):
        print("❌ Sample data not found. Please run 'python sample_data.py' first.")
        return
    
    try:
        # Run demonstrations
        demo_classification()
        demo_regression()
        demo_model_loading()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60)
        
        print("\n🎯 QuickML Features Demonstrated:")
        print("   ✅ Automatic data preprocessing")
        print("   ✅ Multiple model training")
        print("   ✅ Model comparison and selection")
        print("   ✅ Feature importance analysis")
        print("   ✅ Model saving and loading")
        print("   ✅ Predictions on new data")
        
        print("\n🔧 Next Steps:")
        print("   1. Try the web interface: streamlit run app.py")
        print("   2. Use the CLI: python quickml.py --data your_data.csv")
        print("   3. Integrate into your projects: from quickml import QuickML")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")


if __name__ == "__main__":
    main()
