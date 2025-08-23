#!/usr/bin/env python3
"""
Command line interface for QuickML AutoML Engine.
Provides a simple CLI for running QuickML analysis on CSV files.
"""

import argparse
import pandas as pd
import sys
import os
from pathlib import Path
import warnings

from quickml import QuickML

warnings.filterwarnings('ignore')


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="QuickML - Mini AutoML Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detection
  python quickml.py --data dataset.csv
  
  # Specify target column
  python quickml.py --data dataset.csv --target target_column
  
  # Specify task type
  python quickml.py --data dataset.csv --task classification
  
  # Save model to specific file
  python quickml.py --data dataset.csv --output my_model.pkl
  
  # Use different number of CV folds
  python quickml.py --data dataset.csv --cv-folds 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to CSV data file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--target', '-t',
        help='Target column name (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--task',
        choices=['classification', 'regression', 'auto'],
        default='auto',
        help='Task type (default: auto-detect)'
    )
    
    parser.add_argument(
        '--cv-folds', '-cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='best_model.pkl',
        help='Output file for saved model (default: best_model.pkl)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots to files'
    )
    
    parser.add_argument(
        '--plots-dir',
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file '{args.data}' not found.")
        sys.exit(1)
    
    try:
        # Load data
        print(f"📊 Loading data from: {args.data}")
        df = pd.read_csv(args.data)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        
        # Initialize QuickML
        task_type = None if args.task == 'auto' else args.task
        quickml = QuickML(
            target_column=args.target,
            task_type=task_type
        )
        
        # Run analysis
        print("\n🚀 Starting QuickML AutoML Pipeline...")
        results = quickml.fit(df, cv_folds=args.cv_folds)
        
        # Display results
        print("\n" + "="*60)
        print("🎉 QUICKML ANALYSIS COMPLETE!")
        print("="*60)
        
        print(f"\n📊 Dataset Info:")
        print(f"   Shape: {results['data_shape']}")
        print(f"   Target column: {results['target_column']}")
        print(f"   Task type: {results['task_type']}")
        print(f"   Transformed features: {results['transformed_shape'][1]}")
        
        print(f"\n🏆 Best Model:")
        print(f"   Model: {results['best_model_name']}")
        print(f"   Score: {results['best_score']:.4f}")
        
        # Display all model scores
        print(f"\n📈 All Model Scores:")
        for model_name, score in results['cv_scores'].items():
            print(f"   {model_name}: {score:.4f}")
        
        # Display evaluation metrics
        if 'evaluation_metrics' in results:
            print(f"\n📊 Evaluation Metrics:")
            for metric, value in results['evaluation_metrics'].items():
                if metric not in ['test_size', 'train_size']:
                    if isinstance(value, float):
                        print(f"   {metric}: {value:.4f}")
                    else:
                        print(f"   {metric}: {value}")
        
        # Save model
        print(f"\n💾 Saving model to: {args.output}")
        quickml.save_model(args.output)
        
        # Create visualizations if requested
        if args.save_plots:
            print(f"\n📊 Creating visualizations...")
            plots = quickml.create_visualizations(df, save_plots=True, output_dir=args.plots_dir)
            print(f"✅ Plots saved to: {args.plots_dir}/")
        
        # Feature importance
        feature_importance = quickml.get_feature_importance()
        if feature_importance is not None:
            print(f"\n📈 Top 5 Feature Importance:")
            importance_df = pd.DataFrame({
                'Feature': quickml.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
                print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\n✅ QuickML analysis completed successfully!")
        print(f"📁 Model saved to: {args.output}")
        
        if args.save_plots:
            print(f"📊 Plots saved to: {args.plots_dir}/")
        
        print(f"\n🔧 To use the saved model:")
        print(f"   from quickml import QuickML")
        print(f"   quickml = QuickML()")
        print(f"   quickml.load_model('{args.output}')")
        print(f"   predictions = quickml.predict(new_data)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
