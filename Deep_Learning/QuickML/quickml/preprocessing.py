"""
Data preprocessing utilities for QuickML.
Handles missing values, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including missing values,
    encoding, scaling, and feature engineering.
    """
    
    def __init__(self, target_column: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            target_column: Name of the target column (if known)
        """
        self.target_column = target_column
        self.numerical_columns = []
        self.categorical_columns = []
        self.preprocessor = None
        self.label_encoders = {}
        self.scaler = None
        self.is_fitted = False
        
    def detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numerical and categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        numerical_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return numerical_cols, categorical_cols
    
    def detect_target_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the target column if not specified.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the target column
        """
        if self.target_column:
            return self.target_column
            
        # Simple heuristic: last column is often the target
        # For more sophisticated detection, we could use domain knowledge
        # or check for binary columns, etc.
        return df.columns[-1]
    
    def detect_task_type(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Detect if the task is classification or regression.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            'classification' or 'regression'
        """
        target_values = df[target_col].dropna()
        
        # Check if target is numerical
        if not pd.api.types.is_numeric_dtype(target_values):
            return 'classification'
        
        # Check if target has few unique values (classification)
        unique_ratio = len(target_values.unique()) / len(target_values)
        if unique_ratio < 0.1:  # Less than 10% unique values
            return 'classification'
        
        return 'regression'
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Use median for numerical columns (more robust than mean)
            imputer = SimpleImputer(strategy='median')
            df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Use most frequent value for categorical columns
            imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = imputer.fit_transform(df_clean[categorical_cols])
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Use LabelEncoder for ordinal categorical variables
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def create_preprocessing_pipeline(self, numerical_cols: List[str], categorical_cols: List[str]) -> Pipeline:
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            Preprocessing pipeline
        """
        # Numerical features pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, str, str]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (if not already set)
            
        Returns:
            Tuple of (X_transformed, target_column, task_type)
        """
        # Detect target column if not provided
        if target_column:
            self.target_column = target_column
        else:
            self.target_column = self.detect_target_column(df)
        
        # Detect column types
        self.numerical_columns, self.categorical_columns = self.detect_column_types(df)
        
        # Detect task type
        task_type = self.detect_task_type(df, self.target_column)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_clean, self.categorical_columns)
        
        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(
            self.numerical_columns, 
            self.categorical_columns
        )
        
        # Transform features
        X_transformed = self.preprocessor.fit_transform(X_encoded)
        
        self.is_fitted = True
        
        return X_transformed, self.target_column, task_type
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        # Remove target column if present
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
        else:
            X = df.copy()
        
        # Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_clean, self.categorical_columns)
        
        # Transform features
        X_transformed = self.preprocessor.transform(X_encoded)
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(self.numerical_columns)
        
        # Add categorical feature names (after one-hot encoding)
        if hasattr(self.preprocessor, 'named_transformers_'):
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps'):
                encoder = cat_transformer.named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(self.categorical_columns)
                    feature_names.extend(cat_features)
        
        return feature_names
