from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self):
        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
        
    def create_preprocessing_pipeline(self, X):
        """Create preprocessing pipeline based on data types."""
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        self.numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', pd.get_dummies)
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_pipeline, numerical_features),
                ('cat', self.categorical_pipeline, categorical_features)
            ])
        
        return self.preprocessor
    
    def fit_transform(self, X):
        """Fit preprocessor and transform data."""
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(X)
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X):
        """Transform data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.preprocessor.transform(X)
