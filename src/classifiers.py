from base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

class RandomForestWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def name(self):
        return "RandomForest"

class XGBoostWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def name(self):
        return "XGBoost"

class LogisticRegressionWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
        
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def name(self):
        return "LogisticRegression"
