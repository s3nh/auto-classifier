from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

class BaseClassifier(ABC, BaseEstimator):
    """Base class for all classifiers in the AutoML system."""
    
    @abstractmethod
    def fit(self, X, y):
        """Train the classifier."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities."""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return classifier name."""
        pass
