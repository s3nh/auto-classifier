from base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import numpy as np

# ... (previous classifier implementations remain the same) ...

class TabPFNWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = TabPFNClassifier(
            device='cpu',  # Change to 'cuda' for GPU support
            N_ensemble_configurations=3,  # Number of models in ensemble
            **kwargs
        )
        self._is_fitted = False
        
    def fit(self, X, y):
        # TabPFN expects numpy arrays
        X = self._ensure_numpy(X)
        y = self._ensure_numpy(y)
        
        # TabPFN has a unique characteristic where fit() is not really needed
        # as it uses pre-trained neural networks. However, we store the training
        # data for later use in predict
        self.X_train = X
        self.y_train = y
        self._is_fitted = True
        return self
    
    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._ensure_numpy(X)
        return self.model.predict(X, self.X_train, self.y_train)
    
    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._ensure_numpy(X)
        return self.model.predict_proba(X, self.X_train, self.y_train)
    
    @property
    def name(self):
        return "TabPFN"
        
    @staticmethod
    def _ensure_numpy(X):
        """Convert input to numpy array if needed."""
        if isinstance(X, np.ndarray):
            return X
        return X.to_numpy()
