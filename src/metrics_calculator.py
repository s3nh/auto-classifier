from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

class MetricsCalculator:
    """Calculate and store various classification metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics):
        """Format metrics for display."""
        result = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Value': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics.get('roc_auc', 'N/A')
            ]
        })
        return result
