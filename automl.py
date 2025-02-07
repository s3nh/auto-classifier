from data_preprocessor import DataPreprocessor
from metrics_calculator import MetricsCalculator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class AutoML:
    """Main AutoML class that orchestrates the entire process."""
    
    def __init__(self, classifiers=None):
        self.preprocessor = DataPreprocessor()
        self.metrics_calculator = MetricsCalculator()
        self.classifiers = classifiers if classifiers else []
        self.results = {}
        
    def add_classifier(self, classifier):
        """Add a new classifier to the AutoML system."""
        self.classifiers.append(classifier)
        
    def fit(self, X, y, test_size=0.2, random_state=42):
        """Fit all classifiers and calculate metrics."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Train and evaluate each classifier
        for classifier in self.classifiers:
            # Train
            classifier.fit(X_train_processed, y_train)
            
            # Predict
            y_pred = classifier.predict(X_test_processed)
            y_prob = classifier.predict_proba(X_test_processed)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                y_test, y_pred, y_prob
            )
            
            self.results[classifier.name] = {
                'classifier': classifier,
                'metrics': metrics
            }
    
    def get_best_classifier(self, metric='f1'):
        """Return the best classifier based on the specified metric."""
        best_score = -1
        best_classifier = None
        
        for name, result in self.results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_classifier = result['classifier']
                
        return best_classifier
    
    def display_results(self):
        """Display results for all classifiers."""
        all_results = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            formatted_metrics = self.metrics_calculator.format_metrics(metrics)
            formatted_metrics['Classifier'] = name
            all_results.append(formatted_metrics)
            
        return pd.concat(all_results, ignore_index=True)
