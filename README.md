# Flexible AutoML for Binary Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/last%20updated-2025--02--07-brightgreen)

A flexible and extensible AutoML system designed for binary classification tasks. This project provides an easy-to-use framework for automated machine learning with built-in data preprocessing, model training, and evaluation capabilities.

## Features

- 🔄 Automated data preprocessing pipeline
- 🔌 Pluggable classifier architecture
- 📊 Comprehensive metrics calculation
- 🚀 Easy to extend with new classification methods
- 📈 Automated model evaluation and comparison
- 🛠️ Built-in support for popular classifiers (Random Forest, XGBoost, Logistic Regression)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install -r requirements.txt
```

### Project structure

```python
automl/
├── base_classifier.py      # Abstract base class for classifiers
├── data_preprocessor.py    # Automated data preprocessing
├── metrics_calculator.py   # Metrics calculation utilities
├── example_classifiers.py  # Implementation of various classifiers
├── auto_ml.py             # Main AutoML orchestrator
└── example_usage.py       # Usage examples
```

### Quickstart

```

from auto_ml import AutoML
from example_classifiers import (
    RandomForestWrapper,
    XGBoostWrapper,
    LogisticRegressionWrapper
)
import pandas as pd

# Initialize AutoML
auto_ml = AutoML()

# Add classifiers
auto_ml.add_classifier(RandomForestWrapper(n_estimators=100))
auto_ml.add_classifier(XGBoostWrapper(n_estimators=100))
auto_ml.add_classifier(LogisticRegressionWrapper())

# Load your data
X = pd.read_csv("your_features.csv")
y = pd.read_csv("your_target.csv")

# Fit models and calculate metrics
auto_ml.fit(X, y)

# Display results
results = auto_ml.display_results()
print(results)

# Get best classifier
best_classifier = auto_ml.get_best_classifier(metric='f1')
print(f"Best classifier: {best_classifier.name}")

```

### Adding New Classifiers

```python
from base_classifier import BaseClassifier

class YourClassifierWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = YourModel(**kwargs)
        
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def name(self):
        return "YourClassifier"
```
