import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_covtype
from sklearn.metrics import accuracy_score

# Define models
models = [('Random Forest', RandomForestClassifier()), ('LDA', LinearDiscriminantAnalysis(n_components=1))]

# Define datasets
datasets = [('iris', load_iris()), ('breast cancer', load_breast_cancer()), ('diabetes', load_diabetes()), 
            ('digits', load_digits()), ('wine', load_wine()), ('covtype', fetch_covtype())]

for dataset in datasets:
    name, data = dataset
    
    # Split data into features and labels
    x, y = data.data, data.target
    
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    print('Dataset:', name)
    print('Number of features:', x.shape[1])
    
    for model_name, model in models:
        print('Model:', model_name)
        
        # Fit the model on the training set
        model.fit(x_train, y_train)
        
        # Calculate the accuracy on the training set
        train_acc = accuracy_score(y_train, model.predict(x_train))
        
        # Calculate the accuracy on the test set
        test_acc = accuracy_score(y_test, model.predict(x_test))
        
        print('Training accuracy:', train_acc)
        print('Test accuracy:', test_acc)
    
    print('-' * 30)

# Dataset: iris
# Number of features: 4
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 1.0
# Model: LDA
# Training accuracy: 0.975
# Test accuracy: 1.0
# ------------------------------
# Dataset: breast cancer
# Number of features: 30
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 0.9649122807017544
# Model: LDA
# Training accuracy: 0.9648351648351648
# Test accuracy: 0.956140350877193
# ------------------------------
# Dataset: diabetes
# Number of features: 10
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 0.0
# Model: LDA
# Training accuracy: 0.5014164305949008
# Test accuracy: 0.0
# ------------------------------
# Dataset: digits
# Number of features: 64
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 0.975
# Model: LDA
# Training accuracy: 0.9693806541405706
# Test accuracy: 0.9444444444444444
# ------------------------------
# Dataset: wine
# Number of features: 13
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 1.0
# Model: LDA
# Training accuracy: 1.0
# Test accuracy: 1.0
# Dataset: covtype
# Number of features: 54
# Model: Random Forest
# Training accuracy: 1.0
# Test accuracy: 0.9549064998321902
# Model: LDA
# Training accuracy: 0.6797286627410398
# Test accuracy: 0.6782785298142044    
# ------------------------------   