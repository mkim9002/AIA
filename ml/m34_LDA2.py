import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_covtype

# Define models
models = [('Random Forest', RandomForestRegressor()), ('LDA', LinearDiscriminantAnalysis(n_components=1))]

# Define datasets
datasets = [('iris', load_iris()), ('breast cancer', load_breast_cancer()), ('diabetes', load_diabetes()), 
            ('digits', load_digits()), ('wine', load_wine()), ('covtype', fetch_covtype())]

for dataset in datasets:
    name, data = dataset
    
    # Split data into features and labels
    x, y = data.data, data.target
    
    print('Dataset:', name)
    print('Number of features:', x.shape[1])
    
    for model_name, model in models:
        print('Model:', model_name)
        
        # Fit the model
        model.fit(x, y)
        
        # Calculate the R^2 score
        score = model.score(x, y)
        
        print('R^2 score:', score)
    
    print('-' * 30)
