import numpy as np
from sklearn.datasets import load_iris

# 1. datasets
#datasets = load_iris()
#x = datasets.data
#y = datasets ['target']
x, y = load_iris(return_X_y=True)

print(x.shape, y.shape) #(150, 4) (150,)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor

#3. compile
model.fit(x, y)

#4. evaluate, prediction
results = model.score(x,y)

print(results)