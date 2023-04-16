import numpy as np
from sklearn.datasets import load_breast_cancer

x,y = load_breast_cancer(return_X_y=True)


#1. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#1. model
model = RandomForestRegressor()

# 3, compile
model.fit(x,y)

results = model.score(x,y)

print(results)