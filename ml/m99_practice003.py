#분류모델

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_digits, fetch_covtype, load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

index1 = [load_iris(return_X_y=True),load_breast_cancer(return_X_y=True),load_diabetes(return_X_y=True),load_digits(return_X_y=True),fetch_covtype(return_X_y=True),load_wine(return_X_y=True)]
index2 = [LinearSVC(max_iter=1000), LogisticRegression(max_iter=1000), DecisionTreeClassifier(max_depth=1000), RandomForestRegressor(max_depth=1000)]

scaler = MinMaxScaler()
for i in range(len(index1)):
    x, y =index1[i]
    x = scaler.fit_transform(x)
    for j in range(4):
        model = index2[j]
        model.fit(x,y)
        results = model.score(x,y)
        print(results)
        
        ##################