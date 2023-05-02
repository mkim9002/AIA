import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

datasets = [load_iris, load_breast_cancer, load_wine, load_digits]

for dataset in datasets:
    #1. 데이터
    x, y = dataset(return_X_y=True)

    pf = PolynomialFeatures(degree=3)
    x_pf = pf.fit_transform(x)
    print(x_pf.shape) #(150, 15)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #2. model
    model = RandomForestClassifier()

    #3.훈련
    model.fit(x_train,y_train)

    #4.평가
    y_pred = model.predict(x_test)
    print('model.score :', model.score(x_test,y_test))
    print('Voting acc :', accuracy_score(y_test,y_pred))
