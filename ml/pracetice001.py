import numpy as np
from sklearn.datasets import load_breast_cancer

#1.데이터
x,y =load_breast_cancer(return_X_y=True)

print(x.shape, y.shape) #(569, 30) (569,)

#2, 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model =LinearSVC() #0.9209138840070299
# model =LogisticRegression() #0.9472759226713533
# model = DecisionTreeClassifier() #1.0
model = RandomForestClassifier()

# 3.컴파일 훈련
model.fit(x,y)

#4, 훈련 평가
results = model.score(x,y)

print(results)
