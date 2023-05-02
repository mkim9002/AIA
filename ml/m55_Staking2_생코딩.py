import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

li = []

models = [xg,lg,cat]
for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print(y_predict.shape)
    y_predict = y_predict.reshape(y_predict.shape[0],1)
    li.append(y_predict)
    
    score = accuracy_score(y_test,y_predict)
    class_name = model.__class__.__name__
    print("{0}ACC :{1:.4f}".format(class_name,score))

# print(li)
y_stacking_predict= np.concatenate(li, axis=1)
# # print(aaa)
print(y_stacking_predict.shape)  #((114, 3))

model = CatBoostClassifier(verbose=0)
model.fit(y_stacking_predict, y_test)
score = model.score(y_stacking_predict,y_test)
print("스태킹 결과 : ", score)


