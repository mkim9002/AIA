import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import r2_score, accuracy_score


#1. data
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_wine(return_X_y=True)]

model_list =[LinearSVC(),
             LogisticRegression(),
             DecisionTreeClassifier(),
             RandomForestRegressor()]


data_name_list = ['아이리스 :',
                  '브레스트_캔서 :',
                  '와인 :',]


model_name_list = ['LinearSVC :',
                   'LogisticRegression :',
                   'DecisionTreeClassfier :',
                   'RF: ',]


#2 model
for i , value in enumerate(data_list):
    x,y = value
    # print(x.shape, y.shape)
    print("==================")
    print(data_name_list[i])
    
    for j, value2 in  enumerate(model_list):
        model = value2
        #3. 콤파일 훈련
        model.fit(x,y)
        #4. 평가 예측
        results = model.score(x,y)
        print(model_name_list[j], results)
        y_predict = model.predict(x)
        acc = accuracy_score(y, y_predict)
        print(model_name_list[j], "accuracy_score :",acc)
        


    
    
    
    
    # x,y = i
    # print("==================")
    # print(x.shape, y.shape)
    
    
    # model = LinearSVC()
    # model.fit(x, y)
    # results = model.score(x,y)
    # print(results)
    
    

