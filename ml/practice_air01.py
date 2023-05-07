import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
import time 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터
x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True, random_state=1234, test_size=0.2, stratify=y
)
n_splits =3
kfold =KFold(n_splits=n_splits,shuffle=True,random_state=337)

parameters = [
 {"C":[1,10,100,1000],"kernel":['linear'],'degree':[3,4,5]}, #12
 {'C':[1,10,100],"kernel":['rbf','linear'],'gamma':[0.01,0.0001]},    #12
 {'C':[1,10,1000],"kernel":['sigmoid'],
  'gamma':[0.01,0.001,0.0001],"degree":[3,4]},                #24
 {'C':[0.1,1],'gamma':[1,10]},
]

#2 모델
model = HalvingGridSearchCV(SVC(),parameters,cv=5,
                            verbose=1,
                            refit=True,
                            factor=3,
                            n_jobs=-1)

#3. 컴파일 훈련

start_time= time.time()
model.fit(x_train,y_train)
end_time=time.time()


print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)
print("best_score :", model.best_score_)
print('model.score :',model.score(x_test,y_test))

y_predict = model.predict(x_test)

y_pred_best = model.best_estimator_.predict(x_test)
print("acc 최적튠 : ",accuracy_score(y_test,y_pred_best))

print("걸린시간 :", round(end_time-start_time,2), "초")




