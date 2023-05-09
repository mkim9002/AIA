
import time 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

#1. 데이터
x,y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test =train_test_split(
    x,y,shuffle=True, random_state=42,test_size=0.2,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=337)

parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10]},
  ]

#2.모델
model = RandomizedSearchCV(RandomForestClassifier(), parameters, 
                           cv=kfold,verbose=1, refit=True, n_jobs=-1)

#3.컴파일 훈련
start_time=time.time()
model.fit(x_train,x_test)
end_time=time.time()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :", model.best_score_)
print("model.score :",model.score(x_test,y_test))
print("걸린시간 :", (end_time-start_time,2),"초")

#4. 평가 예측
y_predict =model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 튠 acc :", accuracy_score(y_test,y_pred_best))