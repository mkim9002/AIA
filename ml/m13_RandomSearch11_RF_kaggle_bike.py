#실습 
#모델 : RandomForestClassifier
# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6,8,10,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1, 2, 4]}]
#파라미터 조합으로 2개 이상 엮을 것
####################################################
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
# print(train_csv.isnull().sum()) 
#결측치 없음

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10]},
  ]

#2. 모델 
model = RandomizedSearchCV(RandomForestRegressor(), parameters,
                     cv=kfold, verbose=1, refit=True, n_jobs=-1)

#3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수:", model.best_estimator_) 
print("최적의 파라미터:", model.best_params_)
print("best_score:", model.best_score_)
print("model.score:", model.score(x_test, y_test))
print("걸린시간 :", round(end_time-start_time,2), "초")

#4. 평가, 예측
y_predict = model.predict(x_test)
print("r2_score:", r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)            
print("최적 튠 r2:", r2_score(y_test, y_pred_best))

'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수: RandomForestRegressor(max_depth=10, min_samples_leaf=3)
최적의 파라미터: {'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 10}
best_score: 0.3505597557437688
model.score: 0.3698332799333316
걸린시간 : 17.89 초
r2_score: 0.3698332799333316
최적 튠 r2: 0.3698332799333316
'''
#
'''
Fitting 5 folds for each of 30 candidates, totalling 150 fits
최적의 매개변수: RandomForestRegressor(max_depth=10)
최적의 파라미터: {'max_depth': 10}
best_score: 0.35059194460571
model.score: 0.37125646931958367
걸린시간 : 96.35 초
r2_score: 0.37125646931958367
최적 튠 r2: 0.37125646931958367
'''