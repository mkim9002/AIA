import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score



# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] / 디폴트 100 / 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] / 디폴트 0.3 / 0~1 / eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] / 디폴트 6 / 0~inf / 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0 / 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] / 디폴트 1 / 0~inf
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf / L1 절대값 거즁차 규제 / alpha
# 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda

parameters = {'n_estimators' : 1000000,
              'learning_rate' : 0.5,   # 이게 성능이 가장 좋다
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 0.7,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 0.01,
              'random_state' : 1234,
              
              }

# 2. 모델
model = XGBClassifier(**parameters)
model.set_params(early_stopping_rounds=10, **parameters)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train),(x_test,y_test)],
        #   early_stopping_rounds=10,
          verbose=1,
        #   eval_metric='logloss',  #이중분류
          eval_metric='error',    #이중분류
            # eval_metric='auc',    #이중분류
            # eval_metric='merror',   #다중분류
            # eval_metric='rmse','mae','rmsle'      #회귀
            
            
            
        
          )

# model.set_params(early_stopping_rounds=10)

# 4. 평가, 예측
model.score(x_test, y_test)
print(f'result : {model.score(x_test,y_test)}')

#  best score : 0.9780219780219781
#  result : 0.9473684210526315
###################################
print("=================================")
hist = model.evals_result()
print(hist)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy score: " ,acc)

###################################
import pickle
path = 'c:/study/_save/pickle_test/'
# pickle.dump(model, open(path + 'm43_pickle1_save.dat','wb'))

import joblib
joblib.dump(model, path + 'm44_joblib1_save.dat' )





