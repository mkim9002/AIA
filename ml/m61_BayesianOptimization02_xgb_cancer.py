from bayes_opt import BayesianOptimization
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time
from xgboost import XGBClassifier


# data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model

bayesian_params ={
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}


def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample,
              colsample_bytree, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 1000,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),                 #무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                              #0-1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'reg_lambda' : max(reg_lambda,0),                            #무조건 양수만
        'reg_alpha' : reg_alpha
    }
    
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=337)

start_time = time.time()
n_iter = 500
xgb_bo.maximize(init_points=5, n_iter=100)
end_time = time.time()
print(xgb_bo.max)
print(n_iter, "번 걸린시간 :", end_time-start_time)


