import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pandas as pd

#1.데이터
x,y =load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2
)

n_splits =5
KFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {"C": [1,10,100,1000], "kernel":['linear'], 'degree':[3,4,5]},  #12
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma': [0.001, 0.0001]},   #12
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 
     'gamma':[0.01,0.001,0.0001],'degree':[3, 4]}      #24
        #총48회
    
]

#2. 모델
model = GridSearchCV(SVC(),parameters,
                     cv=5,
                     verbose=1,
                     refit=True,
                     n_jobs=1)

#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time =time.time()

print("최적의 매개변수 :", model.best_estimator_)

print("최적의 파라미터 :", model.best_params_)

print("best_score :", model.best_score_)

print("model_score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 튠 acc :", accuracy_score(y_test,y_pred_best))

print("걸린시간 : ", round(end_time-start_time,2),'초')

print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=False))
print(pd.DataFrame(model.cv_results_).columns)

path='./temp'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
.to_csv(path+ 'm10_GridSearch3.csv')


