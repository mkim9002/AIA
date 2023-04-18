#데이터
#모델 : RandomForestClassifoer

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pandas as pd



#import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pandas as pd

#1 data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    # stratify=y
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {'C' : [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'degree': [2, 3, 4]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1,2,4]}
]

# 2. 모델
model = GridSearchCV(SVC(),parameters, 
                     cv=5,                #분류의 디폴드는 StratifiedKFold
                     verbose=1,
                     refit=True,
                     n_jobs=1)
#3. 컴파일 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 :", model.best_estimator_)

print("최적의 파라미터 :", model.best_params_)

print("best_score_ :", model.best_score_)

print("model_score :", model.score(x_test, y_test))

# 최적의 매개변수 : SVC(C=1, kernel='linear')
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ : 0.9916666666666668
# model_score : 1.0

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))
# accuracy_score : 1.0

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 튠 ACC :", accuracy_score(y_test, y_pred_best))
#최적의 튠 ACC : 1.0

print("걸린시간 :", round(end_time-start_time,2),'초')
# 걸린시간 : 0.33 초

####################################################################

# print(pd.DataFrame(model.cv_results_))
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=False))
print(pd.DataFrame(model.cv_results_).columns)

path = './temp'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
.to_csv(path+ 'm10_GridSearch3.csv')