import time
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score

#1. 데이터 
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10]},
  ]

#2. 모델 
model = HalvingRandomSearchCV(RandomForestClassifier(), parameters,
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
print("accuracy_score:", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)            
print("최적 튠 ACC:", accuracy_score(y_test, y_pred_best))

#HalvingRandomSearchCV
'''
최적의 매개변수: RandomForestClassifier(min_samples_split=5)
최적의 파라미터: {'min_samples_split': 5}
best_score: 0.953153320918684
model.score: 0.975
걸린시간 : 7.97 초
accuracy_score: 0.975
최적 튠 ACC: 0.975
'''
#HalvingGridSearchCV
'''
최적의 매개변수: RandomForestClassifier(min_samples_split=5)
최적의 파라미터: {'min_samples_split': 5}
best_score: 0.9609931719428927
model.score: 0.975
걸린시간 : 28.17 초
accuracy_score: 0.975
최적 튠 ACC: 0.975
'''
#RandomizedSearchCV
'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수: RandomForestClassifier(min_samples_split=3)
최적의 파라미터: {'min_samples_split': 3}
best_score: 0.9721544715447153
model.score: 0.9777777777777777
걸린시간 : 7.84 초
accuracy_score: 0.9777777777777777
최적 튠 ACC: 0.9777777777777777
'''
#GridSearchCV
'''
Fitting 5 folds for each of 68 candidates, totalling 340 fits
최적의 매개변수: RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
최적의 파라미터: {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 200}
best_score: 0.9728513356562137
model.score: 0.9666666666666667
걸린시간 : 29.26 초
accuracy_score: 0.9666666666666667
최적 튠 ACC: 0.9666666666666667
'''