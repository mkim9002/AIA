from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np

# 1. 데이터
iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.2,stratify=y)

# kf = KFold(n_splits=5, shuffle=True, random_state=337)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=337)#알고리즘

# 2. 모델
model = RandomForestClassifier()

# 3. 교차 검증 평가
scores = cross_val_score(model, x_train, y_train, cv=kf)
print('cross_val_score :', scores)
print('교차 검증 평균 점수 :', round(np.mean(scores), 4))

# 4. 예측 및 평가
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('ACC:', acc) 
# print("============================================")
# print(np.unique(y_train,return_counts=True))
# print(np.unique(y_test,return_counts=True))

#kf = KFold =ACC: 0.9333333333333333
#kf = StratifiedKFold = ACC: 0.9666666666666667