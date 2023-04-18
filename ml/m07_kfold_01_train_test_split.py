import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# 1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=337, test_size=0.2)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=337)

# 2. 모델
model = SVC()

# 3, 4 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x_train, y_train, cv=kf)
print('cross_val_score : ', score, '\n mean of cross_val_score', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
acc = ('cross_val_preidct acc : ', accuracy_score(y_test, y_predict))