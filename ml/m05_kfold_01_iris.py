import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1 DATA
x,y, = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, shuffle=True, random_state=123, test_size=0.2
# )


n_splits =5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = KFold()

#2. MODEL
model = LinearSVC()


#3,4. compile, epochs evaluationm prediction
scores = cross_val_score(model, x, y, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('ACC :', scores, '\n cross_val_score 평균 : ', round(np.mean(scores),4))

