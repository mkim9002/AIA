import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#1 data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, stratify=y
)

gamma =[0.001, 0.01, 0.1, 1, 10, 100]
C =[0.001, 0.01, 0.1, 1, 10, 100]
max_score = 0
for i in gamma:
    for j in C:
        #.model
        model = SVC(gamma=i, C=j)

        #3, 컴파일 훈련
        model.fit(x_train, y_train)

        #4. 평가 예측
        score = model.score(x_test, y_test)
        

        if max_score < score:
            max_score = score
            best_parameters = {'gamma' : i, 'C' : j}


print("최고점수 : ", max_score)
print("최적의 매개변수 : ", best_parameters)

#acc :  0.9666666666666667
#최고점수 :  1.0
# 최적의 매개변수 :  {'gamma': 10, 'C': 1}