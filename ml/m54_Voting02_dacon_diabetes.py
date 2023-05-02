#실습
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier



#1. 데이터
path = 'c:/study/_data/dacon_diabetes/'
path_save = 'c:/study/_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
print(train_csv)  
# [652 rows x 9 columns] #(652,9)

test_csv= pd.read_csv(path+'test.csv', index_col=0)
print(test_csv) 
#(116,8) #outcome제외

# print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 10000,
              'learning_rate' : 0.01,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 0,
              'subsample': 0.4,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 0,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state' : 123,
            #   'eval_metric' : 'error'
              }
#2. model
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()




model = VotingClassifier(
    estimators=[('LR',lr), ('KNN',knn),('DT',dt)],
    voting='soft'   #default
)

#3.훈련
model.fit(x_train,y_train)

#4.평가
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test,y_test))
print('Voting acc :', accuracy_score(y_test,y_pred))


#hard voting 결과
#BaggingClassifier model.score :  0.9912280701754386   acc :  0.9912280701754386
#RandomForestClassifier model.score : 0.956140350877193  acc : 0.956140350877193
#DecisionTreeClassifier : model.score : 0.8947368421052632 acc : 0.8947368421052632
#VotingClassifier : model.score : 0.9824561403508771   acc : 0.9824561403508771     
# soft :   model.score : 0.9824561403508771  acc : 0.9824561403508771

classifiers = [lr,knn,dt]
for model2 in classifiers:
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name,score2))


# model.score : 0.9824561403508771
# acc : 0.9824561403508771
# LogisticRegression 정확도 : 0.9737
# KNeighborsClassifier 정확도 : 0.9912
# DecisionTreeClassifier 정확도 : 0.9474