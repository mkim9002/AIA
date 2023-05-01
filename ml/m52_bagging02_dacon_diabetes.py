import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaa = LogisticRegression()
model = BaggingClassifier(aaa,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True,     # 디폴트
                        #   bootstrap=False,
                          
                          )

#3.훈련
model.fit(x_train,y_train)

#4.평가
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test,y_test))
print('acc :', accuracy_score(y_test,y_pred))




