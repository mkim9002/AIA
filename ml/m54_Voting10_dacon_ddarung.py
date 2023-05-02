import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


#1. 데이터
path = 'c:/study/_data/ddarung/'
path_save = 'c:/study/_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
train_csv = train_csv.dropna() 

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, # stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. model
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)



model = VotingRegressor(
    estimators=[('XG',xg), ('LG',lg),('CAT',cat)],
    # voting='hard'   #default : hard
)

#3.훈련
model.fit(x_train,y_train)

#4.평가
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test,y_test))
print('Voting r2 :', r2_score(y_test,y_pred))


#hard voting 결과
#BaggingClassifier model.score :  0.9912280701754386   acc :  0.9912280701754386
#RandomForestClassifier model.score : 0.956140350877193  acc : 0.956140350877193
#DecisionTreeClassifier : model.score : 0.8947368421052632 acc : 0.8947368421052632
#VotingClassifier : model.score : 0.9824561403508771   acc : 0.9824561403508771     
# soft :   model.score : 0.9824561403508771  acc : 0.9824561403508771

regressor = [xg,lg,cat]
for model2 in regressor:
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name = model2.__class__.__name__
    print("{0} R2 : {1:.4f}".format(class_name,score2))


# model.score : 0.9824561403508771
# acc : 0.9824561403508771
# LogisticRegression 정확도 : 0.9737
# KNeighborsClassifier 정확도 : 0.9912
# DecisionTreeClassifier 정확도 : 0.9474