#회귀로 만들어
#회귀 데이터 올인 -포문
#scaler 6개 올인- 포문

#정규분포로 만들고, 분위수를 기준으로 0-1 사이로 만들기 떄문에
#이상치에 자유롭다.


from sklearn.datasets import fetch_california_housing,load_iris, load_breast_cancer
from sklearn.datasets import load_wine, fetch_covtype, load_digits, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

#1. data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,  random_state=337, train_size=0.8,
    stratify=y,
)

scaler = QuantileTransformer(n_quantiles=1000)  #디폴트 / 분위수조절
scaler = QuantileTransformer(n_quantiles=10)
scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler = PowerTransformer(method='box-cox')      #양수만 사용가능
scaler = PowerTransformer(method='yeo-johnson')  #디폴트




x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
model = RandomForestClassifier()

#.3 훈련

model.fit(x_train,y_train)

#4. 평가 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test,y_test))
print('acc :', accuracy_score(y_test,y_pred))





