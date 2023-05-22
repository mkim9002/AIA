import autokeras as ak
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd

import time

# 1. 데이터 
path = 'd:/study/_data/dacon_diabetes/'   #점 하나 현재폴더의밑에 점하나는 스터디
path_save = 'd:/study/_save/dacon_diabetes/' 

train_csv = pd.read_csv(path + 'train.csv',
                       index_col=0) 

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 

print(test_csv)   #[116 rows x 8 columns]

print(train_csv)  #[652 rows x 9 columns]

#결측치 처리 1 .제거
# pirnt(train_csv.insul11())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(11)
print(train_csv.info())
print(train_csv.shape)

############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['Outcome'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['Outcome']
print(y)



scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x.shape, y.shape) #(652, 8) (652,)

# 2. 모델 
model = ak.StructuredDataClassifier(
    overwrite=False,  # True일 경우 모델탐색을 처음부터 다시 함(속도 느림) -> 성능이 너무 안좋을때 True사용하기 / 보통이상 성능이면 True일때 더 성능향상이 됨  
    max_trials=2,  # 디폴트 False
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.15)
end = time.time()

# 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

# 최적의 모델 저장
path = './_save/autokeras/'
best_model.save(path + "keras62_autokeras8.h5")

# 4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과:', results)
