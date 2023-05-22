import autokeras as ak
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

import time

# 1. 데이터 
datasets = load_digits()
# print(datasets.DESCR) #(1797, 64)  #pandas : describe()
# print(datasets.feature_names)  #pandas : colums()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(1797, 64) (1797,)
print(x)
print(y)  
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2 3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True))  


scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)



##########데이터 분리전에 one-hot encoding하기##########################
#y값 (1797,) ->  (1797,10) 만들어주기
from keras.utils.np_utils import to_categorical

y = to_categorical(y)
# print(y)
print(y.shape) #(1797, 10)
##################################################################


#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    
)
print(y_train)                                  
print(np.unique(y_train, return_counts=True))  

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
best_model.save(path + "keras62_autokeras6.h5")

# 4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과:', results)
