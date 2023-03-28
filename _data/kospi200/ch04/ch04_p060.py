#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data=(x_val, y_val))
# epochs를 1000 에서 300 으로 줄였습니다.

#4. 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

'''
mse :  [1.9580766092985867e-07, 1.9580765808768774e-07]
[[80.99965 ]
 [81.99964 ]
 [82.999626]
 [83.999626]
 [84.99962 ]
 [85.9996  ]
 [86.999596]
 [87.99958 ]
 [88.99959 ]
 [89.99956 ]
 [90.99956 ]
 [91.99955 ]
 [92.99954 ]
 [93.99952 ]
 [94.99951 ]
 [95.99951 ]
 [96.9995  ]
 [97.99948 ]
 [98.99948 ]
 [99.99948 ]]
RMSE :  0.0004425015942681548
R2 :  0.9999999941110478
'''