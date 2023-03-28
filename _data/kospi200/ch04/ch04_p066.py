#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
       x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1,
validation_data=(x_val, y_val))

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
mse :  [1.0809424566105008e-05, 1.0809424566105008e-05]
[[ 81.00295 ]
 [ 82.002975]
 [ 83.00301 ]
 [ 84.003044]
 [ 85.00309 ]
 [ 86.00312 ]
 [ 87.00316 ]
 [ 88.00319 ]
 [ 89.003235]
 [ 90.00327 ]
 [ 91.00329 ]
 [ 92.003334]
 [ 93.003365]
 [ 94.00341 ]
 [ 95.00344 ]
 [ 96.00347 ]
 [ 97.00352 ]
 [ 98.003555]
 [ 99.003586]
 [100.00361 ]]
RMSE :  0.003287768934415101
R2 :  0.9999996749045243
'''
