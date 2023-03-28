#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
x_test = np.array([[8,9,10], [18,19,20]])
y_test = np.array([[8,9,10], [18,19,20]])
x_predict = np.array([[21,22,23], [31,32,33]])
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
x_predict = np.transpose(x_predict)
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(2)(dense3)
model = Model(inputs = input1, outputs= output1)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

'''
mse :  0.2387085109949112
예측값 :
 [[16.98405  31.746256]
 [17.69146  32.791748]
 [18.398863 33.83725 ]]
'''
 


