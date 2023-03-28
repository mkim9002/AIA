#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_predict = np.array([11,12,13])

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(1)(dense3)
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
mse :  2.1567150287893355e-09
예측값 :
 [[11.000072]
 [12.000083]
 [13.000095]]
'''




