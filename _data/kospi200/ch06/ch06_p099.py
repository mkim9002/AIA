#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_predict = np.array([11,12,13])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 1, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

'''
mse :  6.85486156726256e-09
예측값 :
 [[11.000205]
 [12.000281]
 [13.000355]]
'''


