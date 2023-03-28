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
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 2, activation ='relu'))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(2))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)

'''
mse :  0.18167145550251007
예측값 :
 [[17.039576 30.494492]
 [17.741278 31.473204]
 [18.442978 32.451916]]
'''




