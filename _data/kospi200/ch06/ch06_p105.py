#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5,6,7,], [11,12,13,14,15,16,17]])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([[8,9,10], [18,19,20]])
y_test = np.array([8,9,10])
x_predict = np.array([[21,22,23], [31,32,33]])

print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
x_predict = np.transpose(x_predict)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_test.shape : ' , x_test.shape)
print('y_test.shape : ' , y_test.shape)
print('x_predict.shape : ', x_predict.shape )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim = 2, activation ='relu'))
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
mse :  0.033768292516469955
예측값 :
 [[19.820276]
 [20.734518]
 [21.64575 ]]
'''
 

