#1. 데이터
import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train.shape : ", x_train.shape) #(3, 5, 1)
print("y_train.shape : ", y_train.shape) #(3, )

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (5, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=2, callbacks=[early_stopping])

#4. 평가, 예측
x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape) #(1, 5)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape) # (1, 5, 1)

y_predict = model.predict(x_predict)
print("예측값 : ", y_predict)

'''
예측값 :  [[8.576211]]
'''
