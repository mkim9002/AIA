#1. 데이터
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy1(dataset, 4)
print(x, "\n", y)
print(x.shape)
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_shape=(4, )))
# model.add(Dense(64, input_dim=4))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
mse = model.evaluate(x, y )
print("mse : ", mse)
x_pred = np.array([7, 8, 9, 10])
x_pred = x_pred.reshape(1, x_pred.shape[0])
print(x_pred.shape)

y_pred = model.predict(x_pred)
print(y_pred)

'''
mse :  3.410605131648481e-13
(1, 4)
[[11.000001]]

'''
