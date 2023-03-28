import numpy as np
import pandas as pd

kospi200 = np.load('./kospi200/data/kospi200.npy')
samsung = np.load('./kospi200/data/samsung.npy')

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1) 
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = np.reshape(x_train,
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)
print(x_test.shape)

#### 데이터 전처리 #####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

x_train_scaled = np.reshape(x_train_scaled,
    (x_train_scaled.shape[0], 5, 5))
x_test_scaled = np.reshape(x_test_scaled,
    (x_test_scaled.shape[0], 5, 5))
print(x_train_scaled.shape)
print(x_test_scaled.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(5, 5)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
          batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i], '/ 예측가 : ', y_pred[i])

'''

Epoch 44/100

loss :  1609906.8891261544
mse :  1609906.625
종가 :  [52200] / 예측가 :  [51450.69]
종가 :  [41450] / 예측가 :  [40155.082]
종가 :  [49650] / 예측가 :  [50907.082]
종가 :  [44800] / 예측가 :  [45825.527]
종가 :  [49500] / 예측가 :  [48564.38]


