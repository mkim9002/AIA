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
x1, y1 = split_xy5(samsung, 5, 1) 
x2, y2 = split_xy5(kospi200, 5, 1) 
print(x2[0,:], "\n", y2[0])
print(x2.shape)
print(y2.shape)


# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=2, test_size = 0.3)

print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)

x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)
print(x2_test.shape)


#### 데이터 전처리 #####
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])


from keras.models import Model
from keras.layers import Dense, Input

# 모델구성
input1 = Input(shape=(25, ))
dense1 = Dense(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(25, ))
dense2 = Dense(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1, input2],
              outputs = output3 )


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=100, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])

'''
loss :  1348967.9961648777
mse :  1348968.125
종가 :  [52200] / 예측가 :  [50549.504]
종가 :  [41450] / 예측가 :  [41341.992]
종가 :  [49650] / 예측가 :  [49872.363]
종가 :  [44800] / 예측가 :  [45122.33]
종가 :  [49500] / 예측가 :  [48873.23]
'''