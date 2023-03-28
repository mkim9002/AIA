import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [11,12,13,14,15,16,17,18,19,20],
                    [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)

y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
print(y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(3, 3)))
model.add(Dense(6))

#3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
mse = model.evaluate(x, y, batch_size=1 )
print("mse : ", mse)

x_pred = np.array([[8, 9, 10],[18, 19, 20], [28, 29, 30]])
x_pred = np.transpose(x_pred)
x_pred = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1])
# print(x_pred)
# print(x_pred.shape)
y_pred = model.predict(x_pred, batch_size=1)
print(y_pred)

'''
mse :  0.021349055304502446
[[ 8.95572  19.07867  29.047264 10.222331 20.022017 29.832714]]
'''


