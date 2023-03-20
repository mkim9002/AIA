from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

#실습 :: 잘라봐!!
#x_train, y_train 슬라이싱
x_train = x_train[:10]
y_train = y_train[:10]

print(x_train)
print(y_train)

#x_val, y_val 슬라이싱
x_val = x_val[1:]
y_val = y_val[1:]

print(x_val)
print(y_val)


#x_test, y_test 슬라이싱
x_test = x_test[:-1]
y_test = y_test[:-1]

print(x_test)
print(y_test)


# #2. 모델
# model= Sequential()
# model.add(Dense(5, activation='linear', input_dim =1))
# model.add(Dense(1))


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train,y_train, epochs=10, batch_size=1,
#           validation_data=(x_val,y_val))

# # 4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
# print('loss :', loss)

# result = model.predict([17])
# print('17의 예측값 :', result)