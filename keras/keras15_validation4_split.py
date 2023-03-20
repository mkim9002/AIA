from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))
x = np.array(range(1,17))
y = np.array(range(1,17))
#실습 :: 잘라봐!!!
#train_test_split 로만 잘라
#10:3:3

from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.3, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=1)

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)


#2. 모델
model= Sequential()
model.add(Dense(5, activation='linear', input_dim =1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=10, batch_size=1,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

result = model.predict([17])
print('17의 예측값 :', result)