from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu',input_dim =8)) #print x shape의 (506. 13)input_dim은 x_shape두번째를를 본다
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train,y_train, epochs=9, batch_size=2,
          validation_split=0.2,
          verbose=1)

print(hist.history)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()

