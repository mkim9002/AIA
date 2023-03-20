from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu',input_dim =10)) #print x shape의 (506. 13)input_dim은 x_shape두번째를를 본다
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train,y_train, epochs=10, batch_size=8,
          validation_split=0.2,
          verbose=1)

print(hist.history)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()