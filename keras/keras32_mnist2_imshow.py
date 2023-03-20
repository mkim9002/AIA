from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# x_train = x_train / 255.0
# y_train = y_test /255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 784)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(7,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu'))
model.add(Conv2D(filters=10, kernel_size=(7,7), activation='relu'))
model.add(Flatten())
model.add(Dense(66, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(10, activation='softmax'))

# 3. 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 훈련
history = model.fit(x_train, y_train, epochs=150, batch_size=1200, validation_split=0.2,verbose=1)

# 5. 모델 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy: ', acc)

plt.imshow(x_train[234], 'gray')
plt.show()