from tensorflow.keras.datasets import mnist , cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


#실습 만들기 , 목표 : cnn성능보다 좋게 만들기

#1, 데이터28*28*1 , reshape
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#one-hot-coding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)


print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000,-1) #-1 하면 뒤의 숫자를 자동으로 맞추어준다
# x_test = x_test.reshape(10000,-1) #28*28 혹은 784 라고 적어도 된다.

print(x_train.shape) #(60000, 784)
print(x_test.shape) #(10000, 784)





#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(28,28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10,activation='softmax'))

# 3. 모델 컴파일
import time
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 훈련
history = model.fit(x_train, y_train, epochs=1, batch_size=99, validation_split=0.01,verbose=1)

end_time=time.time()

# 5. 모델 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy: ', acc)




