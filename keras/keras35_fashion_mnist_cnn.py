from tensorflow.keras.datasets import mnist , cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  

x_train = x_train/255.
x_test = x_test/255.

# x_train = x_train.reshape(50000, 32, 32, 3)/255.
# x_test = x_test.reshape(10000, 32, 32, 3)/255.

print(np.max(x_train), np.min(x_train)) #1.0.0.0

print(np.unique(y_train,return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] output_dim='10 개'

#one hot encoder
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 32*32)
# x_test = x_test.reshape(10000, 1024)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(10,(9,9),input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, padding='same', kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'))
model.add(Conv2D(filters=2, kernel_size=(7,7), activation='relu'))
model.add(Flatten())
model.add(Dense(66, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 모델 컴파일
import time
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 훈련
history = model.fit(x_train, y_train, epochs=1, batch_size=1200, validation_split=0.01,verbose=1)

end_time=time.time()

# 5. 모델 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy: ', acc)

print("걸리는 시간:", end_time-start_time)
plt.imshow(x_train[250], cmap='viridis')
plt.show()

#acc ; 0.8911699779249448