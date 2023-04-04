from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# 1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print(x_train)
print(y_train)    # [ 3 4 3 ... 25 3 25 ]
print(x_train.shape, y_train.shape)          # (8982,) (8982,)
print(x_test.shape, y_test.shape)            # (2246,) (2246,)

print(len(x_train[0]), len(x_train[1]), len(x_train[8981]))                  # 87 56 ... 105
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))                 # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))        # 145.5398574927633

print(np.unique(y_train))                    # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(np.unique(y_test))                     # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(type(x_train), type(x_test))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 전처리

pad_x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
print(x_train.shape)
print(x_test.shape)


# softmax 46, embedding input_dim=10000, output_dim=마음대로, input_length=max(len)
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_test = pad_x_test.reshape(pad_x_test.shape[0], pad_x_test.shape[1], 1)

# 2. 모델
model = Sequential()
model.add(Embedding(10000, 32, input_shape=(100,)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, y_train, epochs=100, batch_size=256)

# 4. 평가, 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print('acc : ', acc)