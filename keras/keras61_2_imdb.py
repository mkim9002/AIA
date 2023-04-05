from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#영화평
(x_train, y_train), (x_test,y_test) = imdb.load_data(
    num_words=10000
)

print(x_train)
print(y_train) #[1 0 0 ... 0 1 0]
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(np.unique(y_train,return_counts=True))   #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))                    
#판다스 에선 value_counts
print(pd.value_counts(y_train))
# 1    12500
# 0    12500
print("영화평의 최대길이 : ", max(len(i) for i in x_train))                 # 2494
print("영화평의 평균길이 : ", sum(map(len, x_train)) / len(x_train))        #238.71364


#실습
#만들기

print(np.unique(y_train))                    # 
print(np.unique(y_test))                     # 
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
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax')) # 클래스 개수를 2로 변경
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, y_train, epochs=100, batch_size=256)

# 4. 평가, 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print('acc : ', acc)




