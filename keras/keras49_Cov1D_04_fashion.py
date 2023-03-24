from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(np.unique(y_train,return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

x_train = x_train.reshape(60000,-1) #-1 하면 뒤의 숫자를 자동으로 맞추어준다
x_test = x_test.reshape(10000,-1)

print(x_train.shape) #(60000, 784)
print(x_test.shape) #(10000, 784)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(y_train.shape, y_test.shape) #(60000,) (10000,)

y_train = y_train.reshape(60000,)
y_test = y_test.reshape(10000,)

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))


x_train = x_train.reshape( 60000, 56, 14)
x_test = x_test.reshape( 10000, 56, 14)


#2. 모델구성
model=Sequential()
# model.add(LSTM(10, input_shape = (3,3)))  
model.add(Conv1D(10,2,input_shape = (56,14))) 
model.add(Conv1D(10,2))                    
model.add(Conv1D(10,2, padding='same'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(10,activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=30, mode='max',
                   verbose=1,
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2,
          callbacks=(es))

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

# loss: 0.4915834069252014
# acc: 0.8314999938011169
# acc: 0.8315
