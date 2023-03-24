#과적합 배제

#저장힐때 평가결과값, 훈련시간을 파일에 넣죠

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM , SimpleRNN, Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from tensorflow.keras. callbacks import EarlyStopping

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']


print(x.shape) #(506, 13)
print(y.shape) #(506,)

x = x.reshape(506, 13, 1)

x_train, x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state=333
)




#2. 모델

model=Sequential()
# model.add(LSTM(10, input_shape = (13,1)))  
model.add(Conv1D(10,2,input_shape = (13,1))) 
model.add(Conv1D(10,2))                    
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
print(date) #2023-03-14 11:14:35.924884
date = date.strftime('%m%d_%H%M')
print(date) #0314_1115


filepath = './_save/MCP/keras27_4/'
filename  = '{epoch:04d}-{val_loss:.4f}.hdf5'


model.fit(x_train, y_train, epochs=100)
        


model.save('./_save/MCP/keras27_3_save_model.h5')
#4/ 평가 예측
from sklearn.metrics import r2_score

print("================1. 기본출력==================")
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss :',loss )
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 스코어 :', r2)

