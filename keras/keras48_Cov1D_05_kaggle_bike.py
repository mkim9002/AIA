from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Conv1D, Flatten, Dropout, MaxPooling2D,SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


# 1. 데이터
path = './_data/kaggle_bike/'   #점 하나 현재폴더의밑에 점하나는 스터디
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 
                      
print(test_csv)        #캐쥬얼 레지스트 삭제
print(test_csv.shape)  #출력결과 ((6493, 8))

print(train_csv.info) 

print(type(train_csv)) 


#1.3결측치 처리 1 .제거
# pirnt(train_csv.insul11())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(11)
print(train_csv.info())
print(train_csv.shape)
############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['count','casual','registered'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['count']
print(y)

print(x.shape, y.shape) #(10886, 8,1,1) (10886,)

#1.4 reshape
x = np.array(x)
x = x.reshape(10886, 8, 1)
print(test_csv.shape) #(6493, 8)

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(6493, 8, 1)

###############################train_csv 데이터에서 x와y를 분리




x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print('y의 라벨값 :', np.unique(y))



#2. 모델 구성
model=Sequential()
# model.add(LSTM(10, input_shape = (3,3)))  
model.add(Conv1D(10,2,input_shape = (8,1))) 
model.add(Conv1D(10,2))                    
model.add(Conv1D(10,2, padding='same'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=1800, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=2, batch_size=99,
          validation_split=0.01,
          verbose=1,
          callbacks=(es),
)
     
# print("===========================================================")
# print(hist)
# print("===========================================================")
# print(hist.history)
# print("===========================================================")
# print(hist.history['loss'])
print("===========================================================")
print(hist.history['val_loss'])


#4/ 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss :', )

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)
#r2 스코어 : 0.6503924110719093

#RMSE함수의 정의 
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
#RMSE함수의 실행(사용)
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'samplesubmission.csv',index_col=0)
print(submission) #카운트라는 컬럼에 데이터 데입
submission['count'] = y_submit

import datetime
date= datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_save = './_save/kaggle_bike/' 
submission.to_csv(path_save + 'submit' + date +'.csv') 

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker = '.', c='red', label='로스')
# plt.plot(hist.history['val_loss'], marker = '.', c='blue', label='발_로스')
# plt.title('보스톤')
# plt.xlabel('epochs')
# plt.ylabel('loss,val_loss')
# plt.legend()
# plt.grid()
# plt.show()

