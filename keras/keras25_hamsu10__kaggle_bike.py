from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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
##########################################


print(train_csv.columns) 
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed',]
#       dtype='object')
print(train_csv.info) 

print(type(train_csv)) 

################################
#결측치 처리 1 .제거
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

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x))


###############################train_csv 데이터에서 x와y를 분리




x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max)
test_csv = scaler.transform(test_csv)

print(x.shape, y.shape) #(10886, 8) (10886,)
#2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='sigmoid',input_dim =8)) #print x shape의 (506. 13)input_dim은 x_shape두번째를를 본다
# model.add(Dense(10,activation='sigmoid'))
# model.add(Dense(10,activation='sigmoid'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1,activation='linear'))

input1 = Input(shape=(8,))
dense1 = Dense(10, activation = 'sigmoid')(input1)
dense2 = Dense(10, activation = 'sigmoid')(dense1)
dense3 = Dense(10, activation = 'sigmoid')(dense2)
dense4 = Dense(5, activation = 'relu')(dense3)
output1 = Dense(1, activation = 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=40, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=100, batch_size=16,
          validation_split=0.2,
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
submission.to_csv(path + 'samplesubmission_0314_02.csv') 

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

