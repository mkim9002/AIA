#데이콘 kaggle_bike 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
#1. 데이터
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
###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10, activation='linear'))
model.add(Dense(200,activation='relu'))
model.add(Dense(300 ,activation='relu'))
model.add(Dense(400 ,activation='relu'))
model.add(Dense(70 ,activation='relu'))
model.add(Dense(500 ,activation='relu'))
model.add(Dense(10 ,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(300 ,activation='relu'))
model.add(Dense(400 ,activation='relu'))
model.add(Dense(70 ,activation='relu'))
model.add(Dense(500 ,activation='relu'))
model.add(Dense(10 ,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=600, batch_size=32, 
          verbose=1,validation_split=0.2)
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2: ', r2)

def RMSE(y_test,y_predict): #RMSE 는 만든 변수명
    mean_squared_error(y_test,y_predict) #함수반환 = 리턴을사용
    return np.sqrt(mean_squared_error(y_test, y_predict)) #rmse 에 루트 씌워진다 np.sqrt 사용하면
rmse = RMSE(y_test,y_predict) #RMSE 함수사용
print("RMSE: ", rmse)
y_submit = model.predict(test_csv)
print(y_submit)
submission = pd.read_csv(path + 'samplesubmission.csv',index_col=0)
print(submission) #카운트라는 컬럼에 데이터 데입
submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'samplesubmission_0321_0447.csv') 

#최종값이 음수일경우 에러가뜬다.
#그러므로 양수값을 만들기위해 activation함수사용
#선형회기 activation 은  linear 함수 사용

# model.add(Dense(200,activation='relu'))
# model.add(Dense(300 ,activation='relu'))
# model.add(Dense(400 ,activation='relu'))
# model.add(Dense(70 ,activation='relu'))
# model.add(Dense(500 ,activation='relu'))
# model.add(Dense(10 ,activation='relu')) epochs=300 r2 = 4.41
