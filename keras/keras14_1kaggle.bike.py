# bike-sharing-demand _kaggle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mean_squared_error는평균 제곱 오차 회귀 손실하는 함수
                                                         #r2_score는 R ^ 2 (결정 계수) 회귀 점수 함수
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #첫번쩨 는 아이디 인덱스 이다.
print(train_csv)



test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)

print(test_csv)

print(test_csv.shape) 

print(train_csv.columns)

print(train_csv.info())

print(train_csv.describe())

#2.모델구성
print(type(train_csv))


###################결축지 처리################################

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()  #결축지 삭제
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)



#############################train.csv에서 x와y를 분리  ############

x = train_csv.drop(['count','casual','registered'], axis=1)

print(x)

y = train_csv['count']
print(y)


################ train.csv에서 x와y를 분리  ################

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, train_size=0.7, random_state=777
)
                                      #결축지 제거 후
print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 


# 모델구성
model = Sequential()
model.add(Dense(81,input_dim=8))
model.add(Dense(64, activation = 'relu')) #relu는 음수를 양수로
model.add(Dense(32, activation = 'linear')) #linear 는 이과제에서 필수
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))





#3. 컴파일, 훈련 COMPILE, fit, epochs

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 10, batch_size=10, verbose=1)



#4. 평가, 예측 .evaluate, predict

loss = model.evaluate(x_test,y_test)
print('loss :', loss)   # loss 는 mse  그런데 여기서는 RMSE를 달라고 한다

y_predict = model.predict(x_test)  

r2 = r2_score(y_test, y_predict)
print('r2 스코어', r2)

def RMSE(y_test, y_predict):             #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)           #RMSE 함수 사용
print('RMSE :', rmse)


######## sumission,csv를 만들어 봅시다###########
print(test_csv.isnull().sum())
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path +'sampleSubmission.csv', index_col=0)
print(submission)

submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0307_1258.csv')

