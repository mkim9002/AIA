# 데이콘 따릉이 문제풀ㅇ
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './keras/_data_ddarung/'
# train_csv = pd.read_csv('./_data/ddarung/train.csv')
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #첫번쩨 는 아이디 인덱스 이다.
#헤더와 인덱스틑 연산하지 않는다.
print(train_csv)


test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)

print(test_csv)
print(test_csv.shape)    #715,9

#=======================================================================
print(train_csv.columns)

#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#      dtype='object')

print(train_csv.info())

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64


print(train_csv.describe())

#2.모델구성
print(type(train_csv))  #class 'pandas.core.frame.DataFrame'



###################결축지 처리################################

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()  #결축지 삭제
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)



#############################train.csv에서 x와y를 분리  ############

x = train_csv.drop(['count'], axis=1)

print(x)

y = train_csv['count']
print(y)

################ train.csv에서 x와y를 분리  ################

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, train_size=0.7, random_state=777
)
                                      #결축지 제거 후
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)


# 모델구성
model = Sequential()
model.add(Dense(64,input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))




#3. 컴파일, 훈련 COMPILE, fit, epochs

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 10, batch_size=32, verbose=1)



#4. 평가, 예측 .evaluate, predict

loss = model.evaluate(x_test,y_test)
print('loss :', loss)

'''

y_predict = model.predict(x)          #이렇게 하면 새로운 X 값을 넣어 y값을 예측할 수 있게 된다

import matplotlib.pyplot as plt       #"Matplotlib" 은 2D 그래프를 그릴때 가장 많이 사용되는 Python 라이브러리

plt.scatter(x,y)                      #scatter 산포그래프의 작성형
#plt.scatter(x,y_predict)
plt.plot(x,y_predict, color= 'red')   #plt.plot() 은 현재 컨텍스트 에 플롯 그래프를 그린다
plt.show()


'''