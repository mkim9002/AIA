from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x,y,
        train_size=0.4, shuffle = True, random_state=1234) #여기서 x,y는 x_train,x_test,y_train,y_test 로 분리가 된다

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(19))
model.add(Dense(13))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(1))

#컴파일 , 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs = 300, batch_size=1)

#4.평가,에측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)          #이렇게 하면 새로운 X 값을 넣어 y값을 예측할 수 있게 된다

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

#r2 score : 0.8259455423896532


#결정계수(R²,R Square) 는 회귀모형 내에서 설명변수 x로 설명할 수 있는 반응변수 y의 변동 비율
#총변동(SST)에서 설명 가능한 변동인 SSR이 차지하는 꼴(SSR / SST)로 나타낼 수 있다.
#결정계 수란 = '회귀 모델의 성과 지표', 1에 가까울 수록 좋은 회귀 모델, 0에 가까울 수록 나쁜 모델, 음수가 나올경우, 바로 폐기해야 하는 모델









'''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)
model.score(x, y)
'''


