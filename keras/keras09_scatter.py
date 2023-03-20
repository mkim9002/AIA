from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x,y,
        train_size=0.7, shuffle = True, random_state=2134) #여기서 x,y는 x_train,x_test,y_train,y_test 로 분리가 된다

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
']
#컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 77, batch_size=1)

#4.평가,에측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x)          #이렇게 하면 새로운 X 값을 넣어 y값을 예측할 수 있게 된다

import matplotlib.pyplot as plt       #"Matplotlib" 은 2D 그래프를 그릴때 가장 많이 사용되는 Python 라이브러리

plt.scatter(x,y)                      #scatter 산포그래프의 작성형
#plt.scatter(x,y_predict)
plt.plot(x,y_predict, color= 'red')   #plt.plot() 은 현재 컨텍스트 에 플롯 그래프를 그린다
plt.show()


