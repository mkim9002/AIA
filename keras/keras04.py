import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이타
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=412)

# 평가예측
loss = model.evaluate(x,y )
print('loss : ',loss)

result = model.predict([6]) 
print("[6]의 예측값 :", result)

#[6]의 예측값 : [[5.7025547]]