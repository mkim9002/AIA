import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#1. 데이터
# 데이터 로드 및 전처리
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=543432)

#. 모델
model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam',metrics=['acc'])
from tensorflow.keras.optimizers import Adam
learnig_rate =0.1
optimizer = Adam(learning_rate=learnig_rate)
model.compile(loss='mse', optimizer=optimizer,metrics=['mae'])

model.fit(x_train,y_train, epochs=10,batch_size=32)

#4. 평가 예측
results = model.evaluate(x_test, y_test)

print("loss :", results)




