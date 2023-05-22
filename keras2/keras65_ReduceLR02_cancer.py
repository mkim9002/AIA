import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
#1. 데이터
# 데이터 로드 및 전처리
datasets = load_breast_cancer()
print(datasets)

#print(datasets)
print(datasets.DESCR) #판다스 : .describe()
print(datasets.feature_names)   #판다스 :  .columns()

x = datasets ['data']
y = datasets.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x.shape, y.shape) #(569, 30) (569,) feature,열,columns 는 30
#print(y) #1010101 은 암에 걸린 사람과 아님

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#. 모델
model = Sequential()
model.add(Dense(64, input_dim=30))
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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor= 'val_loss', patience=20, mode='min', verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1, validation_split=0.2,
          callbacks=[es, rlr])

#4. 평가 예측
results = model.evaluate(x_test, y_test)

print("loss :", results)




