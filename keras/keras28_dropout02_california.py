from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='sigmoid',input_dim =8)) 
model.add(Dropout(0.3))
model.add(Dense(10,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(5,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))

# input1 = Input(shape=(8,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(10)(dense1)
# dense3 = Dense(10)(dense2)
# dense4 = Dense(5)(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)







#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=100, batch_size=32,
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

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label='발_로스')
plt.title('보스톤')
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()

