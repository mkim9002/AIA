from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(442, 10) (442,)

x = x.reshape(442, 10, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2)




#2. 모델 구성
model=Sequential()
# model.add(LSTM(10, input_shape = (3,1)))  
model.add(Conv1D(10,2,input_shape = (10,1))) 
model.add(Conv1D(10,2))                    
model.add(Conv1D(10,2))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary()



#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=20, batch_size=128,
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

