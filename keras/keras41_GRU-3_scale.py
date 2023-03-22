import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping


#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7], [6,7,8,],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array ([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70]) #아워너 80


print(x.shape, y.shape) #(13, 3) (13,)

x = x.reshape(13,3,1)
print(x.shape)
                
#2. model
model = Sequential()  
model.add(LSTM(10, input_shape=(3,1), return_sequences=True)) #return_sequences 는 LSTM 두개연결 가능
model.add(LSTM(11, return_sequences=True)) #return_sequences 로 GRU 연결가능
model.add(GRU(12))
model.add(Dense(1))
model.summary()

#3. compile, loss,model.fit, epochs
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x,y, epochs=1000, callbacks=[es])

#4. loss,predict
loss= model.evaluate(x,y)
x_predict = np.array([50,60,70]).reshape(1,3,1) 
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss :', loss)
print('50,60,70의 결과 : ', result)

# 50,60,70의 결과 :  [[78.988335]]



