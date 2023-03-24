import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.keras. callbacks import EarlyStopping

#1. data
dataset = np.array(range(1, 101))
timesteps = 5
x_predict = np.array(range(96, 106))

#100-106 예상값 ,시계열 데이터

def split_x(dataset, timesteps):
    gen=(dataset[i : (i + timesteps)] for i in range(len(dataset)- timesteps +1))
    return np.array(list(gen))

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, 4:5]

print(x.shape) #(96, 4)
print(y.shape) #(96, 1)

a = split_x(x_predict, timesteps)
print(a)

a1 = a[:, :4]
print(a1)

x = x.reshape(96, 4, 1)

a1 = a1.reshape(6, 4 , 1)

#2. model
model = Sequential()
model.add(LSTM(64, input_shape=(4,1), activation='linear', recurrent_activation='tanh'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.compile.practice
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x,y, epochs=100, callbacks=[es])

#4. evaluate, predict
loss = model.evaluate(x,y)
a1 = a1.reshape(6, 4 ,1)

result = model.predict(a1)
print('loss :', loss)
print('result :', result)

    
    
    