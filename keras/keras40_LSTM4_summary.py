import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


#1. data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
#y = ?

x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],
             [5,6,7,8,9]])
y = np.array([6,7,8,9,10])

print(x.shape, y.shape) #(6, 4) (6,)
#x 의shape = (행, 열, 몇개씩 훈련 시키는지 !!!)
x = x.reshape(5,5,1) #
print(x.shape) #(5, 5, 1)

#2. model
model = Sequential()                #[batch, timesteps, feature]
model.add(LSTM(10, input_shape=(5,1))) # (10*10) +(1*10) +10 =120
#units * (feature + bias + units)= params
# units * units + units*feature +units *bias
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. compile, 
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x,y, epochs=1000, callbacks=[es])

#4. loss,predict
loss= model.evaluate(x,y)
x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[[6],[7],[8],[9],[10]]]
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss :', loss)
print('[6,7,8,9,10]의 결과', result)

#[6,7,8,9,10]의 결과 [[10.907812]]
