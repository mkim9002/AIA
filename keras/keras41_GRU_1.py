import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU

#1. data
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
#y = ?

x = np.array([[1,2,3],[2,3,4,],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)
#x 의shape = (행, 열, 몇개씩 훈련 시키는지 !!!)
x = x.reshape(7,3,1) #[[[1],[2],[3], [[2]. [3], [4]. .......]]]
print(x.shape) #(7, 3, 1)

#2. model
model = Sequential()
model.add(GRU(32, input_shape=(3,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. compile, 
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. loss,predict
loss= model.evaluate(x,y)
x_predict = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9].[10]]]
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss :', loss)
print('[8,9,10]의 결과', result)

