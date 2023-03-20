from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()                      #(N,3)
model.add(Dense(10, input_shape=(3,))) #(batch_size, input_dim)
model.add(Dense(units=15))        #출력 (batch_size,units)
model.summary()
#units 아웃풋 노드갯수


