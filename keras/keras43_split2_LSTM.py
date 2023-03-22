import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping


dataset = np.array(range(1, 101))
timesteps = 5
x_predict = np.array(range(96, 106)) #100-106 예상값
# 96,97,98,99
# 97,98,99,100
# 98,99,100,101
# ...
# 102,103,104,105

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

a = split_x(x_predict, timesteps)
print(a)

x = a[: , :4]
print(x)
print(x.shape) #(6, 4)
y = a[:,-1]
print(y)

print(x.shape) #(6, 4)
print(y.shape)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) #(96, 5)

a = bbb[:, :4]
b = bbb[:,-1]

print(a.shape) #(96, 4)
print(y.shape) #(96,)

a = a.reshape(96,4,1)
x = x.reshape(6,4,1)


#2. model
model = Sequential()
model.add(LSTM(10, input_shape=(4,1),return_sequences=True))
model.add(LSTM(11))
#model.add(GRU(12))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()



#3. compile, loss,model.fit, epochs
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(a,b, epochs=100, callbacks=[es])

#4. loss,predict

result = model.predict(x)

print('result : ', result)
