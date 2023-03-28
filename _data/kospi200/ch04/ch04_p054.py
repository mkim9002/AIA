from keras.models import Sequential
from keras.layers import Dense
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1 , activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data = (x_test, y_test))
loss, mse = model.evaluate(x_test, y_test, batch_size =1)

print("loss : ", loss)
print("mse : ", mse)
y_predict = model.predict(x_test)
print("결과물 : \n", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

'''
loss :  88.37814407348633
mse :  88.37814331054688
결과물 :
 [[ 92.031555]
 [ 92.936325]
 [ 93.841095]
 [ 94.745865]
 [ 95.65063 ]
 [ 96.5554  ]
 [ 97.460175]
 [ 98.36494 ]
 [ 99.2697  ]
 [100.17448 ]]
RMSE :  9.400965052995423
R2 :  -9.712502294259542
'''
