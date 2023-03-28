#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(501, 601), range(601, 701)])
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
x1, y1, random_state=66, test_size=0.2, shuffle = False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
x1_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)
x2_train, x2_test, y2_train, y2_test = train_test_split(
x2, y2, random_state=66, test_size=0.2, shuffle = False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
x2_test, y2_test, random_state=66, test_size=0.5, shuffle = False
)

print('x2_train.shape : ', x2_train.shape)
print('x2_val.shape : ', x2_val.shape)
print('x2_test.shape : ', x2_test.shape)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

input2 = Input(shape=(2,))
dense2 = Dense(50, activation='relu')(input2)
dense2 = Dense(30)(dense2)
dense2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(30)(middle2)

output1 = Dense(30)(middle3)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

output2 = Dense(20)(middle3)
output2 = Dense(70)(output2)
output2 = Dense(2)(output2)

model = Model(inputs = [input1, input2],
outputs = [output1, output2]
)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train],
epochs=50, batch_size=1,
validation_data=([x1_val, x2_val] , [y1_val, y2_val]))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
                     [y1_test, y2_test], batch_size=1)
print("mse : ", mse)
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y1_predict, "\n y2 예측값 : \n", y2_predict)

'''
mse :  [7.217605972290039, 4.728159427642822, 2.4894468784332275, 4.728159427642822, 2.4894468784332275]
y1 예측값 :
 [[ 89.20771  189.7767  ]
 [ 90.04419  190.67882 ]
 [ 90.88073  191.58107 ]
 [ 91.71727  192.48332 ]
 [ 92.55382  193.38557 ]
 [ 93.39036  194.28772 ]
 [ 94.22693  195.18999 ]
 [ 95.06351  196.09224 ]
 [ 95.899994 196.99432 ]
 [ 96.73661  197.89656 ]]
 y2 예측값 :
 [[591.9566  692.2949 ]
 [593.0293  693.4038 ]
 [594.102   694.51276]
 [595.1748  695.62195]
 [596.24744 696.73114]
 [597.3204  697.8404 ]
 [598.393   698.9494 ]
 [599.4657  700.0583 ]
 [600.5386  701.16766]
 [601.6112  702.2765 ]]
'''
