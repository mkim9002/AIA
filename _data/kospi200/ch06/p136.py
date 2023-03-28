# 앙상블 다:다 모델2
#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(501, 601), range(601, 701)])
y3 = np.array([range(701, 801), range(801, 901)])

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
print(y3.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
print(y3.shape)

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
# y3 데이터의 분리
y3_train, y3_test = train_test_split(
y3 , random_state=66, test_size=0.2, shuffle=False
)
y3_val, y3_test = train_test_split(
y3_test , random_state=66, test_size=0.5, shuffle=False
)

y3_train.shape : (80, 2)
y3_val.shape : (10, 2)
y3_test.shape : (10, 2)

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

output3 = Dense(25)(middle3)
output3 = Dense(5)(output3)
output3 = Dense(2)(output3)

model = Model(inputs = [input1, input2],
outputs = [output1, output2, output3]
)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train],
          epochs=50, batch_size=1,
          validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val]))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
[y1_test, y2_test, y3_test], batch_size=1)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y1_predict,
      "\n y2 예측값 : \n", y2_predict, "\n y3 예측값 : \n", y3_predict)

'''
y1 예측값 :
 [[ 88.44821  194.85641 ]
 [ 89.43481  195.92892 ]
 [ 90.4214   197.00153 ]
 [ 91.38869  198.0581  ]
 [ 92.34643  199.10693 ]
 [ 93.303505 200.15495 ]
 [ 94.26077  201.20305 ]
 [ 95.21781  202.25096 ]
 [ 96.17501  203.29916 ]
 [ 97.13211  204.3472  ]]
 y2 예측값 :
 [[596.1657  697.1939 ]
 [597.34674 698.4307 ]
 [598.52795 699.6677 ]
 [599.7184  700.9193 ]
 [600.9155  702.18066]
 [602.128   703.4594 ]
 [603.34015 704.7383 ]
 [604.5525  706.017  ]
 [605.7646  707.29565]
 [606.97687 708.5743 ]]
 y3 예측값 :
 [[797.687   897.50885]
 [799.0154  898.765  ]
 [800.3435  900.021  ]
 [801.69324 901.29895]
 [803.0575  902.5908 ]
 [804.44696 903.90466]
 [805.83673 905.21875]
 [807.22626 906.53265]
 [808.6156  907.8467 ]
 [810.0051  909.1606 ]]
'''















