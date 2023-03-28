#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201)])
x2 = np.array([range(501, 601), range(601, 701)])
y = np.array([range(1, 101), range(101, 201)])
print(x1.shape)
print(x2.shape)
print(y.shape)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)
print(x1.shape)
print(x2.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(
x1, y, random_state=66, test_size=0.2, shuffle = False
)
x1_val, x1_test, y_val, y_test = train_test_split(
x1_test, y_test, random_state=66, test_size=0.5, shuffle = False
)
x2_train, x2_test = train_test_split(
x2, random_state=66, test_size=0.2, shuffle = False
)
x2_val, x2_test = train_test_split(
x2_test, random_state=66, test_size=0.5, shuffle = False
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

output1 = Dense(30)(merge1)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

model = Model(inputs = [input1, input2],
outputs = output1
)

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y_train,
epochs=50, batch_size=1,
validation_data=([x1_val, x2_val], y_val))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict([x1_test, x2_test])
print("y1 예측값 : \n", y_predict)

'''
mse :  [5.279398131370544, 5.279398441314697]      
y1 예측값 :
 [[ 88.77564  189.58499 ]
 [ 89.66991  190.51288 ]
 [ 90.56419  191.44077 ]
 [ 91.45554  192.36635 ]
 [ 92.34682  193.29192 ]
 [ 93.236565 194.21439 ]
 [ 94.12629  195.1369  ]
 [ 95.01603  196.05936 ]
 [ 95.905754 196.98183 ]
 [ 96.79552  197.90427 ]]
'''

