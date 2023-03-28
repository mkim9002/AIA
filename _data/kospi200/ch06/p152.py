#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201)])
y1 = np.array([range(501, 601), range(601, 701)])
y2 = np.array([range(1, 101), range(101, 201)])
print(x.shape)
print(y1.shape)
print(y2.shape)
x = np.transpose(x)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
print(x.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test = train_test_split(
x, y1, random_state=66, test_size=0.2, shuffle = False
)
x_val, x_test, y1_val, y1_test = train_test_split(
x_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)
y2_train, y2_test = train_test_split(
y2, random_state=66, test_size=0.2, shuffle = False
)
y2_val, y2_test = train_test_split(
y2_test, random_state=66, test_size=0.5, shuffle = False
)
print('y2_train.shape : ', y2_train.shape)
print('y2_val.shape : ', y2_val.shape)
print('y2_test.shape : ', y2_test.shape)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)

output1 = Dense(30)(dense1)
output1 = Dense(7)(output1)
output1 = Dense(2)(output1)

output2 = Dense(30)(dense1)
output2 = Dense(7)(output1)
output2 = Dense(2)(output1)

model = Model(inputs = input1,
outputs = [output1, output2]
)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, [y1_train, y2_train],
epochs=50, batch_size=1,
validation_data=(x_val, [y1_val, y2_val]))
#4. 평가 예측
mse = model.evaluate(x_test, [y1_test, y2_test], batch_size=1)
print("mse : ", mse)
y1_predict, y2_predict = model.predict(x_test)
print("y1 예측값 : \n", y1_predict, "\n y2 예측값 : \n", y2_predict)

'''
mse :  [2135.31376953125, 392.51361083984375, 1742.8001708984375, 392.51361083984375, 1742.8001708984375]
y1 예측값 :
 [[600.42944 709.84674]
 [602.3368  712.0305 ]
 [604.2434  714.2063 ]
 [606.1499  716.3818 ]
 [608.05634 718.55725]
 [609.96277 720.7328 ]
 [611.86945 722.9084 ]
 [613.776   725.0841 ]
 [615.6877  727.2662 ]
 [617.7094  729.58826]]
 y2 예측값 :
 [[ 47.70852  157.86458 ]
 [ 47.861324 158.3448  ]
 [ 48.01408  158.82281 ]
 [ 48.166824 159.30074 ]
 [ 48.319565 159.77866 ]
 [ 48.472305 160.2566  ]
 [ 48.625065 160.73454 ]
 [ 48.777817 161.21254 ]
 [ 48.930977 161.69196 ]
 [ 49.092937 162.2031  ]]
''' 



