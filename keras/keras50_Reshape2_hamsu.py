
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,Conv1D,Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(np.unique(y_train,return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))


x_train = x_train.reshape(60000,28,28,1)/255
x_test = x_test.reshape(10000,28,28,1)/255
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)





#2. 모델구성
# model =Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3),
#                padding='same', input_shape=(28,28,1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,(3,3)))
# model.add(Conv2D(10,3))
# model.add(MaxPooling2D())
# model.add(Flatten())     # (N, 250)
# model.add(Reshape(target_shape=(25,10)))
# model.add(Conv1D(10,3, padding='same'))
# model.add(LSTM(784))
# model.add(Reshape(target_shape=(28, 28, 1)))
# model.add(Conv2D(32,(3,3)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))
# model.summary()


input1 = Input(shape=(28,28,1))
dense1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(input1)
dense2 = MaxPooling2D()(dense1)
dense3 = Conv2D(32,(3,3))(dense2)
dense4 = Conv2D(10,3)(dense3)
dense5 = MaxPooling2D()(dense4)
dense6 = Flatten()(dense5)
dense7 = Reshape(target_shape=(25,10))(dense6)
dense8 = Conv1D(10,3, padding='same')(dense7)
dense9 = LSTM(784)(dense8)
dense10 = Reshape(target_shape=(28, 28, 1))(dense9)
dense11 = Conv2D(32,(3,3))(dense10)
dense12 = Flatten()(dense11)
output1 = Dense(10, activation = 'softmax')(dense12)
model = Model(inputs=input1, outputs=output1)
model.summary()







# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_acc', patience=30, mode='max',
#                    verbose=1,
#                    restore_best_weights=True
#                    )

# model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2,
#           callbacks=(es))

# #4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('loss:', results[0])
# print('acc:', results[1])


# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)

# acc = accuracy_score(y_test, y_pred)
# print('acc:', acc)