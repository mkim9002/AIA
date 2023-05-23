from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(32, (2, 2)))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
end = time.time()

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc', results[1])
print('걸린시간:', end - start)

model.save('./_save/keras70_1_mnist_graph.h5')

############ 시각화 #############
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('loss')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker=',', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker=',', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()