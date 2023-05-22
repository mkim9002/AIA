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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2)
end = time.time()

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc', results[1])
print('걸린시간:', end - start)

#flatten
# loss: 0.3457137644290924
# acc 0.9843999743461609
# 걸린시간: 1395.3789780139923

# #GlobalAveragePooling
# loss: 0.1285160928964615
# acc 0.9589999914169312
# 걸린시간: 1358.137152671814
