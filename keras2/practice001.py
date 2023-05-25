import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import time

###
# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 모든 GPU 메모리 증가 옵션 활성화
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # GPU를 사용하여 모든 연산 실행
        tf.config.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 GPU가 설정되어야 함
        print(e)

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