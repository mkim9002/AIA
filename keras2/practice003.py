

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import time

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0:7], 'GPU')  # GPU 2와 GPU 3만 사용하도록 설정
        for gpu in gpus[0:7]:
            tf.config.experimental.set_memory_growth(gpu, True)  # 메모리 증가 설정

        strategy = tf.distribute.MirroredStrategy(devices=["GPU:2", "GPU:3"])  # 사용할 GPU 디바이스 목록 지정
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    except RuntimeError as e:
        print(e)

        
############
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
with strategy.scope():  # 분산 전략 적용
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