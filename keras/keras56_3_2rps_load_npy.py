# 불러와서 모델 완성
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
# 1. 데이터
save_path = 'd:/study_data/_save/rps/'

rps_x_train = np.load(save_path + 'keras56_rps_x_train.npy')
rps_x_test = np.load(save_path + 'keras56_rps_x_test.npy')
rps_y_train = np.load(save_path + 'keras56_rps_y_train.npy')
rps_y_test = np.load(save_path + 'keras56_rps_y_test.npy')

# 2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(rps_x_train, rps_y_train, epochs=30, validation_data=(rps_x_test, rps_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()

# 4. 평가, 예측
loss = model.evaluate(rps_x_test, rps_y_test)
print('loss : ', loss)

y_predict = np.argmax(model.predict(rps_x_test), axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(rps_y_test, y_predict)
print('acc : ', acc)