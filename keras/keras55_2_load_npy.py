import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#1. data
path= 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr=xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])        
# np.save(path + 'keras55_1_y_test.npy', arr=xy_test[0][1])


x_train = np.load(path +'keras55_1_x_train.npy' )
x_test = np.load(path +'keras55_1_x_test.npy' )
y_train = np.load(path +'keras55_1_y_train.npy' )
y_test = np.load(path +'keras55_1_y_test.npy' )


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape=(100 ,100, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# #.3. 컴파일 훈련
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['acc'] )
# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=10,
#           batch_size=15,
#           validation_data=(xy_test[0][0], xy_test=[0][1]))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics =['acc'] )
# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=10,
#           batch_size=15,
#           validation_data=(xy_test[0][0], xy_test=[0][1]))


hist = model.fit(x_train,y_train, epochs=30,
                    steps_per_epoch=32,       #전체데이터/batch = 160/5 = 32
                    validation_data=[x_test, y_test],
                    validation_steps=24,       #전체데이터/batch = 120/5 = 24
                    )   

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc)
print(acc[-1])
print('loss :', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc :', acc[-1])
print('val_acc :', val_acc[-1])

#1. 그림그리기   subplot()
#2. 튜닝 !! 0.95 이상


import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.subplot(2,2,1)
plt.plot(loss)
plt.subplot(2,2,2)
plt.plot(val_loss)
plt.subplot(2,2,3)
plt.plot(acc)
plt.subplot(2,2,4)
plt.plot(val_acc)
plt.show()





