from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
        rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

augment_size = 100

print(x_train.shape)    #(60000, 28, 28)
print(x_train[0].shape)   #(28, 28)
print(x_train[1].shape)
print(x_train[0][0].shape)

print(np.tile(x_train[0]. reshape(28*28), 
              augment_size).reshape(-1,28,28,1). shape)
#(100, 28, 28, 1)
#np.tile(데이터, 증폭 시킬 갯수)

print(np.zeros(augment_size))
print(np.zeros(augment_size).shape) #(100,)

x_data = train_datagen.flow(
    np.tile(x_train[0]. reshape(28*28), 
              augment_size).reshape(-1,28,28,1),  #x 데이터
    np.zeros(augment_size), # y data: 그림만 그린다
    batch_size=augment_size,
    shuffle=True
).next()

###############################.next() 사용#############
print(x_data)  #x와 y 가 합쳐진 데이터 출력
print(type(x_data))  #<class 'tuple'>
print(x_data[0])          # x 데이터
print(x_data[1])
print(x_data[0].shape, x_data[1].shape)   #(100, 28, 28, 1) (100,)
print(type(x_data[0]))  #<class 'numpy.ndarray'>

#############################. next() 사용##########


#<keras.preprocessing.image.NumpyArrayIterator object at 0x0000022791E3DEE0>
print(x_data[0])   #x,y 가 모두 포함
print(x_data[0][0].shape)  #(100, 28, 28, 1)
print(x_data[0][1].shape) 

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7, i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][0][i], cmap='gray')     #.next() 미사용
    plt.imshow(x_data[0][i], cmap='gray')     #.next() 미사용
    
plt.show()

