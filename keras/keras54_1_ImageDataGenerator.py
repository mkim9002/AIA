import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

xy_train= train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test= train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) #Found 120 images belonging to 2 classes.

print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x0000019285EEE670>
print(xy_test[0])
#print(xy_train.shape) #'DirectoryIterator' object has no attribute 'shape'
print(len(xy_train))    #32
print(len(xy_train[0])) #2
print(xy_train[0][0])   # x 다섯개 들어감
print(xy_train[0][1])   #[0. 0. 1. 0. 0.]
print(xy_train[0][0].shape) #(5, 200, 200, 1)
print(xy_train[0][1].shape) #(5,)


print('=====================================================')
print(type(xy_train))   #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0]))  #<class 'numpy.ndarray'>
print(type(xy_train[0][1]))  #<class 'numpy.ndarray'>


