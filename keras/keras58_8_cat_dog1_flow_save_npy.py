from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

cat_dog = train_datagen.flow_from_directory('d:/study_data/_data/cat_dog/PetImages/', target_size=(100, 100), batch_size=100, class_mode='binary', color_mode='rgb', shuffle=True)

cat_dog_x = cat_dog[0][0]
cat_dog_y = cat_dog[0][1]

cat_dog_x_train, cat_dog_x_test, cat_dog_y_train, cat_dog_y_test = train_test_split(cat_dog_x, cat_dog_y, train_size=0.7, shuffle=True, random_state=123)

augment_size = 100

np.random.seed(0)
randidx = np.random.randint(cat_dog_x_train.shape[0], size=augment_size)

x_augmented = cat_dog_x_train[randidx].copy()
y_augmented = cat_dog_y_train[randidx].copy()

# cat_dog_x_train = cat_dog_x_train.reshape(-1, 32, 32, 3)
# cat_dog_x_test = cat_dog_x_test.reshape(cat_dog_x_test.shape[0], cat_dog_x_test.shape[1], cat_dog_x_test.shape[2], 1)
# x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False).next()[0]

cat_dog_x_train = np.concatenate([cat_dog_x_train/255., x_augmented], axis=0)
cat_dog_y_train = np.concatenate([cat_dog_y_train, y_augmented], axis=0)
# cat_dog_y_train = to_categorical(cat_dog_y_train)
# cat_dog_y_test = to_categorical(cat_dog_y_test)
cat_dog_x_test = cat_dog_x_test/255.

path_save = 'd:/study_data/_save/cat_dog/'
np.save(path_save + 'keras58_8_cat_dog_x_train.npy', arr=cat_dog_x_train)
np.save(path_save + 'keras58_8_cat_dog_x_test.npy', arr=cat_dog_x_test)
np.save(path_save + 'keras58_8_cat_dog_y_train.npy', arr=cat_dog_y_train)
np.save(path_save + 'keras58_8_cat_dog_y_test.npy', arr=cat_dog_y_test)