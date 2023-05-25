#가중치 등록과 동결하지 않았을떄, 그라고 원래와 성능 비교
# Flatten과  GAP 과 차이

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# VGG16 model with pre-trained weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable =True

# Using Flatten layer
model_flatten = Sequential()
model_flatten.add(vgg16)
model_flatten.add(Flatten())
model_flatten.add(Dense(100))
model_flatten.add(Dense(10, activation='softmax'))

model_flatten.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_flatten.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Using GlobalAveragePooling2D layer
model_gap = Sequential()
model_gap.add(vgg16)
model_gap.add(GlobalAveragePooling2D())
model_gap.add(Dense(100))
model_gap.add(Dense(10, activation='softmax'))

model_gap.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gap.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Compare performance
_, accuracy_flatten = model_flatten.evaluate(x_test, y_test, verbose=0)
_, accuracy_gap = model_gap.evaluate(x_test, y_test, verbose=0)

print("Accuracy with Flatten layer:", accuracy_flatten)
print("Accuracy with GlobalAveragePooling2D layer:", accuracy_gap)

# False
# Accuracy with Flatten layer: 0.5820000171661377
# Accuracy with GlobalAveragePooling2D layer: 0.5813999772071838

#True
# Accuracy with Flatten layer: 0.6284999847412109
# Accuracy with GlobalAveragePooling2D layer: 0.8041999936103821