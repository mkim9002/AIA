import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# VGG16 model with pre-trained weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False # Freeze weights

# Using Flatten layer
model_flatten = Sequential()
model_flatten.add(vgg16)
model_flatten.add(Flatten())
model_flatten.add(Dense(100))
model_flatten.add(Dense(100, activation='softmax'))

model_flatten.summary()

print(len(model_flatten.weights))
print(len(model_flatten.trainable_weights))

# Compile and train the model with Flatten layer
model_flatten.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_flatten.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Using GlobalAveragePooling2D layer
model_gap = Sequential()
model_gap.add(vgg16)
model_gap.add(GlobalAveragePooling2D())
model_gap.add(Dense(100))
model_gap.add(Dense(100, activation='softmax'))

model_gap.summary()

print(len(model_gap.weights))
print(len(model_gap.trainable_weights))

# Compile and train the model with GlobalAveragePooling2D layer
model_gap.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gap.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Compare performance
_, accuracy_flatten = model_flatten.evaluate(x_test, y_test, verbose=0)
_, accuracy_gap = model_gap.evaluate(x_test, y_test, verbose=0)

print("Accuracy with Flatten layer:", accuracy_flatten)
print("Accuracy with GlobalAveragePooling2D layer:", accuracy_gap)

