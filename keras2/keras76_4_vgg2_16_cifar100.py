import numpy as np
import tensorflow as tf
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

# Define the strategy to use all available GPUs
strategy = tf.distribute.MirroredStrategy()

# Create the model within the strategy scope
with strategy.scope():
    # Using Flatten layer
    model_flatten = Sequential()
    model_flatten.add(vgg16)
    model_flatten.add(Flatten())
    model_flatten.add(Dense(100))
    model_flatten.add(Dense(100, activation='softmax'))
    
    model_flatten.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Using GlobalAveragePooling2D layer
    model_gap = Sequential()
    model_gap.add(vgg16)
    model_gap.add(GlobalAveragePooling2D())
    model_gap.add(Dense(100))
    model_gap.add(Dense(100, activation='softmax'))

    model_gap.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the models using all available GPUs
batch_size_per_replica = 32
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(global_batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(global_batch_size)

model_flatten.fit(train_dataset, epochs=10, validation_data=test_dataset)
model_gap.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Compare performance
_, accuracy_flatten = model_flatten.evaluate(test_dataset)
_, accuracy_gap = model_gap.evaluate(test_dataset)

print("Accuracy with Flatten layer:", accuracy_flatten)
print("Accuracy with GlobalAveragePooling2D layer:", accuracy_gap)