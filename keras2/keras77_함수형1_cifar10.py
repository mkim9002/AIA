import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
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

vgg16.trainable = True # Freeze weights

# Using Flatten layer
inputs = Input(shape=(32, 32, 3))
x = vgg16(inputs)
x = Flatten()(x)
x = Dense(100)(x)
outputs = Dense(10, activation='softmax')(x)

model_flatten = Model(inputs=inputs, outputs=outputs)
model_flatten.summary()

# Compile and train the model with Flatten layer
model_flatten.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_flatten.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Using GlobalAveragePooling2D layer
inputs = Input(shape=(32, 32, 3))
x = vgg16(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
outputs = Dense(10, activation='softmax')(x)

model_gap = Model(inputs=inputs, outputs=outputs)
model_gap.summary()

# Compile and train the model with GlobalAveragePooling2D layer
model_gap.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gap.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Compare performance
_, accuracy_flatten = model_flatten.evaluate(x_test, y_test, verbose=0)
_, accuracy_gap = model_gap.evaluate(x_test, y_test, verbose=0)

print("Accuracy with Flatten layer:", accuracy_flatten)
print("Accuracy with GlobalAveragePooling2D layer:", accuracy_gap)
 

#false
# Accuracy with Flatten layer: 0.5846999883651733
# Accuracy with GlobalAveragePooling2D layer: 0.5806000232696533

#True
# Accuracy with Flatten layer: 0.6901999711990356
# Accuracy with GlobalAveragePooling2D layer: 0.8029999732971191