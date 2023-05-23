#loss 와 weight 의 관계를 그려
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import time
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the model
model = Sequential()
model.add(Conv2D(64, (2, 2), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(32, (2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model
start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
end = time.time()

# Save the model
model.save('./_save/keras_mnist_model.h5')

# Load the model from the saved file
model = load_model('./_save/keras_mnist_model.h5')

# Get the training history from the loaded model
training_loss = hist.history['loss']
validation_loss = hist.history['val_loss']
accuracy = hist.history['acc']
validation_accuracy = hist.history['val_acc']

# Plot the loss graph
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(training_loss, marker=',', c='red', label='loss')
plt.plot(validation_loss, marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')

# Plot the accuracy graph
plt.subplot(2, 1, 2)
plt.plot(accuracy, marker=',', c='red', label='acc')
plt.plot(validation_accuracy, marker=',', c='blue', label='val_acc')
plt.grid()
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['acc', 'val_acc'])

plt.tight_layout()
plt.show()
