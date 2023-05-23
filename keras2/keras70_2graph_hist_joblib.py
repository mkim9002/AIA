#실습 훈련시키지 말고 ,가중치든 뭐든 땡겨다가, 그래프를 그려
#eppoch 와 loss/acc 그래프


from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import time

import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=5, verbose=True)

start = time.time()
hist = model.fit(x_train, y_train)
end = time.time()

# Save the model using joblib
joblib.dump(model, './_save/mlp_mnist_model.joblib')

# Load the model from the saved file
model = joblib.load('./_save/mlp_mnist_model.joblib')

# Get the training history from the loaded model
training_loss = hist.loss_curve_

# Evaluate the model on the validation set
val_loss = model.score(x_val, y_val)
val_accuracy = model.score(x_val, y_val)

# Plot the loss graph
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(training_loss, marker=',', c='red', label='loss')
plt.grid()
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')

# Plot the accuracy graph
plt.subplot(2, 1, 2)
plt.plot(val_loss, marker=',', c='red', label='val_loss')
plt.plot(val_accuracy, marker=',', c='blue', label='val_accuracy')
plt.grid()
plt.title('Validation')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epochs')
plt.legend(['val_loss', 'val_accuracy'])

plt.tight_layout()
plt.show()
