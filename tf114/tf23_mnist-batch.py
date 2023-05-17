from tensorflow.keras.datasets import mnist
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28) / 255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]) / 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w1 = tf.Variable(tf.random_normal([784, 128]), name='w1')
b1 = tf.Variable(tf.zeros([128]), name='b1')
layer1 = tf.matmul(x, w1) + b1
dropout1 = tf.nn.dropout(layer1, rate=0.3)

w2 = tf.Variable(tf.random_normal([128, 64]), name='w2')
b2 = tf.Variable(tf.zeros([64]), name='b2')
layer2 = tf.nn.selu(tf.matmul(dropout1, w2) + b2)

w3 = tf.Variable(tf.random_normal([64, 32]), name='w3')
b3 = tf.Variable(tf.zeros([32]), name='b3')
layer3 = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([32, 10]), name='w4')
b4 = tf.Variable(tf.zeros([10]), name='b4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

batch_size = 100
total_batch = int(len(x_train) / batch_size)
epochs = 100

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            start = i * batch_size
            end = (i + 1) * batch_size

            batch_x, batch_y = x_train[start:end], y_train[start:end]

            _, c = sess.run([train, loss], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print("Epoch:", epoch + 1, "loss =", "{:.9f}".format(avg_cost))

    print("Training completed!")
