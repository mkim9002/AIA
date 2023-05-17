import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 2. 모델 구성
w1 = tf.Variable(tf.random_normal([784, 10], dtype=tf.float32), name='weight1')
b1 = tf.Variable(tf.zeros([10], dtype=tf.float32), name='bias1')
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([10, 7], dtype=tf.float32), name='weight2')
b2 = tf.Variable(tf.zeros([7], dtype=tf.float32), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([7, 1], dtype=tf.float32), name='weight3')
b3 = tf.Variable(tf.zeros([1], dtype=tf.float32), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # Cross Entropy Loss

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})
    print("Predictions:", y_pred)

    # 평가 지표 계산
    y_pred_label = np.argmax(y_pred, axis=1)
    y_test_label = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_label, y_pred_label)
    print("Accuracy:", accuracy)
