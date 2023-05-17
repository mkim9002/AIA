#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential

print(keras.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape), (y_train.shape)
# print(x_test.shape), (y_test.shape)

x_train = x_train.reshape(60000, 28*28)/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape), (y_train.shape) #(60000, 784)
print(x_test.shape), (y_test.shape) #(10000, 784)

#2. 모델구성
x = tf.compat.v1.placeholder('float',[None, 784])
y = tf.compat.v1.placeholder('float',[None,10])

w1 = tf.Variable(tf.random_normal([784,128]), name='w1')
b1 = tf.Variable(tf.zeros([128]),name='b1')
layer1 = tf.compat.v1.matmul(x,w1) +b1
dropout1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.Variable(tf.random_normal([128,64]), name='w2')
b2 = tf.Variable(tf.zeros([64]),name='b2')
layer2 = tf.nn.selu(tf.compat.v1.matmul(dropout1,w2) +b2)


w3 = tf.Variable(tf.random_normal([64,32]), name='w3')
b3 = tf.Variable(tf.zeros([32]),name='b3')
layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2,w3) +b3)


w4 = tf.Variable(tf.random_normal([32,10]), name='w4')
b4 = tf.Variable(tf.zeros([10]),name='b4')
hypothesis = tf.nn.softmax(tf.matmul(layer3,w4) +b4)
# hypothesis = tf.matmul(layer3,w4) +b4


#3. 컴파일 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # MSE

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 10
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

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
