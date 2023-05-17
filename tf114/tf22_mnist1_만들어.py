#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
from keras.utils import to_categorical
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
print(x_train.shape), (y_train.shape)
print(x_test.shape), (y_test.shape)

#살습 만들기
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# model.add(Dense(10,input_shape=2))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 10], dtype=tf.float32), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]
layer1 = tf.compat.v1.matmul(x, w1) + b1

# model.add(Dense(7))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 7], dtype=tf.float32), name= 'weight2')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([7], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2) + b2)


# model.add(Dense(1,activation ='sigmoid'))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([7, 1], dtype=tf.float32), name= 'weight2')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3) + b3) 



# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # Cross Entropy Loss

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 1000
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
