import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

# 1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],   #2
          [0,0,1],
          [0,0,1],
          [0,1,0],   #1
          [0,1,0],
          [0,1,0],
          [1,0,0],   #0
          [1,0,0]]

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.zeros([1,3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.matmul(x, w) + b

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))  # MSE
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_data})
    print("Predictions:", y_pred)

    # 평가 지표 계산
    r2 = r2_score(y_data, y_pred)
    mse = mean_squared_error(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)
    accuracy = accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_pred, axis=1))

    print("R2 Score:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("Accuracy:", accuracy)
