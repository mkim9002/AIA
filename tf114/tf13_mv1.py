import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

#실습 만들어

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal{[1]})
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal{[1]})
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal{[1]})
b= tf.compat.v1.Variable(tf.compat.v1.random_normal{[1]})

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 +b

#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#4. 학습
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(5001):
_, loss_val, hypo_val = sess.run([train, loss, hypothesis], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
if step % 500 == 0:
print(f"Step: {step}, Loss: {loss_val}, Prediction: {hypo_val}")

#5. 결과 예측
print(sess.run(hypothesis, feed_dict={x1: [80.], x2: [90.], x3: [85.]}))

#6. 세션 종료
sess.close()
