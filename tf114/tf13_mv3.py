import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

tf.compat.v1.set_random_seed(337)

x_data = [[73, 51, 65], [92, 98, 11], [89, 31, 33], [99, 33, 100], [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))


#2. 모델 
hypothesis = tf.compat.v1.matmul(x, w) + b


#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 4. 훈련  
epochs = 1000
for step in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if step % 100 == 0:
        print("Step:", step, "Loss:", loss_val)

y_pred = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, y_pred)
mse = mean_squared_error(y_data, y_pred)

print("R2 Score:", r2)
print("MSE:", mse)
