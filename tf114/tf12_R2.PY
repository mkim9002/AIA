import tensorflow as tf

x_train = [1,2,3]
y_train = [4,5,6]
x_test = [1,2,3]
y_test = [4,5,6]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))   #mse

###############옵티마이저###########

lr=0.01  # 학습률을 낮추어 보정 계수 갱신이 더욱 부드럽게 일어나도록 함
gradient = tf.reduce_mean((hypothesis - y) * x)  # hypothesis - y 로 수정
descent = w - lr * gradient
update = w.assign(descent)
###############옵티마이저###########

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(1001):  # 더 많은 epoch 수행
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x: x_train, y: y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

############ [실습] R2, mae 만들어 ###################

from sklearn.metrics import r2_score, mean_absolute_error

# Generate predictions
# predictions = w_history[-1] * x_train
y_predict = x_test *w_v
print()






# Calculate R2 score
r2 = r2_score(y_train, predictions)

# Calculate MAE
mae = mean_absolute_error(y_train, predictions)


# Print the results
print("R2 score:", r2)
print("MAE:", mae)

