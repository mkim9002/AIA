import tensorflow as tf
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

tf.set_random_seed(4145)

# 1 데이터
random_state = 1234

data_list = [load_iris,
             load_breast_cancer,
             load_wine,
             load_digits]

for d in range(len(data_list)):
    try:
        x, y = data_list[d](return_X_y = True)
        # if d < 2:
        #     x, y = data_list[d](return_X_y = True)
        y = y.reshape(-1, 1) # (442, 1)
        # else:
        #     x, y = data_list[d]
        #     y = y.values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = random_state, shuffle = True)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        n_features = x_train.shape[1]
        
        n_neurons1 = 5
        n_neurons2 = 5
        
        x_p = tf.compat.v1.placeholder(tf.float32, shape = [None, n_features])
        y_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

        w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, n_neurons1], name = 'weight1'))
        b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons1], name = 'bias1'))
        layer1 = tf.compat.v1.matmul(x_p, w1) + b1
        
        w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight2'))
        b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias2'))
        layer2 = tf.compat.v1.matmul(layer1, w2) + b2
        
        w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight3'))
        b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias3'))
        layer3 = tf.compat.v1.matmul(layer2, w3) + b3
        
        w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight4'))
        b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias4'))
        layer4 = tf.compat.v1.matmul(layer3, w4) + b4
        
        w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight5'))
        b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias5'))
        layer5 = tf.compat.v1.matmul(layer4, w5) + b5
        
        w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight6'))
        b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias6'))
        hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer5, w6) + b6)
        

        # 3-1 컴파일
        loss = -tf.reduce_mean(y_p * tf.log(hypothesis + 0.3) + (1 - y_p) * tf.log(1 - hypothesis + 0.4))      

        train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)

        # 3-2 훈련
        sess = tf.compat.v1.Session()

        sess.run(tf.compat.v1.global_variables_initializer())

        epochs = 101
        
        for s in range(epochs):
            _, loss_val = sess.run([train, loss], feed_dict = {x_p : x_train, y_p : y_train})
            
            if s % 20 == 0:  # Print loss every 200 steps
                print(f'step : {s}, loss : {loss_val}')
                
                y_predict = sess.run(hypothesis, feed_dict = {x_p : x_test})
                

                y_pred = np.argmax(y_predict, axis=1)
                y_true = y_test
        # 4 평가
        acc = accuracy_score(y_true, y_pred)
        print(f'데이터 : {d}, acc_score : {acc}')
    except ValueError as ve:
        print(f'데이터 : {d}, 에러다 ㅋㅋ : {ve}') # 어떤 데이터셋에서 에러가 발생하는지 확인