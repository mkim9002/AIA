import tensorflow as tf
tf.set_random_seed(337) # 랜던 씨드 고정

#1. 데이터
x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # random_uniform -> 균등분포 (엔빵), random_normal -> 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # [1] 스칼라 의미 크기가 1인 1차원 텐서

# y를 찾는 부분
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # hypothesis - y 의 제곱의 평균 -> mse (loss지표)

# 웨이트 빼기 고정된 러닝레이트와 로스를 웨이트로 편미분 해준값을 곱한다 -> 값이 계속 작아진다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법 방식으로 optimizer를 최적화 시켜준다.
train = optimizer.minimize(loss) # 따라서 로스를 최저로 뽑음 
# model.complie(loss='mse', optimizer = 'sgd') 이것과 같다?

#3-2 훈련
with tf.compat.v1.Session() as sess : # <- 위의 코드를 대체 / 아래 코드들 안으로 넣어준다. (with sess 안에 넣어줌) 범위 지정이라 close 안해줘도됨
    sess.run(tf.global_variables_initializer()) # 전체 변수 초기화

    #4 모델의 훈련
    epochs = 2001
    for step in range(epochs):
        # sess.run(train) # 에포당 그래프가 한번에 모두 연산됨
        _ ,loss_val, w_val, b_val = sess.run([ train, loss, w , b ], 
                                             feed_dict = {x:[1,2,3,4,5], y:[2,4,6,8,10]}) # 반환값으로 잡아줘 / 안그러면 아래 loss의 sess.run에서 마찬가지로 x,y 값이 필요해서 loss 안뽑힘 
        
        if step %20 ==0: # 나머지가 0 일때만 뽑을게
            # print(step,sess.run(loss_val), sess.run(w_val), sess.run(b_val))
            print(step,loss_val,w_val,b_val)
            
############## 실습 ###############
# x_data = [6,7,8]
#예측값을 구해라
###################################

    # x_data = tf.placeholder(tf.float32, shape = [None])
    # y_predict = x_data * w_val + b_val
    # print('y_predict : ',sess.run(y_predict,feed_dict={x_data:[6,7,8]}))
    
    '''
    y_predict :  [12.000409 14.000581 16.000753]
    '''
    
    x_test = tf.placeholder(tf.float32, shape = [None])
    y_predict = x_test *w_val+ b_val
    
    sess.run(y_predict, feed_dict={x_test:[6,7,8]})
    
    