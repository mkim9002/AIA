import tensorflow as tf
tf.set_random_seed(337) # 랜던 씨드 고정

#1. 데이터
x = [1,2,3,4,5]
y = [2,4,6,8,10]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # random_uniform -> 균등분포 (엔빵), random_normal -> 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # [1] 스칼라 의미 크기가 1인 1차원 텐서

####################### 텐서 자체에서 제공하는거라 Variable 안쓰고 쓸수있음 #################################
# w = tf.random_normal([1])
# b = tf.random_normal([1])
###################################################

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w)) #[-0.4121612] 초기값

#################### with로 정의 & 변수 초기화 ###########################

# with tf.compat.v1.Session() as sess : 
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(w))

####################################################
#2. 모델 구성
# y = wx + b -> 사실은 쌤이 그징말 했어 
# y = xw + b이다 행렬은 순서가 중요하기 때문에 값이 다르다. ( 데이터에 웨이트가 연산되는거지 웨이트가 먼저있고 데이터가 연산되는게 아니니까)

# y를 찾는 부분
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # hypothesis - y 의 제곱의 평균 -> mse (loss지표)

# 웨이트는? 웨이트 빼기 고정된 러닝레이트와 로스를 웨이트로 편미분 해준값을 곱한다 -> 값이 계속 작아진다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법 방식으로 optimizer를 최적화 시켜준다.
train = optimizer.minimize(loss) # 따라서 로스를 최저로 뽑음 
# model.complie(loss='mse', optimizer = 'sgd') 이것과 같다?

#3-2 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess : # <- 위의 코드를 대체 / 아래 코드들 안으로 넣어준다. (with sess 안에 넣어줌) 범위 지정이라 close 안해줘도됨
    sess.run(tf.global_variables_initializer()) # 전체 변수 초기화

    #4 모델의 훈련
    epochs = 2001
    for step in range(epochs):
        sess.run(train) # 에포당 그래프가 한번에 모두 연산됨
        if step %20 ==0:
            print(step,sess.run(loss), sess.run(w), sess.run(b))

# sess.close()