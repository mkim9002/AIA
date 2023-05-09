#실습
#lr 수정해서 epoch 100 번 이하로 줄여
# step =100, w




import tensorflow as tf
tf.set_random_seed(337) # 랜던 씨드 고정
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08) # 경사하강법 방식으로 optimizer를 최적화 시켜준다.
train = optimizer.minimize(loss) # 따라서 로스를 최저로 뽑음 
# model.complie(loss='mse', optimizer = 'sgd') 이것과 같다?

#3-2 훈련

loss_val_list =[]
w_val_list =[]

with tf.compat.v1.Session() as sess : # <- 위의 코드를 대체 / 아래 코드들 안으로 넣어준다. (with sess 안에 넣어줌) 범위 지정이라 close 안해줘도됨
    sess.run(tf.global_variables_initializer()) # 전체 변수 초기화

    #4 모델의 훈련
    epochs = 101
    for step in range(epochs):
        _ ,loss_val, w_val, b_val = sess.run([ train, loss, w , b ], 
                                             feed_dict = {x:[1,2,3,4,5], y:[2,4,6,8,10]}) 
        
        if step %20 ==0: # 나머지가 0 일때만 뽑을게
            print(step,loss_val,w_val,b_val)
            
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

    # subplot으로 그래프 그리기
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    axes[0].plot(loss_val_list)
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Loss')
    
    axes[1].plot(w_val_list)
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('weight')
    axes[1].set_title('Weight')
    
    axes[2].plot(w_val_list, loss_val_list)
    axes[2].set_xlabel('weights')


#subplot 으로 위 세개의 그래프 그려
    
    