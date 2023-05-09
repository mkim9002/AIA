import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 =tf.compat.v1.Variable(tf.random_normal([2]), name='weight')
print(변수)
#<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

    
#초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa =sess.run(변수)
print('aaa :', aaa) #aaa : [-1.5080816]
sess.close()

#초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) #텐서 플로우 데이터 형인 변수를 파이썬에서 볼수 있게
print('bbb :',bbb)
sess.close()

#초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc :', ccc)
sess.close()

# aaa : [-1.5080816   0.26086742]
# bbb : [-1.5080816   0.26086742]
# ccc : [-1.5080816   0.26086742]