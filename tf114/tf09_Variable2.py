import tensorflow as tf
tf.compat.v1.set_random_seed(337)

변수 =tf.compat.v1.Variable(tf.random_normal([2]), name='weight')
print(변수)
#<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

#실습
#08_2 카피해서 만들어

###################### 1. Session() //  sess.run(변수)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa =sess.run(변수)
print('aaa :', aaa)
s



##################### 2. Session() // 변수.eval(session=sess)




#################### 3. InteractiveSession() //변수.eval()





