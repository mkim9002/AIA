import tensorflow as tf
print(tf.__version__)     #1.14.0
#즉시 실행모드
print(tf.executing_eagerly())   #False 
# 1.14.0
# False
# 2.7.3
# True

# tf.compat.v1.disable_eager_execution()  #즉시 실행모드 끔/ 텐서2.0에서 1.0 방식으로
tf.compat.v1.enable_eager_execution()

aaa =tf.constant('hello world')

sess = tf.compat.v1.Session()
# print(sess.run(aaa))  #2.대에는 see.run 이 없어짐




