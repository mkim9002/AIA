import tensorflow as tf
print(tf.__version__)

print("hello world")

aaa = tf.constant('hello world')
print(aaa)      #Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(aaa))    #b'hello world'
