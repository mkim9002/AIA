######현재 버전이 1.0 이면 그냥 출력
######현제 버전이 2.0 이면 즉시실행모드를 끄고 출력
###### if문 써서 1번 쏘스를 변경!!


import tensorflow as tf

print(tf.__version__)

if tf.__version__[0] == '1':
    # TensorFlow 1.x에서는 그냥 출력
    print(tf.executing_eagerly())
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    aaa = tf.constant('hello world')
    print(sess.run(aaa))
else:
    # TensorFlow 2.x에서는 즉시 실행모드 끄고 출력
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    aaa = tf.constant('hello world')
    print(sess.run(aaa))





