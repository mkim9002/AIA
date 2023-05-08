import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)


#실습
#덧셈 node3
node3 =tf.add(node1,node2)
node4 = tf.subtract(node1,node2)
node5 = tf.multiply(node1,node2)
node6 = tf.divide(node1,node2)

sess = tf.compat.v1.Session()
print(sess.run(node3)) 

# 뺄셈 결과 출력
print(sess.run(node4)) 

# 곱셈 결과 출력
print(sess.run(node5))

# 나눗셈 결과 출력
print(sess.run(node6)) 

#만들기