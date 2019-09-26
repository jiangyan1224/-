import tensorflow as tf
x=tf.constant([[1.0,2.0]])
y=tf.constant([[3.0],[4.0]])
w=tf.matmul(x,y)
print(w)
with tf.Session() as sess:
    print(sess.run(w))