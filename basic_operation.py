import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as session:
    print(session.run(a + b))
    print(session.run(a * b))

c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)

add = tf.add(c, d)
multiply = tf.multiply(c, d)

with tf.Session() as session:
    feed_dict: dict = {c: 2, d: 3}
    print(session.run(add, feed_dict))
    print(session.run(multiply, feed_dict))

matrix1 = tf.constant([[3., 3.]])  # 1 * 2

matrix2 = tf.constant([[2.], [2.]])  # 2 * 1

product = tf.matmul(matrix1, matrix2)

with tf.Session() as session:
    print(session.run(product))
