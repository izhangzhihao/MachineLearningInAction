import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist: Datasets = input_data.read_data_sets("data/MNIST", one_hot=True)

learning_rate: float = 0.01
train_epoch: int = 25
batch_size: int = 64
display_step: int = 1

number_feature: int = 28 * 28
number_class: int = 10

X = tf.placeholder(tf.float32, shape=[None, number_feature])
Y = tf.placeholder(tf.float32, shape=[None, number_class])

W = tf.Variable(tf.truncated_normal(shape=[number_feature, number_class], stddev=0.05))
B = tf.ones(shape=[number_class]) * 0.01

logits = tf.matmul(X, W) + B

pred = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(train_epoch):
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            c = session.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Acc : " + str(acc.eval({X: mnist.test.images, Y: mnist.test.labels})))
