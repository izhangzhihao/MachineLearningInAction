import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist: Datasets = input_data.read_data_sets("data/MNIST", one_hot=True)

learning_rate: float = 0.001
train_epoch: int = 200
batch_size: int = 128
display_epoch: int = 10
image_size: int = 28
input_features: int = image_size * image_size
class_number: int = 10
dropout: float = 0.75

X = tf.placeholder(tf.float32, [None, input_features])
Y = tf.placeholder(tf.float32, [None, class_number])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.05)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.05)),
    'wd1': tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.05)),
    'out': tf.Variable(tf.truncated_normal([1024, class_number], stddev=0.05))
}

biases = {
    'bc1': tf.Variable(tf.constant(0.05, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.05, shape=[64])),
    'bd1': tf.Variable(tf.constant(0.05, shape=[1024])),
    'out': tf.Variable(tf.constant(0.05, shape=[class_number]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    conv2 = maxpool2d(conv2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.dropout(fc1, dropout)
    fc1 = tf.nn.relu(fc1)
    return tf.add(tf.matmul(fc1, weights['out']), biases['out'])


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(1, train_epoch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        session.run(train_op, feed_dict={
            X: batch_x, Y: batch_y, keep_prob: 0.75})
        if epoch % display_epoch == 0 or epoch == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={
                X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("Epoch " + str(epoch) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Testing Accuracy:",
          session.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                           Y: mnist.test.labels[:256],
                                           keep_prob: 1.0}))
