import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data

mnist: DataSet = input_data.read_data_sets('data/MNIST', one_hot=True)

learning_rate: float = 0.001
training_iters: int = 10000
batch_size: int = 64
display_step: int = 20

input_features: int = 28 * 28
classes: int = 10
dropout: int = 0.75

x = tf.placeholder(tf.float32, [None, input_features])
y = tf.placeholder(tf.float32, [None, classes])
keep_prop = tf.placeholder(tf.float32)


def conv2d(name, input, w, b):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME'),
            b
        ),
        name=name
    )


def max_pool(name, input, k):
    return tf.nn.max_pool(
        input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name
    )


def dropout_and_max_pool(name, input, k, dropout):
    return max_pool(name, tf.nn.dropout(input, dropout), k)


def norm(name, input, size=4):
    return tf.nn.lrn(
        input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name
    )


def alex_net(input, weights, biases, dropout):
    X = tf.reshape(input, shape=[-1, 28, 28, 1])

    # [None, 28, 28, 64]
    conv1 = conv2d('Conv1', X, weights['wc1'], biases['bc1'])
    # [None, 14, 14, 64]
    pool1 = dropout_and_max_pool('Pool1', conv1, k=2, dropout=dropout)
    norm1 = norm('Norm1', pool1, size=4)

    # [None, 14, 14, 128]
    conv2 = conv2d('Conv2', norm1, weights['wc2'], biases['bc2'])
    # [None, 7, 7, 128]
    pool2 = dropout_and_max_pool('Pool2', conv2, k=2, dropout=dropout)
    norm2 = norm('Norm2', pool2, size=4)

    # [None, 7, 7, 256]
    conv3 = conv2d('Conv3', norm2, weights['wc3'], biases['bc3'])
    # [None, 4, 4, 256]
    pool3 = dropout_and_max_pool('Pool3', conv3, k=2, dropout=dropout)
    norm3 = norm('Norm3', pool3, size=4)

    # [None, 4096]
    dense1 = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]])
    # [None, 1024]
    dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1')
    # [None, 1024]
    dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')
    # [None, 10]
    return tf.matmul(dense2, weights['out']) + biases['out']


weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.05)),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.05)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.05)),
    'wd1': tf.Variable(tf.truncated_normal([4 * 4 * 256, 1024], stddev=0.05)),
    'wd2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.05)),
    'out': tf.Variable(tf.truncated_normal([1024, classes], stddev=0.05))
}

biases = {
    'bc1': tf.Variable(tf.constant(0.05, shape=[64])),
    'bc2': tf.Variable(tf.constant(0.05, shape=[128])),
    'bc3': tf.Variable(tf.constant(0.05, shape=[256])),
    'bd1': tf.Variable(tf.constant(0.05, shape=[1024])),
    'bd2': tf.Variable(tf.constant(0.05, shape=[1024])),
    'out': tf.Variable(tf.constant(0.05, shape=[classes]))
}

pred = alex_net(x, weights, biases, keep_prop)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        session.run(
            optimizer, feed_dict={x: batch_x, y: batch_y, keep_prop: dropout}
        )
        if step % display_step == 0:
            acc = session.run(
                accuracy, feed_dict={x: batch_x, y: batch_y, keep_prop: 1.0}
            )
            loss = session.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prop: 1.0})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Testing Accuracy:",
          session.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prop: 1.0}))
