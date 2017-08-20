import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST", one_hot=True)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weight = tf.Variable(tf.zeros([img_size_flat, num_classes]))
bias = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x, weight) + bias

y_pre = tf.nn.softmax(logits)
y_pre_cls = tf.argmax(y_pre, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

correct_prediction = tf.equal(y_pre_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128

feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: np.argmax(data.test.labels, axis=1)}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())


    def optimize(num_iterations):
        for i in range(num_iterations):
            x_batch, y_true_batch = data.train.next_batch(batch_size)
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            session.run(optimizer, feed_dict_train)


    def print_accuracy():
        acc = session.run(accuracy, feed_dict=feed_dict_test)
        print("Accuracy on test-set: {0:.1%}".format(acc))


    optimize(100)
    print_accuracy()
