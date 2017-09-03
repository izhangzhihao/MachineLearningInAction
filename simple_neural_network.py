from typing import List

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist: Datasets = input_data.read_data_sets("data/MNIST", one_hot=True)
learning_rate: float = 0.01
train_epoch: int = 500
batch_size: int = 128
display_epoch: int = 100

neurons_layer_1: int = 256
neurons_layer_2: int = 256
input_features: int = 28 * 28
class_number: int = 10

X = tf.placeholder(tf.float32, shape=[None, input_features])
Y = tf.placeholder(tf.float32, shape=[None, class_number])


def new_weights(shape: List[int]):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length: int):
    return tf.Variable(tf.constant(0.05, shape=[length]))


weight_dict: dict = {
    "h1": new_weights([input_features, neurons_layer_1]),
    "h2": new_weights([neurons_layer_1, neurons_layer_2]),
    "output": new_weights([neurons_layer_2, class_number])
}

bias_dict: dict = {
    "h1": new_biases(neurons_layer_1),
    "h2": new_biases(neurons_layer_2),
    "output": new_biases(class_number)
}


def neural_net(x):
    layer1 = tf.add(tf.matmul(x, weight_dict['h1']), bias_dict['h1'])
    layer2 = tf.add(tf.matmul(layer1, weight_dict['h2']), bias_dict['h2'])
    return tf.add(tf.matmul(layer2, weight_dict['output']), bias_dict['output'])


logits = neural_net(X)

prediction = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(1, train_epoch + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        session.run(optimizer, {X: batch_x, Y: batch_y})

        if epoch % display_epoch == 0 or epoch == 0:
            loss, acc = session.run([cost, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(epoch) + ", Mini-batch Loss= " + "{:.4f}".format(
                loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    print("Testing Accuracy:", session.run(accuracy, feed_dict={X: mnist.test.images,
                                                                Y: mnist.test.labels}))
