import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.data import Dataset

from tensorflow.examples.tutorials.mnist import input_data

mnist: Dataset = input_data.read_data_sets('data/MNIST')

learning_rate: float = 0.01
training_epochs: int = 5
batch_size: int = 256
display_step: int = 1

input_features: int = 28 * 28
X = tf.placeholder(tf.float32, [None, input_features])

hidden_layer_1_features: int = 256
hidden_layer_2_features: int = 128

weights: dict = {
    'encoder_h1': tf.Variable(tf.random_normal([input_features, hidden_layer_1_features])),
    'encoder_h2': tf.Variable(tf.random_normal([hidden_layer_1_features, hidden_layer_2_features])),
    'decoder_h1': tf.Variable(tf.random_normal([hidden_layer_2_features, hidden_layer_1_features])),
    'decoder_h2': tf.Variable(tf.random_normal([hidden_layer_1_features, input_features]))
}

biases: dict = {
    'encoder_b1': tf.Variable(tf.random_normal([hidden_layer_1_features])),
    'encoder_b2': tf.Variable(tf.random_normal([hidden_layer_2_features])),
    'decoder_b1': tf.Variable(tf.random_normal([hidden_layer_1_features])),
    'decoder_b2': tf.Variable(tf.random_normal([input_features]))
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, c = session.run([optimizer, cost], feed_dict={X: batch_x})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

    result = session.run(y_pred, feed_dict={X: mnist.test.images[:10]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(result[i], (28, 28)))
    plt.show()
