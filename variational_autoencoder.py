import array

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist: Dataset = input_data.read_data_sets('data/MNIST')

learning_rate: float = 0.0001
num_steps: int = 50000
batch_size: int = 64

input_features: int = 28 * 28
hidden_features: int = 512
latent_features: int = 2


def glorot_init(shape: array):
    return tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0))


weights: dict = {
    'encoder_h': tf.Variable(glorot_init([input_features, hidden_features])),
    'z_mean': tf.Variable(glorot_init([hidden_features, latent_features])),
    'z_std': tf.Variable(glorot_init([hidden_features, latent_features])),
    'decoder_h': tf.Variable(glorot_init([latent_features, hidden_features])),
    'decoder_out': tf.Variable(glorot_init([hidden_features, input_features]))
}

biases: dict = {
    'encoder_h': tf.Variable(glorot_init([hidden_features])),
    'z_mean': tf.Variable(glorot_init([latent_features])),
    'z_std': tf.Variable(glorot_init([latent_features])),
    'decoder_h': tf.Variable(glorot_init([hidden_features])),
    'decoder_out': tf.Variable(glorot_init([input_features]))
}

X = tf.placeholder(tf.float32, shape=[None, input_features])
encoder = tf.matmul(X, weights['encoder_h']) + biases['encoder_h']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

eps = tf.random_normal(
    tf.shape(z_std),
    name='epsilon'
)

z = z_mean + tf.exp(z_std / 2) * eps

decoder = tf.matmul(z, weights['decoder_h']) + biases['decoder_h']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


def vae_loss(x_reconstructed, x_true):
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


loss_op = vae_loss(decoder, X)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1, num_steps + 1):
        # noinspection PyUnresolvedReferences
        batch_x, _ = mnist.train.next_batch(batch_size)
        _, loss = session.run([train_op, loss_op], feed_dict={
            X: batch_x
        })
        if i % 200 == 0 or i == 1:
            print('Step:\t' + str(i) + "\t loss:" + str(loss))

    noise_input = tf.placeholder(tf.float32, shape=[None, latent_features])
    decoder = tf.matmul(noise_input, weights['decoder_h']) + biases['decoder_h']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = session.run(decoder, feed_dict={
                noise_input: z_mu
            })
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.show()
