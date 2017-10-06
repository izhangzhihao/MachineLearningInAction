import array
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist: Dataset = input_data.read_data_sets('data/MNIST')

num_steps: int = 100000
batch_size: int = 128
learning_rate: float = 0.0002

input_features: int = 28 * 28
gen_hidden_features: int = 256
disc_hidden_features: int = 256
noise_features: int = 100


def glorot_init(shape: array):
    return tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0))


weights: dict = {
    'gen_hidden': tf.Variable(glorot_init([noise_features, gen_hidden_features])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_features, input_features])),
    'disc_hidden': tf.Variable(glorot_init([input_features, disc_hidden_features])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_features, 1]))
}

biases: dict = {
    'gen_hidden': tf.Variable(tf.zeros([gen_hidden_features])),
    'gen_out': tf.Variable(tf.zeros([input_features])),
    'disc_hidden': tf.Variable(tf.zeros([disc_hidden_features])),
    'disc_out': tf.Variable(tf.zeros([1]))
}


def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.sigmoid(out_layer)
    return out_layer


gen_input = tf.placeholder(tf.float32, shape=[None, noise_features], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, input_features], name='disc_input')

gen_sample = generator(gen_input)

disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

gen_loss = - tf.reduce_mean(tf.log(disc_fake))
disc_loss = - tf.reduce_mean(tf.log(disc_real) + tf.log(1.0 - disc_fake))

optimizer_gen = tf.train.AdamOptimizer(learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate)

gen_vars: list = [
    weights['gen_hidden'], weights['gen_out'],
    biases['gen_hidden'], biases['gen_out']
]

disc_vars: list = [
    weights['disc_hidden'], weights['disc_out'],
    biases['disc_hidden'], biases['disc_out']
]

train_gen_op = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc_op = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        # noinspection PyUnresolvedReferences
        z = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_features])

        session.run([train_gen_op, train_disc_op],
                    feed_dict={
                        disc_input: batch_x,
                        gen_input: z
                    }
                    )
        if i % 500 == 0 or i == 1:
            _, _, gl, dl = session.run([train_gen_op, train_disc_op, gen_loss, disc_loss],
                                       feed_dict={
                                           disc_input: batch_x,
                                           gen_input: z
                                       }
                                       )
            print("Step :\t" + str(i) + "\t generator loss: " + str(gl) + "\t discriminator loss ： " + str(dl))

    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # noinspection PyUnresolvedReferences
        z = np.random.uniform(-1.0, 1.0, size=[4, noise_features])
        g = session.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        g = -1 * (g - 1)
        for j in range(4):
            img = np.reshape(np.repeat(
                g[j][:, :, np.newaxis], 3, axis=2
            ), newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.show()
