import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist: Dataset = input_data.read_data_sets('data/MNIST')

num_steps: int = 20000
batch_size: int = 32
input_features: int = 28 * 28
gen_hidden_features: int = 256
disc_hidden_features: int = 256
noise_features: int = 200


def generator(x, reuse=False):
    """
    x : noise
    :rtype: image
    """
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])

        # [None,14,14,64]
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)

        # [None,28,28,1]
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)

        x = tf.nn.sigmoid(x)

    return x


def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 4)
        x = tf.nn.tanh(x)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 4)
        x = tf.nn.tanh(x)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
    return x


noise_input = tf.placeholder(tf.float32, [None, noise_features])
real_image_input = tf.placeholder(tf.float32, [None, 28, 28, 1])

gen_sample = generator(noise_input)

disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_contat = tf.concat([disc_real, disc_fake], axis=0)

stacked_gen = discriminator(gen_sample, reuse=True)

disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

disc_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_contat, labels=disc_target
    )
)

gen_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=stacked_gen, labels=gen_target
    )
)

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

gen_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'
)

disc_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'
)

train_gen_op = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc_op = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])

        z = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_features])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0
        )

        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        feed_dict: dict = {
            real_image_input: batch_x,
            noise_input: z,
            disc_target: batch_disc_y,
            gen_target: batch_gen_y
        }

        session.run([train_gen_op, train_disc_op], feed_dict=feed_dict)

        if i % 100 == 0 or i == 1:
            _, _, gl, dl = session.run([train_gen_op, train_disc_op, gen_loss, disc_loss], feed_dict=feed_dict)
            print('Step: \t{0}\t Generator loss: \t{1}\t Discriminator loss: \t{2}'.format(str(i), str(gl), str(dl)))

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_features])
        g = session.run(gen_sample, feed_dict={noise_input: z})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
