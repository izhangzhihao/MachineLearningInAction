from typing import List

import numpy as np
import tensorflow as tf
from numpy import random
import matplotlib.pyplot as plot

learning_rate: float = 0.01
train_epochs: int = 1000
display_steps: int = 100

train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])

train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples: int = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(random.randn(), name="Weight")
B = tf.Variable(random.randn(), name="Bias")

pred = tf.add(tf.multiply(X, W), B)

cost = tf.reduce_sum(tf.pow(pred - Y, 2) / (2 * n_samples))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(train_epochs):
        session.run(optimizer, {X: train_X, Y: train_Y})

        if epoch % display_steps == 0 or epoch == 0:
            loss = session.run(cost, {X: train_Y, Y: train_Y})
            print("Epoch:" + str(epoch) + ", Loss : " + str(loss))

    plot.plot(train_X, train_Y, 'ro', label="Original Data")
    plot.plot(train_X, session.run(W) * train_X + session.run(B), label="Fitted Line")
    plot.legend()
    plot.show()
