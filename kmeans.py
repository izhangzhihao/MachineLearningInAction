from typing import List

from tensorflow.contrib.factorization import KMeans
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from numpy import ndarray

mnist_data: Datasets = input_data.read_data_sets('data/MNIST/', one_hot=True)

train_times: int = 50
batch_size: int = 1024
k: int = 5
number_class: int = 10
number_features: int = 28 * 28

X = tf.placeholder(tf.float32, shape=[None, number_features])
Y = tf.placeholder(tf.float32, shape=[None, number_class])

kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = kmeans.training_graph()

cluster_idx = cluster_idx[0]

avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

with tf.Session() as session:
    feed_dict: dict = {X: mnist_data.train.images}
    session.run(init_vars, feed_dict=feed_dict)
    session.run(init_op, feed_dict=feed_dict)

    for i in range(1, train_times + 1):
        _, d, idx = session.run([train_op, avg_distance, cluster_idx], feed_dict=feed_dict)
        if i % 10 == 0 or i == 1:
            print("Step : %i , avg distance %f " % (i, d))

    counts = np.zeros(shape=(k, number_class))
    # noinspection PyUnboundLocalVariable
    for i in range(len(idx)):
        counts[idx[i]] += mnist_data.train.labels[i]

    # noinspection PyUnresolvedReferences
    labels_map: List[ndarray[int]] = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = mnist_data.test.images, mnist_data.test.labels
    print("Test Accuracy:", session.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
