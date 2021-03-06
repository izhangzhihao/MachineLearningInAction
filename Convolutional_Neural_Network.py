from typing import Tuple, List
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from os import path, makedirs
from tensorflow import Variable
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

# Convolutional Layer 1.
from tensorflow.python.training.saver import Saver

filter_size1: int = 5  # Convolution filters are 5 x 5 pixels.
num_filters1: int = 16  # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2: int = 5  # Convolution filters are 5 x 5 pixels.
num_filters2: int = 36  # There are 36 of these filters.

fc_size: int = 128  # Number of neurons in fully-connected layer.

data: Datasets = input_data.read_data_sets('data/MNIST/', one_hot=True)

img_size: int = 28
img_size_flat: int = img_size * img_size
img_shape: Tuple[int, int] = (img_size, img_size)
num_channels: int = 1
num_classes: int = 10


def new_weights(shape: List[int]):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length: int):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(pre_layer,
                   num_input_channels: int,
                   filter_size: int,
                   num_filters: int,
                   use_pooling: bool = True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape: List[int] = [filter_size, filter_size, num_input_channels, num_filters]

    weights: Variable = new_weights(shape)

    biases: Variable = new_biases(num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=pre_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.
    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(pre_layer,
                 num_inputs: int,
                 num_outputs: int,
                 use_relu: bool = True):
    weights: Variable = new_weights([num_inputs, num_outputs])
    biases: Variable = new_biases(num_outputs)

    layer = tf.matmul(pre_layer, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1 = new_conv_layer(pre_layer=x_image,
                             num_input_channels=num_channels,
                             filter_size=filter_size1,
                             num_filters=num_filters1,
                             use_pooling=True)

layer_conv2 = new_conv_layer(pre_layer=layer_conv1,
                             num_input_channels=num_filters1,
                             filter_size=filter_size2,
                             num_filters=num_filters2,
                             use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(pre_layer=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(pre_layer=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

best_validation_acc: float = 0.0

last_improvement: float = 0.0

require_improvement: int = 1000

saver: Saver = tf.train.Saver()

save_dir = "checkpoints/"

if not path.exists(save_dir):
    makedirs(save_dir)

save_path = path.join(save_dir, "best_validation")

train_batch_size: int = 64

feed_dict_test: dict = {x: data.test.images,
                        y_true: data.test.labels,
                        y_true_cls: np.argmax(data.test.labels, axis=1)}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    total_iterations: int = 0


    def optimize(num_iterations: int):
        # Ensure we update the global variable rather than a local copy.
        global total_iterations
        global best_validation_acc
        global last_improvement

        start_time: float = time.time()

        for i in range(total_iterations,
                       total_iterations + num_iterations):

            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            feed_dict_train: dict = {x: x_batch,
                                     y_true: y_true_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if i % 100 == 0:
                # Calculate the accuracy on the training-set.
                acc = session.run(accuracy, feed_dict=feed_dict_train)

                if acc > best_validation_acc:
                    best_validation_acc = acc
                    last_improvement = total_iterations
                    saver.save(sess=session, save_path=save_path)

                # Message for printing.
                msg: str = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

                print(msg.format(i + 1, acc))

                if total_iterations - last_improvement > require_improvement:
                    print("No improvement found in a while, stopping optimization.")
                    break

        # Update the total number of iterations performed.
        total_iterations += num_iterations

        end_time: float = time.time()

        time_dif: float = end_time - start_time

        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    optimize(1000)

    acc = session.run(accuracy, feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))

    # saver.restore(sess=session, save_path=save_path)
