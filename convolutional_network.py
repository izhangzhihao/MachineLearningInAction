import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist: Datasets = input_data.read_data_sets("data/MNIST", one_hot=False)

learning_rate: float = 0.001
num_steps: int = 2000
batch_size: int = 128
display_epoch: int = 10
image_size: int = 28
input_features: int = image_size * image_size
class_number: int = 10
dropout: float = 0.75


def conv_net(x_dict, n_classes, dropout: float, reuse: bool, is_training: bool):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out


def model_fn(features, labels, mode):
    logits_train = conv_net(features, class_number, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, class_number, dropout, reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)
    ))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op,
                                                                            global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )


model = tf.estimator.Estimator(model_fn=model_fn, model_dir='./output')

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)

model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=True
)

e = model.evaluate(input_fn)

print("Test acc:", e['accuracy'])
