import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist: Datasets = input_data.read_data_sets("data/MNIST", one_hot=False)

learning_rate: float = 0.1
train_epoch: int = 1000
batch_size: int = 128
display_epoch: int = 100
input_features: int = 28 * 28
class_number: int = 10
dropout: float = 0.75
neurons_layer_1: int = 256
neurons_layer_2: int = 256


def neural_net(x_dict: dict):
    x = x_dict['images']
    layer1 = tf.layers.dense(x, neurons_layer_1)
    layer2 = tf.layers.dense(layer1, neurons_layer_2)
    return tf.layers.dense(layer2, class_number)


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    logits = neural_net(features)

    pred_classes = tf.argmax(logits, axis=1)

    # If in prediction mode , early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)
    ))

    optimizer = tf.train \
        .GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=tf.train.get_global_step())

    acc = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss,
        train_op=optimizer,
        eval_metric_ops={'accuracy': acc}
    )

    return estim_specs


model = tf.estimator.Estimator(model_fn=model_fn, model_dir='./output')

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)

model.train(input_fn=input_fn, steps=train_epoch)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=True
)

e = model.evaluate(input_fn)

print("Test accuracy:", e['accuracy'])
