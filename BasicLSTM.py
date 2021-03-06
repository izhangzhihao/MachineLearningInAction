import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn_cell

mnist: DataSet = input_data.read_data_sets('data/MNIST', one_hot=True)

learning_rate: float = 0.001
training_steps: int = 10000
batch_size: int = 128
display_step: int = 200

num_input: int = 28
timesteps: int = 28
num_hidden: int = 128
num_classes: int = 10

X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

weights: dict = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases: dict = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights: dict, biases: dict):
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    # 初始化全零 state
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # 如果 inputs 为 (batches, steps, inputs) ==> time_major=False
    # 如果 inputs 为 (inputssteps, batches, inputs) ==> time_major=True
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=init_state, time_major=False)
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(1, training_steps + 1):
        # 128 * 784
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 128 * 28 * 28
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        session.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={
                X: batch_x, Y: batch_y
            })
            print("Step " + str(step) + "\t loss" + str(loss) + "\t acc" + str(acc))

    test_data = mnist.test.images[:128].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:128]
    print("Test acc " + str(
        session.run(
            accuracy, feed_dict={X: test_data, Y: test_label}
        )
    ))
