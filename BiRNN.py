import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist: Dataset = input_data.read_data_sets('data/MNIST', one_hot=True)

learning_rate: float = 0.001
training_steps: int = 10000
batch_size: int = 128
display_step: int = 200
input_features: int = 28
timesteps: int = 28
hidden_features: int = 128
classes_num: int = 10

X = tf.placeholder(tf.float32, [None, timesteps, input_features])
Y = tf.placeholder(tf.float32, [None, classes_num])

weights: dict = {
    'out': tf.Variable(tf.random_normal([2 * hidden_features, classes_num]))
}

biases: dict = {
    'out': tf.Variable(tf.random_normal([classes_num]))
}


# noinspection PyUnresolvedReferences
def BiRNN(x, weights: dict, biases: dict):
    x = tf.unstack(x, timesteps, 1)

    # forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(hidden_features, forget_bias=1.0)

    # backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(hidden_features, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, input_features))
        session.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step: " + str(step) + "\tloss: " + str(loss) + "\tacc: " + str(acc))

    test_data = mnist.test.images[:128].reshape((-1, timesteps, input_features))
    test_label = mnist.test.labels[:128]
    print("Test accuracy : " + str(session.run(accuracy, feed_dict={X: test_data, Y: test_label})))
