import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

data_sets: Dataset = input_data.read_data_sets('data/MNIST', one_hot=True)

learning_rate: float = 0.001
training_steps: int = 1000
batch_size: int = 128
display_step: int = 100
num_input: int = 28
timesteps: int = 28
num_hidden: int = 128
num_classes: int = 10
drop_rate: float = 0.5
num_layers: int = 2

X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

weights: dict = {
    'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
}

biases: dict = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def BiLSTM(x, weights: dict, biases: dict):
    # inputs: A length T list of inputs, each a tensor of shape [batch_size, input_size]
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    inputs = tf.unstack(x, timesteps, 1)

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=(1 - drop_rate))
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=(1 - drop_rate))

    # lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * num_layers)
    # lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * num_layers)

    initial_state_fw = lstm_fw_cell.zero_state(batch_size, tf.float32)
    initial_state_bw = lstm_bw_cell.zero_state(batch_size, tf.float32)

    outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(
        lstm_fw_cell, lstm_bw_cell, inputs, initial_state_fw, initial_state_bw
    )

    # outputs = output_state_fw concat output_state_bw
    # shape of outputs : [steps,batch_size,num_hidden*2]

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = BiLSTM(X, weights, biases)
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
        batch_x, batch_y = data_sets.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        session.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step: " + str(step) + "\tloss: " + str(loss) + "\tacc: " + str(acc))

    test_data = data_sets.test.images[:128].reshape((-1, timesteps, num_input))
    test_label = data_sets.test.labels[:128]
    print("Test accuracy : " + str(session.run(accuracy, feed_dict={X: test_data, Y: test_label})))
