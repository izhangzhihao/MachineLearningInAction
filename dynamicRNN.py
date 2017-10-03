import tensorflow as tf
import random


class GenerateDynamicLengthSeqData:

    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(len)
            if random.random() < 0.5:
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1.0, 0.0])
            else:
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                s += [[0.0] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0.0, 1.0])
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


learning_rate: float = 0.01
training_steps: int = 1000
batch_size: int = 128
display_steps: int = 200

seq_max_len: int = 20
hidden_features: int = 64
classes_num: int = 2

trainset = GenerateDynamicLengthSeqData(n_samples=1000, max_seq_len=seq_max_len)
testset = GenerateDynamicLengthSeqData(n_samples=500, max_seq_len=seq_max_len)

X = tf.placeholder(tf.float32, [None, seq_max_len, 1])
Y = tf.placeholder(tf.float32, [None, classes_num])

seqlen = tf.placeholder(tf.int32, [None])

weights: dict = {
    'out': tf.Variable(tf.random_normal([hidden_features, classes_num]))
}

biases: dict = {
    'out': tf.Variable(tf.random_normal([classes_num]))
}


def dynamicRNN(x, seqlen, weights, biases):
    x = tf.unstack(x, seq_max_len, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_features)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, hidden_features]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']


logits = dynamicRNN(X, seqlen, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        session.run(optimizer, feed_dict={X: batch_x, Y: batch_y, seqlen: batch_seqlen})

        if step % display_steps == 0 or step == 1:
            acc, loss = session.run([accuracy, cost], feed_dict={X: batch_x, Y: batch_y, seqlen: batch_seqlen})
            print("Accuracy : " + str(acc) + "\t loss:" + str(loss))
    print("Test accuracy : " + str(session.run(
        accuracy, feed_dict={X: testset.data, Y: testset.labels, seqlen: testset.seqlen}
    )))
