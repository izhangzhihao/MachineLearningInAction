import tensorflow as tf
from tensorflow.contrib import rnn


class BiLSTMSegmentation(object):

    def __init__(self, decay: float = 0.85, decay_epoch: int = 5, max_epoch: int = 10, timestep_size: int = 32,
                 max_len: int = 32, vocab_size: int = 5159,
                 input_size: int = 64, embedding_size: int = 64, class_num: int = 5, hidden_size: int = 128,
                 layer_num: int = 2, max_grad_norm: float = 5.0,
                 model_save_path: str = 'data/bi-lstm-segmentation.ckpt'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.decay: float = decay
        self.decay_epoch: int = decay_epoch
        self.max_epoch: int = max_epoch
        self.timestep_size: int = timestep_size
        self.max_len: int = max_len
        self.vocab_size: int = vocab_size
        self.input_size: int = input_size
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num
        self.hidden_size: int = hidden_size
        self.layer_num: int = layer_num
        self.max_grad_norm: float = max_grad_norm
        self.model_save_path: str = model_save_path

    def __inputs__(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size])
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size])
            self.lr = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [])
            self.keep_prob = tf.placeholder(tf.float32, [])

    def __embedding__(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)

    @staticmethod
    def __weight_variable__(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def __bias_variable__(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def __lstm_cell__(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def __bi_lstm__(self, X_inputs):
        """build the bi-LSTMs network. Return the y_pred"""
        self.__embedding__()
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.unstack(inputs, self.timestep_size, 1)

        # lstm_fw_cell = rnn.MultiRNNCell([self.__lstm_cell__()] * self.layer_num)
        # lstm_bw_cell = rnn.MultiRNNCell([self.__lstm_cell__()] * self.layer_num)

        lstm_fw_cell = self.__lstm_cell__()
        lstm_bw_cell = self.__lstm_cell__()

        initial_state_fw = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
        initial_state_bw = lstm_bw_cell.zero_state(self.batch_size, tf.float32)
        outputs, _, _ = rnn.static_bidirectional_rnn(
            lstm_fw_cell, lstm_bw_cell, inputs, initial_state_fw, initial_state_bw)
        return tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size * 2])

    def __outputs__(self):
        self.__inputs__()
        bilstm_output = self.__bi_lstm__(self.X_inputs)

        with tf.variable_scope('outputs'):
            softmax_w = self.__weight_variable__([self.hidden_size * 2, self.class_num])
            softmax_b = self.__bias_variable__([self.class_num])
            self.y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

    def build_model(self):
        # adding extra statistics to monitor
        # y_inputs.shape = [batch_size, timestep_size]
        self.__outputs__()
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        accuracy: float = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.y_inputs, [-1]), logits=self.y_pred))

        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        train_op = optimizer.apply_gradients(zip(grads, tvars),
                                             global_step=tf.contrib.framework.get_or_create_global_step())
        print('Finished build the bi-lstm model.')
        return accuracy, cost, train_op
