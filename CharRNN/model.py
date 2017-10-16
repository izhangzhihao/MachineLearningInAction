import datetime
import os

import numpy as np
import tensorflow as tf


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    return np.random.choice(vocab_size, 1, p=p)[0]


# noinspection PyAttributeOutsideInit
class CharRNN(object):
    def __init__(self, num_classes, num_seqs=64, num_steps=50, lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128,
                 text_converter=None):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.text_converter = text_converter

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps
            ), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps
            ), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_porb')

            # 中文需要使用embedding层
            # 英文不需要
            if not self.use_embedding:
                self.lstm_inputs = tf.one_hot(
                    self.inputs, self.num_classes
                )
            else:
                with tf.device('/cpu:0'):
                    embedding = tf.get_variable(
                        'embedding', [self.num_classes, self.embedding_size]
                    )
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell, self.lstm_inputs, initial_state=self.initial_state
            )
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal(
                    [self.lstm_size, self.num_classes], stddev=0.1
                ))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(logits=self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss_op = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y_reshaped
            )
            self.loss = tf.reduce_mean(loss_op)

    def build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(
            self.loss, tvars
        ), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as session:
            session.run(tf.global_variables_initializer())
            step = 0
            new_state = session.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = datetime.datetime.now()
                feed_dict: dict = {
                    self.inputs: x,
                    self.targets: y,
                    self.keep_prob: self.train_keep_prob,
                    self.initial_state: new_state
                }
                new_state, _ = session.run(
                    [self.final_state, self.optimizer],
                    feed_dict=feed_dict
                )
                end = datetime.datetime.now()
                if step % log_every_n == 0:
                    batch_loss, new_state, _ = session.run(
                        [self.loss, self.final_state, self.optimizer],
                        feed_dict=feed_dict
                    )
                    print(
                        'Step: {0}\tloss: {1} \tsec/batch: {2}'.format(str(step), str(batch_loss),
                                                                       str((end - start).total_seconds())))
                if step % save_every_n == 0:
                    self.save_checkpoint(save_path, session, step)
                if step >= max_steps:
                    break
            self.save_checkpoint(save_path, session, step)
            self.save_meta_data(save_path)

    def save_checkpoint(self, save_path, session, step):
        self.saver.save(session, os.path.join(save_path, 'model'), global_step=step)
        print('Checkpoint at step: {0} saved.'.format(str(step)))

    def save_meta_data(self, save_path):
        if self.text_converter is not None and self.use_embedding:
            with open(save_path + '/metadata.tsv', 'w') as f:
                f.write("Index\tWord\n")
                for index in range(self.text_converter.vocab_size):
                    f.write("%d\t%s\n" % (index, self.text_converter.int_to_word(index)))

    # noinspection PyUnusedLocal
    def sample(self, n_samples, start, vocab_size):
        samples = []
        session = self.session
        new_state = session.run(self.initial_state)
        c = 0

        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed_dict: dict = {self.inputs: x,
                               self.keep_prob: 1.0,
                               self.initial_state: new_state}
            preds, new_state = session.run(
                [self.proba_prediction, self.final_state],
                feed_dict=feed_dict
            )

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restore from {0}'.format(str(checkpoint)))
