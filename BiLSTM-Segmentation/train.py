from model import BiLSTMSegmentation
from prepare_data import generate_batch_data
import tensorflow as tf
from tqdm import tqdm
import time

model_save_path = "model/bi-lstm.ckpt"


def test_acc(dataset, accuracy, lstm_segmentation):
    """Testing or valid."""
    _batch_size = 128
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    for i in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {lstm_segmentation.X_inputs: X_batch,
                     lstm_segmentation.y_inputs: y_batch,
                     lstm_segmentation.lr: 1e-3}
        _acc = lstm_segmentation.session.run(accuracy, feed_dict)
        _accs += _acc
    mean_acc = _accs / batch_num
    return mean_acc


def train():
    decay_epoch: int = 3
    max_epoch: int = 2
    decay: float = 0.9
    batch_size: int = 128
    lstm_segmentation = BiLSTMSegmentation(decay=decay, decay_epoch=decay_epoch, max_epoch=max_epoch,
                                           batch_size=batch_size)
    accuracy, cost, train_op = lstm_segmentation.build_model()
    lstm_segmentation.session.run(tf.global_variables_initializer())
    data_train, data_valid, data_test = generate_batch_data()
    display_num: int = 5
    tr_batch_num: int = int(data_train.y.shape[0] / batch_size)
    display_batch: int = int(tr_batch_num / display_num)
    saver = tf.train.Saver(max_to_keep=10)

    start_time = time.time()

    for epoch in range(max_epoch):
        _lr = 1e-3
        if epoch > decay_epoch:
            _lr = _lr * (decay ** (epoch - decay_epoch))
        print('EPOCH %d， lr=%g' % (epoch + 1, _lr))

        for batch in tqdm(range(tr_batch_num)):
            X_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {lstm_segmentation.X_inputs: X_batch,
                         lstm_segmentation.y_inputs: y_batch,
                         lstm_segmentation.lr: _lr}
            _ = lstm_segmentation.session.run(train_op, feed_dict)
            if (batch + 1) % display_batch == 0:
                _acc, _cost, _ = lstm_segmentation.session.run([accuracy, cost, train_op], feed_dict)
                print('\ttraining acc={:g}, cost={:.8g}'.format(_acc, _cost))

        if (epoch + 1) % 2 == 0:
            save_path = saver.save(lstm_segmentation.session, model_save_path, global_step=(epoch + 1))
            print('model saved in ', save_path)
        print('Epoch %d training finished, ' % epoch)
    print('Train finished, use %g' % (time.time() - start_time))
    test_acca = test_acc(data_test, accuracy, lstm_segmentation)
    print('test set acc= %g ' % test_acca)


train()
