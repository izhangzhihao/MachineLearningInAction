import re
from model import BiLSTMSegmentation
import numpy as np
import pandas as pd
import tensorflow as tf

from prepare_data import contact_words_tags, read_data_from_disk

lstm_segmentation = BiLSTMSegmentation()
_ = lstm_segmentation.build_model()
session = lstm_segmentation.session
best_model_path = 'model/bi-lstm.ckpt-2'
tf.train.Saver().restore(session, best_model_path)


def transition_probability():
    # 利用 tags（即状态序列）来统计转移概率
    # 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
    # A统计状态转移的频数
    A = {
        'sb': 0,
        'ss': 0,
        'be': 0,
        'bm': 0,
        'me': 0,
        'mm': 0,
        'eb': 0,
        'es': 0
    }
    words, tags = contact_words_tags()
    # _zy 表示转移概率矩阵
    _zy = dict()
    for tag in tags:
        for t in range(len(tag) - 1):
            key = tag[t] + tag[t + 1]
            A[key] += 1.0
    _zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
    _zy['ss'] = 1.0 - _zy['sb']
    _zy['be'] = A['be'] / (A['be'] + A['bm'])
    _zy['bm'] = 1.0 - _zy['be']
    _zy['me'] = A['me'] / (A['me'] + A['mm'])
    _zy['mm'] = 1.0 - _zy['me']
    _zy['eb'] = A['eb'] / (A['eb'] + A['es'])
    _zy['es'] = 1.0 - _zy['eb']
    keys = sorted(_zy.keys())
    print('the transition probability: ')
    for key in keys:
        print(key, _zy[key])
    return {i: np.log(_zy[i]) for i in _zy.keys()}


zy = transition_probability()

X, y, word2id, id2word, tag2id, id2tag = read_data_from_disk()


def viterbi(nodes):
    """
    维特比译码：除了第一层以外，每一层有4个节点。
    计算当前层（第一层不需要计算）四个节点的最短路径：
       对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
       上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
       paths 采用字典的形式保存（路径：路径长度）。
       一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
    """
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[
                        path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]  # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


max_len = 32


def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        print('输出片段超过%d部分无法处理' % max_len)
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    return ids


def simple_cut(text):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        text_len = len(text)
        X_batch = text2ids(text)  # 这里每个 batch 是一个样本
        fetches = [lstm_segmentation.y_pred]
        feed_dict = {lstm_segmentation.X_inputs: X_batch, lstm_segmentation.lr: 1.0, lstm_segmentation.batch_size: 1,
                     lstm_segmentation.keep_prob: 1.0}
        _y_pred = session.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        nodes = [dict(zip(['s', 'b', 'm', 'e'], each[1:])) for each in _y_pred]
        tags = viterbi(nodes)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'b']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []


def cut_word(sentence):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:]))
    return result


sentence = '人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，\
      而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print(rss)
