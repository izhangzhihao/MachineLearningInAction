import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def clean(s: str) -> str:
    if '“/s' not in s:
        return s.replace(' ”/s', '')
    elif '”/s' not in s:
        return s.replace('“/s ', '')
    elif '‘/s' not in s:
        return s.replace(' ’/s', '')
    elif '’/s' not in s:
        return s.replace('‘/s ', '')
    else:
        return s


def read_sentences_from_data() -> list:
    with open('../data/MSRA/msr_train.txt', 'rb') as msr_train:
        texts: str = msr_train.read().decode('utf8')
    return texts.split('\r\n')


def split_sentences() -> list:
    sentences: list = read_sentences_from_data()
    texts: str = ''.join(map(clean, sentences))
    sentences: list = re.split('[，。！？、‘’“”]/[bems]', texts)
    print('Sentences number:', len(sentences))
    print('Sentence Example:\n', sentences[1])
    return sentences


def split_sentence_into_words_and_tags(sentence: str) -> tuple:
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags: list = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags  # 所有的字和tag分别存为 data / label
    return None


def process_words_and_tags_for_all_sentences() -> None:
    global df_data
    words = list()
    tags = list()
    sentences: list = split_sentences()
    for sentence in tqdm(iter(sentences)):
        result: tuple = split_sentence_into_words_and_tags(sentence)
        if result:
            words.append(result[0])
            tags.append(result[1])
    print('Length of words is %d' % len(words))
    print('Datas example: ', words[0])
    print('Labels example:', tags[0])
    df_data = pd.DataFrame({'words': words, 'tags': tags}, index=range(len(words)))
    df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
    # print(df_data.head(2))
    # plot_graph()


def plot_graph() -> None:
    import matplotlib.pyplot as plt
    df_data['sentence_len'].hist(bins=100)
    plt.xlim(0, 100)
    plt.xlabel('sentence_length')
    plt.ylabel('sentence_num')
    plt.title('Distribution of the Length of Sentence')
    plt.show()


def build_vocab() -> None:
    global word2id, id2word, tag2id, id2tag
    from itertools import chain
    process_words_and_tags_for_all_sentences()
    all_words = list(chain(*df_data['words'].values))
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)
    tags = ['x', 's', 'b', 'm', 'e']
    tag_ids = range(len(tags))
    # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    vocab_size = len(set_words)
    print('vocab_size={}'.format(vocab_size))


max_len: int = 32


def sentence_padding(words: list) -> list:
    ids = list(word2id[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def tag_padding(tags: list) -> list:
    ids = list(tag2id[tags])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def padding_words_and_tags() -> None:
    global X, y
    df_data['X'] = df_data['words'].apply(sentence_padding)
    df_data['y'] = df_data['tags'].apply(tag_padding)
    X = np.asarray(list(df_data['X'].values))
    y = np.asarray(list(df_data['y'].values))
    print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
    print('Example of words: ', df_data['words'].values[0])
    print('Example of X: ', X[0])
    print('Example of tags: ', df_data['tags'].values[0])
    print('Example of y: ', y[0])


pkl_file_path = 'data/data.pkl'


def save_data_to_disk() -> None:
    build_vocab()
    padding_words_and_tags()
    import pickle
    import os
    if not os.path.exists('data/'):
        os.makedirs('data/')
    with open(pkl_file_path, 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
    print('** Finished saving the data.')


def read_data_from_disk() -> tuple:
    import pickle
    if os.path.isfile(pkl_file_path):
        print("Read data from pkl file...")
        with open(pkl_file_path, 'rb') as inp:
            X = pickle.load(inp)
            y = pickle.load(inp)
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
        return X, y, word2id, id2word, tag2id, id2tag
    else:
        print("Pkl file does not exist...")
        save_data_to_disk()
        return read_data_from_disk()


def generate_batch_data():
    from BatchGenerator import BatchGenerator

    X, y, word2id, id2word, tag2id, id2tag = read_data_from_disk()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(
        'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
            X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)
    )

    data_train = BatchGenerator(X_train, y_train, shuffle=True)
    data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
    data_test = BatchGenerator(X_test, y_test, shuffle=False)
    return data_train, data_valid, data_test
