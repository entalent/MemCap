import os
from collections import Counter

import torch
from keras_preprocessing.text import tokenizer_from_json

import sys
sys.path.append('.')

from sklearn import metrics
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np

import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import backend
# from tensorflow.keras.layers import Dropout, Dense, GRU, Embedding, Input, Conv1D, MaxPooling1D, Concatenate, Flatten
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras import backend
from keras.layers import Dropout, Dense, GRU, Embedding, Input, Conv1D, MaxPooling1D, Concatenate, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from util import *
from config import *

num_cores = 3

def get_session():
    _config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                             inter_op_parallelism_threads=num_cores,
                             device_count={'CPU': 1, 'GPU': 0})
    _session = tf.Session(config=_config)
    return _session

backend.set_session(get_session())

def load_sents(dataset_name):
    d = load_custom(os.path.join(annotation_path, 'dataset_{}.json'.format(dataset_name)))['caption_item']
    sents = []
    for item in d:
        for i in item.sentences:
            i.split = item.split
        sents.extend(item.sentences)
    return sents

def get_data(sents, style_tag, balance=True):
    all_sents = sents

    X = [' '.join(sent.words) for sent in all_sents]
    Y = [1 if hasattr(sent, 'tag') and sent.tag == style_tag else 0 for sent in all_sents]
    splits = [sent.split for sent in all_sents]

    train_pos, train_neg = [], []
    test_pos, test_neg = [], []

    for i, split in enumerate(splits):
        # if split == 'test':
        if random.random() < 0.15:
            if Y[i] == 1:
                test_pos.append((X[i], Y[i]))
            else:
                test_neg.append((X[i], Y[i]))
        else:
            if Y[i] == 1:
                train_pos.append((X[i], Y[i]))
            else:
                train_neg.append((X[i], Y[i]))

    if balance:
        if len(train_pos) > len(train_neg):
            train_pos = random.sample(train_pos, k=len(train_neg))
        if len(train_neg) > len(train_pos):
            train_neg = random.sample(train_neg, k=len(train_pos))
        if len(test_neg) > len(test_pos):
            test_neg = random.sample(test_neg, k=len(test_pos))
        if len(test_pos) > len(test_neg):
            test_pos = random.sample(test_pos, k=len(test_neg))

    X_train, Y_train = [[i[j] for i in train_pos + train_neg] for j in (0, 1)]
    X_test, Y_test = [[i[j] for i in test_pos + test_neg] for j in (0, 1)]

    print('train_pos:', len(train_pos), 'train_neg:', len(train_neg))
    print('test_pos:', len(test_pos), 'test_neg:', len(test_neg))

    return X_train, Y_train, X_test, Y_test

# TODO: max length -> 25 (or other values)
def loadData_Tokenizer(X_train, X_test,MAX_NB_WORDS=75000, MAX_SEQUENCE_LENGTH=500):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):, ]
    embeddings_index = {}
    f = open("/media/wentian/sdb2/work/caption_dataset/Chinese Word Vectors/sgns.wiki.word", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        if word not in word_index:
            continue
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))

    return (X_train, X_test, word_index, embeddings_index, tokenizer)

def Build_Model_RNN_Text(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
    """
    def buildModel_RNN(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
    word_index in word index ,
    embeddings_index is embeddings index, look at data_helper.py
    nClasses is number of classes,
    MAX_SEQUENCE_LENGTH is maximum lenght of text sequences
    """
    model = Sequential()
    hidden_layer = 3
    gru_node = 32

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    for i in range(0,hidden_layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))


    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def Build_Model_CNN_Text(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):

    """
        def buildModel_CNN(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
        word_index in word index ,
        embeddings_index is embeddings index, look at data_helper.py
        nClasses is number of classes,
        MAX_SEQUENCE_LENGTH is maximum lenght of text sequences,
        EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py
    """

    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = []
    layer = 5
    print("Filter  ",layer)
    for fl in range(0,layer):
        filter_sizes.append((fl+2))

    node = 128
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        #l_pool = Dropout(0.25)(l_pool)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
    l_cov1 = Dropout(dropout)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
    l_cov2 = Dropout(dropout)(l_cov2)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(1024, activation='relu')(l_flat)
    l_dense = Dropout(dropout)(l_dense)
    l_dense = Dense(512, activation='relu')(l_dense)
    l_dense = Dropout(dropout)(l_dense)
    preds = Dense(nclasses, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



    return model

def train(style_dataset, target_style, all_datasets=('coco', 'senticap', 'flickrstyle')):
    MAX_SEQUENCE_LENGTH = 25
    nclasses = 2

    all_sent = []
    for d in all_datasets:
        all_sent.extend(load_sents(d))

    c = Counter()
    c.update([i.tag if hasattr(i, 'tag') else None for i in all_sent])

    X_train, Y_train, X_test, Y_test = get_data(all_sent, target_style)
    print('get_data done')
    X_train_Glove, X_test_Glove, word_index, embeddings_index, tokenizer = loadData_Tokenizer(X_train, X_test,
                                                                                              MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    print('loadData_Tokenizer done')

    model_RNN = Build_Model_RNN_Text(word_index, embeddings_index, nclasses=nclasses, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                     EMBEDDING_DIM=300)
    _ = time.time()
    model_RNN.fit(X_train_Glove, Y_train,
                  validation_data=(X_test_Glove, Y_test),
                  epochs=15,
                  batch_size=128,
                  verbose=2)
    print('fit used {}'.format(time.time() - _))
    predicted = model_RNN.predict_classes(X_test_Glove)
    print(metrics.classification_report(Y_test, predicted))

    model_save_dir = '../data/clf_nn'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    with open(os.path.join(model_save_dir, 'model_rnn_info_{}_{}.pkl'.format(style_dataset, target_style)), 'wb') as f:
        pickle.dump(obj={'word_index': word_index, 'embeddings_index': embeddings_index,
                         'nclasses': nclasses, 'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
                         'EMBEDDING_DIM': 300,
                         'tokenizer_config': tokenizer.to_json(),
                         '_classification_report': metrics.classification_report(Y_test, predicted)},
                    file=f)
    model_RNN.save_weights(filepath=os.path.join(model_save_dir, 'model_rnn_{}_{}.h5'.format(style_dataset, target_style)))

# def train_multi():
#     MAX_SEQUENCE_LENGTH = 25
#     nclasses = 5
#
#     sents_coco = load_sents('coco')
#     sents_senticap = load_sents('senticap')
#     sents_flickrstyle = load_sents('flickrstyle')
#
#     all_sent = sents_coco + sents_senticap + sents_flickrstyle
#
#     c = Counter()
#     c.update([i.tag if hasattr(i, 'tag') else None for i in all_sent])
#
#     X_train, Y_train, X_test, Y_test = get_data(all_sent, target_style)
#     print('get_data done')
#     X_train_Glove, X_test_Glove, word_index, embeddings_index, tokenizer = loadData_Tokenizer(X_train, X_test,
#                                                                                               MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
#     print('loadData_Tokenizer done')
#
#     model_RNN = Build_Model_RNN_Text(word_index, embeddings_index, nclasses=nclasses, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
#     _ = time.time()
#     model_RNN.fit(X_train_Glove, Y_train,
#                   validation_data=(X_test_Glove, Y_test),
#                   epochs=15,
#                   batch_size=128,
#                   verbose=2)
#     print('fit used {}'.format(time.time() - _))
#     predicted = model_RNN.predict_classes(X_test_Glove)
#     print(metrics.classification_report(Y_test, predicted))
#
#     model_save_dir = '../data/clf_nn'
#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir)
#     with open(os.path.join(model_save_dir, 'model_rnn_info_{}_{}.pkl'.format(style_dataset, target_style)), 'wb') as f:
#         pickle.dump(obj={'word_index': word_index, 'embeddings_index': embeddings_index,
#                          'nclasses': nclasses, 'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
#                          'tokenizer_config': tokenizer.to_json(),
#                          '_classification_report': metrics.classification_report(Y_test, predicted)},
#                     file=f)
#     model_RNN.save_weights(filepath=os.path.join(model_save_dir, 'model_rnn_{}_{}.h5'.format(style_dataset, target_style)))

def test_tokenize(tokenizer, sents, MAX_SEQUENCE_LENGTH=500):
    sequences = tokenizer.texts_to_sequences(sents)
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return text

def test(dataset, style, test_file):
    obj = json.load(open(test_file, 'r'))
    # obj = json.load(open('/media/wentian/sdb2/work/caption_ma/save/2019-10-04_21-09-37_2agent_neg/annotation.json', 'r'))['annotations']
    sents = [i['caption'] for i in obj]

    d = pickle.load(open(r'../data/clf_nn/model_rnn_info_{}_{}.pkl'.format(dataset, style), 'rb'))
    w, e, tokenizer_config = d['word_index'], d['embeddings_index'], d['tokenizer_config']
    MAX_SEQUENCE_LENGTH, nclasses, EMBEDDING_DIM = d['MAX_SEQUENCE_LENGTH'], d['nclasses'], d['EMBEDDING_DIM']
    tokenizer = tokenizer_from_json(tokenizer_config)
    model_RNN = Build_Model_RNN_Text(w, e, nclasses=nclasses, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, EMBEDDING_DIM=EMBEDDING_DIM)
    model_RNN.load_weights('../data/clf_nn/model_rnn_{}_{}.h5'.format(dataset, style))

    X_train_Glove = test_tokenize(tokenizer, sents, MAX_SEQUENCE_LENGTH)
    predicted = model_RNN.predict_classes(X_train_Glove, verbose=0)

    for i in range(len(predicted)):
        if predicted[i] == 0:
            print(sents[i], predicted[i])
    print(sum(predicted) / len(predicted))


if __name__ == '__main__':
    # train('senticap', 'positive')
    # train('senticap', 'negative')
    # train('flickrstyle', 'humor')
    # train('flickrstyle', 'romantic')

    # train('chn_styled_word', 'positive', all_datasets=['youku_chn_word', 'chn_styled_word'])
    # train('chn_styled_word', 'negative', all_datasets=['youku_chn_word', 'chn_styled_word'])
    # train('chn_styled_word', 'humor', all_datasets=['youku_chn_word', 'chn_styled_word'])
    # train('chn_styled_word', 'romantic', all_datasets=['youku_chn_word', 'chn_styled_word'])

    # test('youku_chn_word', 'positive', '/media/wentian/sdb2/work/styled_caption/save/2019-08-22_21-17-19_chn_positive_word/result_youku_chn_word_positive_10.json')
    # test('youku_chn_word', 'negative', '/media/wentian/sdb2/work/styled_caption/save/2019-08-22_23-50-59_chn_negative_word/result_youku_chn_word_negative_20.json')
    # test('youku_chn_word', 'humor', '/media/wentian/sdb2/work/styled_caption/save/2019-08-22_21-08-48_chn_humor_word/result_youku_chn_word_humor_7.json')
    test('youku_chn_word', 'romantic', '/media/wentian/sdb2/work/styled_caption/save/2019-08-22_21-13-01_chn_romantic_word/result_youku_chn_word_romantic_8.json')