#%%
import os
import sys
sys.path.append('.')
import random
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegressionCV
import json
import numpy as np
import sklearn.metrics as metrics
import pickle

import util
from util import *
from config import *

class FeatureExtractor(object):
    def __init__(self, count=True, hashing=False):
        self.count = count
        self.hashing = hashing
        self.was_fit = False

    def fit(self, X_text):
        if self.count:
            self.cv = CountVectorizer()
            self.cv.fit(X_text)
        if self.hashing:
            self.hv = HashingVectorizer(ngram_range=(1,2), norm=None, alternate_sign=False, binary=True)
            self.hv.fit(X_text)

        self.was_fit = True
        return

    def transform(self, X_text):
        assert self.was_fit
        if self.count:
            X_count = self.cv.transform(X_text)
            X = X_count
        if self.hashing:
            X_hashing = self.hv.transform(X_text)
            X = X_hashing
        if self.hashing and self.count:
            X = np.hstack([X_count, X_hashing])
        return X


def load_sents(dataset_path, style_tag=None, split='train'):
    dataset = load_custom(dataset_path)
    dataset = dataset['caption_item']

    sents = []
    for item in dataset:
        if split == 'all' or split == item.split:
            for sent in item.sentences:
                if style_tag is None or sent.tag == style_tag:
                    sents.append(sent)
    return sents


def get_samples(sents, style_tag):
    X, Y = [], []
    for sent in sents:
        X.append(sent.raw)
        Y.append(sent.tag == style_tag)

    return X, Y


def get_train_test(X_text, Y):
    np.random.seed(123)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_text, Y, np.arange(len(X_text)), train_size=0.9, shuffle=True, stratify=Y)
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def train_classifier(factual_dataset, styled_dataset, style_tag, output_file):
    #%%
    factual_sents = load_sents(factual_dataset)
    styled_sents = load_sents(styled_dataset)

    styled_sents = list(filter(lambda x: x.tag == style_tag, styled_sents))
    factual_sents = random.sample(factual_sents, k=len(styled_sents))

    all_tags = Counter(sent.tag for sent in factual_sents + styled_sents)
    print('valid tags:', all_tags)
    print('using tag:', style_tag)
    assert style_tag in all_tags

    X_text, y = get_samples(factual_sents + styled_sents, style_tag)

    print('--------')
    print('total labels: {}, 1: {}'.format(len(y), (np.array(y) == 1).sum()))

    Xt_train, Xt_test, y_train, y_test, idx_train, idx_test = get_train_test(X_text, y)
    #%%
    ext = FeatureExtractor(count=False, hashing=True)
    ext.fit(Xt_train)
    #%%
    X_train = ext.transform(Xt_train)
    X_test = ext.transform(Xt_test)
    #%%

    def run_clf(clf, X_test):
        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)[:, 1]
        return y_test_pred, y_test_prob

    def eval_clf(y_test, y_test_pred, y_test_prob):
        acc = metrics.accuracy_score(y_test, y_test_pred)
        auc = metrics.roc_auc_score(y_test, y_test_prob)
        prec = metrics.precision_score(y_test, y_test_pred)
        rec = metrics.recall_score(y_test, y_test_pred)
        print("Accuracy:", acc)
        print("AUC:", auc)
        print("Precision:", prec)
        print("Recall:", rec)
        return {'accuracy': acc, 'auc': auc, 'precision': prec, 'recall': rec}

    #%%
    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    y_pred, y_prob = run_clf(nb, X_test)
    print('BernoulliNB, {} {} {}'.format(factual_dataset, styled_dataset, style_tag))
    m = eval_clf(y_test, y_pred, y_prob)

    pickle.dump({'fe': ext, 'nb': nb, 'metrics': m}, open(output_file, 'wb'), protocol=2)

    _ = time.time()
    #%%
    lr = LogisticRegressionCV(max_iter=300)
    lr.fit(X_train, y_train)
    y_pred, y_prob = run_clf(lr, X_test)
    print('LogisticRegressionCV, {} {} {}'.format(factual_dataset, styled_dataset, style_tag))
    m = eval_clf(y_test, y_pred, y_prob)
    print('used {}'.format(time.time() - _))

    #%%
    pickle.dump({"fe":ext, "clf": lr, 'metrics': m}, open(output_file, "wb"), protocol=2)

    # #%%
    # import  scipy.sparse as sparse
    # print(X_train.shape)
    # print(X_test.shape)
    # #print sparse.vstack([X_train, X_test])
    # #%%
    #
    #
    # text_mymethod = read_mymethod()
    # X_mymethod = ext.transform(text_mymethod)
    # text_showtell = read_showtell()
    # X_showtell = ext.transform(text_showtell)
    # #%%
    # print("My errors:", np.sum(lr.predict(X_mymethod)))
    # print("ShowTell errors:", np.sum(lr.predict(X_showtell)))
    # #%%
    # out_probas = lr.predict_proba(X_mymethod)[:, 1]
    # order = np.argsort(-out_probas)
    # for sent in text_mymethod[order[:10]]:
    #     print (sent)
    # #%%
    # out_probas = lr.predict_proba(X_showtell)[:, 1]
    # order = np.argsort(-out_probas)
    # for sent in text_showtell[order[:20]]:
    #     print (sent)
    # #%%


def load_clf(clf_path):
    return pickle.load(clf_path)


def eval_sentences(clf, sents):
    if 'fe' in clf and 'clf' in clf:
        ext, lr = clf['fe'], clf['clf']
        X = ext.transform(sents)
        y_test_pred = lr.predict(X)
        return float(np.sum(y_test_pred)) / len(sents), y_test_pred
    elif 'fe' in clf and 'nb' in clf:
        ext, nb = clf['fe'], clf['nb']
        X = ext.transform(sents)
        y_test_pred = nb.predict(X)
        return float(np.sum(y_test_pred)) / len(sents), y_test_pred


def load_sents(dataset_name):
    d = load_custom(os.path.join(annotation_path, 'dataset_{}.json'.format(dataset_name)))['caption_item']
    sents = []
    for item in d:
        sents.extend(item.sentences)
    return sents


if __name__ == '__main__':
    # train_classifier('coco', 'flickrstyle', 'humor', os.path.join(data_path, 'clf', 'clf_lr_flickrstyle_humor.pik'))
    # train_classifier('coco', 'flickrstyle', 'romantic', os.path.join(data_path, 'clf', 'clf_lr_flickrstyle_romantic.pik'))
    # train_classifier('coco', 'senticap', 'positive', os.path.join(data_path, 'clf', 'clf_lr_senticap_positive.pik'))
    # train_classifier('coco', 'senticap', 'negative', os.path.join(data_path, 'clf', 'clf_lr_senticap_negative.pik'))

    for mode in ['word']:
        for style in ['positive', 'negative', 'humor', 'romantic']:
            dataset_name = 'youku_chn_' + mode
            styled_dataset_name = 'chn_styled_' + mode
            train_classifier(dataset_name, styled_dataset_name, style, os.path.join(data_path, 'clf',
                                                                                    'clf_lr_{}_{}.pik'.format(styled_dataset_name, style)))

    clf = load_clf(open('/media/wentian/sdb2/work/styled_caption/data/clf/clf_lr_chn_styled_char_positive.pik', 'rb'))

    d = json.load(open('/media/wentian/sdb2/work/styled_caption/save/2019-08-22_21-17-19_chn_positive_word/result_youku_chn_word_positive_10.json', 'r'))
    # sents = [' '.join(''.join(i['caption'].split())) for i in d]
    sents = [' '.join(''.join(i['meta']['factual'].split())) for i in d]

    score, _ = eval_sentences(clf, sents)
    print(score)
