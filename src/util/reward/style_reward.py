import os
import sys
from collections import OrderedDict
from multiprocessing.pool import Pool

import util
import util.reward
from util import *
from config import *
from styled_eval import lm_srilm, clf
from .ciderD.ciderD import *

from styled_eval import NNCLFEvaluator

# clf_path = None
# lm = None
_init_list = []
pool = None

all_nn_clf = {}
all_clf = {}
all_lm = {}
all_styles = set()

bleu4_scorer = None
all_cider_scorer = {}

@arg_type(tokens=[list, np.ndarray], vocab=Vocabulary)
def _to_str(tokens, vocab, keep_end_token=True):
    trimmed_tokens = trim_generated_tokens(tokens)
    s = ' '.join(vocab.get_word(i) for i in trimmed_tokens)
    if keep_end_token:
        s += ' ' + Vocabulary.end_token
    return s


def init_style_scorer(styles):
    _init_list = styles

    global all_clf
    global all_lm
    global all_styles

    style_tag_to_name = {
        'humor': 3,
        'romantic' : 4,
        'positive' : 5,
        'negative': 6,
    }
    style_to_dataset_name = {
        'factual': 'coco',
        'positive': 'senticap',
        'negative': 'senticap',
        'humor': 'flickrstyle',
        'romantic': 'flickrstyle',
    }

    # FIXME: for Chinese
    # style_to_dataset_name = {
    #     'factual': 'youku_chn',
    #     'positive': 'chn_styled_word',
    #     'negative': 'chn_styled_word',
    #     'humor': 'chn_styled_word',
    #     'romantic': 'chn_styled_word',
    # }

    for style in styles:
        dataset_name = style_to_dataset_name[style]
        df = '../data/preprocessed/ngram_{}_train_words.p'.format(dataset_name)
        # util.reward.init_scorer(df=df)
        all_cider_scorer[dataset_name] = CiderD(df=df)

        if style == 'factual':
            continue

        s = style_tag_to_name[style]

        clf_path = os.path.join(data_path, 'clf', 'clf_lr_{}_{}.pik'.format(dataset_name, style))
        with open(clf_path, 'rb') as f:
            all_clf[s] = pickle.load(f)

        # FIXME: for chinese only!!!
        # all_nn_clf[s] = NNCLFEvaluator(dataset_name, style)

        lm_path = os.path.join(data_path, 'srilm', '{}_{}.srilm'.format(dataset_name, style))
        all_lm[s] = lm_path
        all_styles.add(s)

def lm_reward(ppl):
    return -1 * ppl

def _compute_cider(dataset, gts, res):
    """

    :param dataset: list
    :param gts:
    :param res:
    :return:
    """
    global all_cider_scorer
    valid_ds = set(dataset.values())
    res_ = [{'image_id': i, 'caption': [res[i]]} for i in res.keys()]

    dataset = np.array([dataset[i] for i in res.keys()])

    score = np.zeros([len(res_),])
    for d in valid_ds:
        _, score_cider = all_cider_scorer[d].compute_score(gts, res_)
        score += (d == dataset) * score_cider
    return score

def _compute_bleu(_, gts, res):
    res__ = {i: [res[i]] for i in res.keys()}
    _, score_bleu = bleu4_scorer.compute_score(gts, res__)
    score_bleu4 = np.array(score_bleu[3])
    return score_bleu4

def get_scores(dataset, res, gts, weights, n_threads=4):
    """
        :param dataset: ['flickrstyle', 'flickrstyle']
        :param res: ['candidate1', 'candidate2']
        :param gts: {0: ['sent1', 'sent2'], 1: ['sent3', 'sent4']}
        :param weights: {'cider': 0.5, 'bleu': 0.5}
        :return:
        """

    score = 0.

    if n_threads <= 0:      # single thread
        _dataset = dict(enumerate(dataset))
        if weights['cider'] > 0:
            score_cider = _compute_cider(_dataset, gts, res)
            score = score_cider + score
        if weights['bleu'] > 0:
            score_bleu4 = _compute_bleu(_dataset, gts, res)
            score = score_bleu4 + score
    else:                   # parallel
        def _get_chunk_index(n_samples, n_chunks):
            chunk_size = n_samples // n_chunks
            r = n_samples % n_chunks
            sizes = [chunk_size + 1 if i < r else chunk_size for i in range(n_chunks)]

            chunks = []
            i = 0
            for size in sizes:
                chunks.append(range(i, i + size))
                i += size
            return chunks

        global pool
        if pool is None:    # initialize thread pool, and initialize each thread
            pool = Pool(processes=n_threads, initializer=init_style_scorer, initargs=[_init_list])
        chunk_index = _get_chunk_index(n_samples=len(res), n_chunks=n_threads)
        chunked_args = []
        for i in range(n_threads):
            _dataset = {}
            _gts = {}
            _res = OrderedDict()
            for _i in chunk_index[i]:
                _dataset[_i] = dataset[_i]
                _gts[_i] = gts[_i]
                _res[_i] = res[_i]

            chunked_args.append([_dataset, _gts, _res])

        if weights['cider'] > 0:
            score_cider = pool.starmap(func=_compute_cider, iterable=chunked_args)
            score_cider = np.concatenate(score_cider)
            score = score_cider * weights['cider'] + score
        if weights['bleu'] > 0:
            score_bleu4 = pool.starmap(func=_compute_bleu, iterable=chunked_args)
            score_bleu4 = np.concatenate(score_bleu4)
            score = score_bleu4 * weights['bleu'] + score

    return score

@arg_type(sample_result_tokens=np.ndarray, greedy_result_tokens=np.ndarray, weights=dict, vocab=Vocabulary)
def _get_self_critical_reward(dataset, sample_result_tokens, greedy_result_tokens, gts_raw, weights, vocab, n_threads=4):
    """
    :param sample_result_tokens: np.array, [batch_size, max_len]
    :param greedy_result_tokens: np.array, [batch_size, max_len_1]
    :param gts_raw: [['raw sent 1', 'raw sent 2'], ['raw sent 3', 'raw sent 4'], ...]
    :param weights: {'bleu': 0.5, 'cider': 0.5}
    :param vocab:
    :return:
    """

    # check shape
    batch_size, max_len = sample_result_tokens.shape
    assert batch_size == len(greedy_result_tokens) and batch_size == len(greedy_result_tokens)

    # check weights
    weight_sum = 0
    for key in ['cider', 'bleu']:
        weight_sum += weights[key]
    assert weight_sum == 1.0, 'invalid reward weight: {}'.format(weights)

    # sample_result_str = [_to_str(_, vocab) for _ in sample_result_tokens]
    sample_result_str = OrderedDict()
    for i, _ in enumerate(sample_result_tokens):
        sample_result_str[i] = _to_str(_, vocab)

    # greedy_result_str = [_to_str(_, vocab) for _ in greedy_result_tokens]
    greedy_result_str = OrderedDict()
    for i, _ in enumerate(greedy_result_tokens):
        greedy_result_str[i] = _to_str(_, vocab)

    gts = {i: gts_raw[i] for i in range(batch_size)}

    # sample_score = get_scores(dataset, sample_result_str, gts, weights, n_threads)
    # greedy_score = get_scores(dataset, greedy_result_str, gts, weights, n_threads)

    _dataset = dataset + dataset
    _result_str = OrderedDict()
    _gts = OrderedDict()
    for i in range(batch_size):
        _result_str[i] = sample_result_str[i]
        _result_str[i + batch_size] = greedy_result_str[i]
        _gts[i] = gts[i]
        _gts[i + batch_size] = gts[i]
    _scores = get_scores(_dataset, _result_str, _gts, weights, n_threads)
    sample_score, greedy_score = _scores[:batch_size], _scores[batch_size:]

    reward = sample_score - greedy_score
    reward = np.repeat(reward[:, np.newaxis], repeats=max_len, axis=1)

    return reward

def get_style_reward(sample_seq, greedy_seq, gts_raw, vocab, style_labels, reward_weights=None):
    if reward_weights is None:
        reward_weights = {'cider': 0.0, 'ppl': 1.0, 'clf': 0.0}

    batch_size, max_len = sample_seq.shape
    assert batch_size == len(greedy_seq) and batch_size == len(greedy_seq)

    sample_sent = [_to_str(i, vocab) for i in sample_seq]
    greedy_sent = [_to_str(i, vocab) for i in greedy_seq]

    dataset = []
    for label in style_labels:
        # style_tag_to_name = {
        #     'humor': 3,
        #     'romantic': 4,
        #     'positive': 5,
        #     'negative': 6,
        # }

        if label == 5 or label == 6:
            dataset.append('senticap')
        else:
            dataset.append('flickrstyle')

        # FIXME: for chinese only !!!
        # dataset.append('chn_styled_word')

    if reward_weights['cider'] > 0:
        reward_cider = _get_self_critical_reward(dataset, sample_seq, greedy_seq, gts_raw, weights={'bleu': 0, 'cider': 1.0}, vocab=vocab)
    else:
        reward_cider = 0.

    if reward_weights['ppl'] > 0:
        _r = 0.
        th = 200.
        for style in all_styles:
            _, score1 = lm_srilm.eval_sentences(all_lm[style], sample_sent)
            _, score2 = lm_srilm.eval_sentences(all_lm[style], greedy_sent)

            reward_ppl = -1 * (np.array(score1) - np.array(score2))
            reward_ppl = np.clip(reward_ppl, -th, th) / th
            reward_ppl = np.sign(reward_ppl)
            mask = style_labels == style
            _r += mask * reward_ppl

        reward_ppl = np.repeat(_r[:, np.newaxis], repeats=max_len, axis=1)
    else:
        reward_ppl = 0.

    if reward_weights['clf'] > 0:
        _r = 0.
        for style in all_styles:
            _clf = all_clf[style]
            _, score1 = clf.eval_sentences(_clf, sample_sent)
            _, score2 = clf.eval_sentences(_clf, greedy_sent)
            reward_clf = score1.astype(int) - score2.astype(int)
            reward_clf = np.sign(reward_clf)

            mask = style_labels == style
            _r += mask * reward_clf

        reward_clf = np.repeat(_r[:, np.newaxis], repeats=max_len, axis=1)

        # FIXME: for chinese only
        # _r = 0.
        # for style in all_styles:
        #     _clf = all_nn_clf[style]
        #     _, score1 = _clf.eval_sentences(sample_sent)
        #     _, score2 = _clf.eval_sentences(greedy_sent)
        #     reward_clf = score1.astype(int) - score2.astype(int)
        #     reward_clf = np.sign(reward_clf)
        #
        #     mask = style_labels == style
        #     _r += mask * reward_clf
        # reward_clf = np.repeat(_r[:, np.newaxis], repeats=max_len, axis=1)
    else:
        reward_clf = 0.

    reward = np.zeros((batch_size, max_len), dtype=np.float)
    reward += reward_weights['cider'] * reward_cider
    reward += reward_weights['ppl'] * reward_ppl
    reward += reward_weights['clf'] * reward_clf

    return reward