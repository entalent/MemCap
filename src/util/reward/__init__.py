import time
from collections import OrderedDict

import numpy as np
import torch
from joblib import Parallel, delayed

from .. import arg_type, trim_generated_tokens
from ..vocabulary import Vocabulary
from .bleu.bleu import *
from .ciderD.ciderD import *

ciderD_scorer = None
bleu4_scorer = None


def init_scorer(df):
    global ciderD_scorer
    global bleu4_scorer

    if ciderD_scorer is None:
        print('init CIDEr-D scorer using df {}'.format(df))
        ciderD_scorer = CiderD(df=df)
    if bleu4_scorer is None:
        bleu4_scorer = Bleu(4)

def get_scores(res, gts, weights):
    """
        :param res: ['candidate1', 'candidate2']
        :param gts: {0: ['sent1', 'sent2'], 1: ['sent3', 'sent4']}
        :param weights: {'cider': 0.5, 'bleu': 0.5}
        :return:
        """

    def _compute_cider(gts, res, weight):
        res_ = [{'image_id': i, 'caption': [res[i]]} for i in range(len(res))]
        _, score_cider = ciderD_scorer.compute_score(gts, res_)
        return score_cider * weight

    def _compute_bleu(gts, res, weight):
        res__ = {i: [res[i]] for i in range(len(res))}
        _, score_bleu = bleu4_scorer.compute_score(gts, res__)
        score_bleu4 = np.array(score_bleu[3])
        return score_bleu4 * weight

    score = 0.

    # single thread
    if weights['cider'] > 0:
        score_cider = _compute_cider(gts, res, weights['cider'])
        score = score_cider + score
    if weights['bleu'] > 0:
        score_bleu4 = _compute_bleu(gts, res, weights['cider'])
        score = score_bleu4 + score

    return score


@arg_type(tokens=[list, np.ndarray], vocab=Vocabulary)
def _to_str(tokens, vocab):
    trimmed_tokens = trim_generated_tokens(tokens)
    s = ' '.join(vocab.get_word(i) for i in trimmed_tokens)
    s += ' ' + Vocabulary.end_token
    return s


@arg_type(sample_result_tokens=np.ndarray, greedy_result_tokens=np.ndarray, weights=dict, vocab=Vocabulary)
def get_self_critical_reward(sample_result_tokens, greedy_result_tokens, gts_raw, weights, vocab):
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

    sample_score = get_scores(sample_result_str, gts, weights)
    greedy_score = get_scores(greedy_result_str, gts, weights)

    reward = sample_score - greedy_score
    reward = np.repeat(reward[:, np.newaxis], repeats=max_len, axis=1)

    return reward

@arg_type(log_prob=torch.Tensor, generated_seq=np.ndarray, reward=np.ndarray)
def rl_criterion(log_prob, generated_seq, reward):
    """
    :param log_prob: (batch_size, max_len), log probability
    :param generated_seq: (batch_size, max_len)
    :param reward: (batch_size, max_len)
    :return:
    """
    device = log_prob.device

    batch_size, max_len = log_prob.shape[:2]

    assert (batch_size, max_len) == (reward.shape[0], reward.shape[1])
    assert (batch_size, max_len) == (generated_seq.shape[0], generated_seq.shape[1])

    mask = np.zeros(shape=(batch_size, max_len), dtype=np.float32)
    for i in range(batch_size):
        for j in range(max_len):
            mask[i][j] = 1
            if generated_seq[i][j] == Vocabulary.end_token_id:
                break
    mask = torch.Tensor(mask).to(device)
    reward = torch.Tensor(reward).to(device)

    log_prob_flat = log_prob.view(-1, 1)
    log_probs_seq = log_prob_flat
    reward_flat = reward.reshape(shape=(-1, 1))    # (batch_size * max_len, 1)
    mask_flat = mask.reshape(shape=(-1, 1))
    loss = - log_probs_seq * reward_flat * mask_flat
    loss = torch.sum(loss) / torch.sum(mask)
    return loss
