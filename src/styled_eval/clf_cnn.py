import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import *

from util import load_custom
from config import *

word2idx_path = clf_textcnn_word2idx_path


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.Wt = nn.Linear(input_size, input_size)
        self.Wh = nn.Linear(input_size, input_size)

    def forward(self, x):
        t = torch.sigmoid(self.Wt(x))
        return t * torch.relu(self.Wh(x)) + (1-t) * x


class StyleCLS(nn.Module):
    def __init__(self, word2idx, word_emb_dim=512, filters=(3, 4, 5), out_dim=100):
        super(StyleCLS, self).__init__()
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
        self.unk_idx = word2idx['<UNK>']
        self.word_emb = nn.Embedding(self.vocab_size, word_emb_dim, padding_idx=word2idx['<PAD>'])

        self.conv_list = nn.ModuleList()
        for kernel_size in filters:
            self.conv_list.append(nn.Conv1d(word_emb_dim, out_dim, kernel_size))

        input_dim = len(filters) * out_dim
        self.highway = Highway(input_dim)
        self.fc_drop = nn.Dropout(0.5)
        self.fc = nn.Linear(input_dim, 1)

        # self.weight_cliping()

    def forward(self, captions):
        """

        :param captions:  [batch, seq]
        :return:
        """
        batch_size = captions.size(0)

        word_embs = self.word_emb(captions)  # [batch, seq, word_dim]
        word_embs = word_embs.permute(0, 2, 1)  # [batch, word_dim, seq]
        out = []
        for conv in self.conv_list:
            conv_out = conv(word_embs)  # [batch, out_dim, seq-kernel+1]
            conv_out = torch.relu(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).view(batch_size, -1)  # [batch, out_dim]
            out.append(conv_out)
        out = torch.cat(out, dim=-1)  # [batch, 3*out_dim]
        out = self.highway(out)  # [batch, 3*out_dim]
        out = self.fc_drop(out)
        return self.fc(out).squeeze(1)  # [batch]

    def sample(self, captions):
        # captions: ['a', 'b', 'c', ...]
        self.eval()
        device = self.fc.weight.device

        if isinstance(captions, str):
            captions = captions.split()
        assert len(captions) > 0
        if captions[0] != '<SOS>':
            captions = ['<SOS>'] + captions
        if captions[-1] != '<EOS>':
            captions += ['<EOS>']

        captions_idx = torch.LongTensor(1, len(captions)).to(device)  # [1, seq]
        for i, word in enumerate(captions):
            captions_idx[0, i] = self.word2idx.get(word, self.unk_idx)
        out = self.forward(captions_idx)  # [1]
        out.sigmoid_()
        return 1 if out >= 0.5 else 0

    def weight_cliping(self, limit=0.01):
        for p in self.parameters():
            p.data.clamp_(-limit, limit)

    def load(self, path):
        if torch.cuda.is_available():
            d = torch.load(path)
        else:
            d = torch.load(path, map_location=lambda s, l: s)
        self.load_state_dict(d)
        return self

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.BCEWithLogitsLoss()


def load_clf(clf_path):
    with open(word2idx_path, 'r') as f:
        word2idx = json.load(f)
    clf = StyleCLS(word2idx).cuda()

    state_dict = torch.load(clf_path)
    clf.load_state_dict(state_dict)

    return clf


def eval_sentences(clf, sents, verbose=False):
    outputs = []

    for s in sents:
        output = clf.sample(captions=s.split())
        outputs.append(output)
        if verbose:
            print(s, output)

    score = np.sum(outputs) / len(outputs)
    return score, outputs

def load_sents(dataset_name):
    d = load_custom(os.path.join(annotation_path, 'dataset_{}.json'.format(dataset_name)))['caption_item']
    sents = []
    for item in d:
        sents.extend(item.sentences)
    return sents


def main():
    clf = load_clf('/media/wentian/sdb2/work/styled_caption/data/clf_cnn/15_humor_cls_0.03_0.11_1.00_0.96_0730-1340.model')     # humor
    # clf = load_clf('/media/wentian/sdb2/work/styled_caption/data/clf_cnn/15_roman_cls_0.05_0.12_0.99_0.95_0730-1359.model')     # romantic
    # clf = load_clf('/media/wentian/sdb2/work/styled_caption/data/clf_cnn/16_pos_cls_0.05_0.10_0.99_0.97_0730-1355.model')       # positive
    # clf = load_clf('/media/wentian/sdb2/work/styled_caption/data/clf_cnn/17_neg_cls_0.02_0.13_1.00_0.95_0730-1352.model')     # negative

    # d = json.load(open('/media/wentian/sdb2/work/styled_caption/save/2019-07-31_09-42-29_4style_cat/result_senticap_positive_1.json', 'r'))
    # sents = [i['caption'] for i in d]

    # sents = load_sents('flickrstyle')
    # sents = list(filter(lambda x: x.tag == 'humor', sents))

    # sents = load_sents('senticap')
    # sents = list(filter(lambda x: x.tag == 'positive', sents))

    sents = load_sents('coco')
    sents = random.sample(sents, k=1000)

    sents = [s.raw for s in sents]

    score, _ = eval_sentences(clf, sents, verbose=True)
    print(score)

if __name__ == '__main__':
    main()