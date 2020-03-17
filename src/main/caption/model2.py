import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.model import *


class Attention(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(Attention, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_mask=None):
        # att_mask: batch_size * att_size

        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        # dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        # weight = F.softmax(dot)  # batch * att_size
        weight = F.softmax(dot, dim=-1)  # batch * att_size

        if att_mask is not None:
            weight = weight * att_mask.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1

        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class TopDownAttnModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_args = {
            'feat_dim': 2048, 'embedding_dim': 300, 'hidden_dim': 512, 'dropout_prob': 0.5,
            'attn_hidden_dim': 512,
        }
        default_args.update(kwargs)
        kwargs = default_args

        feat_dim = kwargs['feat_dim']
        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']
        image_embedding_dim = hidden_dim
        dropout_prob = kwargs['dropout_prob']
        attn_hidden_dim = kwargs['attn_hidden_dim']

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim

        self.input_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_dim)
        self.image_embedding = nn.Sequential(nn.Linear(feat_dim, image_embedding_dim),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_prob))
        self.image_embedding_avg = nn.Sequential(nn.Linear(feat_dim, image_embedding_dim),
                                                 nn.ReLU(),
                                                 nn.Dropout(dropout_prob))

        self.lstm_0 = nn.LSTMCell(input_size=hidden_dim+image_embedding_dim+embedding_dim,
                                  hidden_size=hidden_dim)
        self.ctx2att = nn.Linear(image_embedding_dim, attn_hidden_dim)
        self.att = Attention(rnn_size=hidden_dim, att_hid_size=attn_hidden_dim)
        self.lstm_1 = nn.LSTMCell(input_size=image_embedding_dim+hidden_dim, hidden_size=hidden_dim)
        self.output_embedding = nn.Linear(in_features=hidden_dim, out_features=len(self.vocab))

    def prepare_feat(self, input_feature, **kwargs):
        img_feat_avg, img_feat_attn, att_mask = input_feature
        assert len(img_feat_avg) == len(img_feat_attn), 'batch size not consistent: {}, {}'.format(img_feat_avg.shape, img_feat_attn.shape)
        batch_size, attn_size, _ = img_feat_attn.shape
        img_feat_avg = self.image_embedding_avg(img_feat_avg)
        img_feat_attn = self.image_embedding(img_feat_attn)
        p_att_feats = self.ctx2att(img_feat_attn)
        return batch_size, (img_feat_avg, img_feat_attn, p_att_feats, att_mask)

    def init_state(self, input_feature, **kwargs):
        (img_feat_avg, img_feat_attn, p_att_feats, att_mask) = input_feature[:4]
        device = img_feat_avg.device
        batch_size = len(img_feat_avg)

        h_0 = torch.zeros((batch_size, self.hidden_dim)).to(device)
        c_0 = h_0
        return (h_0, c_0), (h_0, c_0)

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        (img_feat_avg, img_feat_attn, p_att_feats, att_mask) = input_feature[:4]
        device = img_feat_avg.device
        batch_size, attn_size, dim_feat = img_feat_attn.shape
        if not torch.is_tensor(last_word_id_batch):
            last_word_id_batch = torch.LongTensor(last_word_id_batch).to(device)


        (h_attn_0, c_attn_0), (h_lang_0, c_lang_0) = last_state     # h_lang_0: (batch_size, hidden_dim)

        last_word_embedding = self.input_embedding(last_word_id_batch)  # (batch_size, embedding_dim)
        x_attn = torch.cat([h_lang_0, img_feat_avg, last_word_embedding], dim=1)
        h_attn_1, c_attn_1 = self.lstm_0(x_attn, (h_attn_0, c_attn_0))

        att = self.att(h_attn_1, img_feat_attn, p_att_feats, att_mask)

        x_lang = torch.cat([att, h_attn_1], dim=1)
        h_lang_1, c_lang_1 = self.lstm_1(x_lang, (h_lang_0, c_lang_0))

        _output = F.dropout(h_lang_1, self.dropout_prob, self.training)
        output = self.output_embedding(_output)

        current_state = ((h_attn_1, c_attn_1), (h_lang_1, c_lang_1))

        return output, current_state, None  # output: (batch_size, vocab_size) not normalized



