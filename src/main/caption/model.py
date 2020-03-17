import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import *
from util.model import *


def seq_softmax(logits, lengths, dim=-1):
    """

    :param logits: torch.Tensor, (batch_size, max_len)
    :param lengths: torch.LongTensor, (batch_size)
    :return:
    """
    eps = 1e-12

    assert logits.shape[0] == lengths.shape[0]
    batch_size = logits.shape[0]

    mask = logits.new_zeros(size=logits.shape, dtype=logits.dtype)
    for i in range(batch_size):
        mask[i][:lengths[i]] = 1

    result = F.softmax(logits * mask, dim)
    result = result * mask
    result = result / (result.sum(dim=1, keepdim=True) + eps)

    return result


class LSTMLanguageModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_args = {
            'feat_dim': 2048, 'embedding_dim': 300, 'hidden_dim': 512, 'dropout_prob': 0.5
        }
        default_args.update(kwargs)
        kwargs = default_args
        print('init {} using args {}'.format(self.__class__.__name__, kwargs))

        feat_dim = kwargs['feat_dim']
        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']
        dropout_prob = kwargs['dropout_prob']

        embedding = kwargs.get('pretrained_embedding', None)

        if embedding is None:
            self.use_pretrained_embedding = False
            print('init word embedding')
            self.word_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_dim,
                                               padding_idx=0, _weight=embedding)
        else:
            self.use_pretrained_embedding = True
            print('use pre-trained word embedding')
        self.image_embedding = nn.Linear(in_features=feat_dim, out_features=embedding_dim)
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_dim)
        self.output_embedding = nn.Linear(in_features=hidden_dim, out_features=len(self.vocab))
        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if not self.use_pretrained_embedding:
            self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.bias.data.fill_(0)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)

    def prepare_feat(self, input_feature, **kwargs):
        batch_size = len(input_feature)
        prepared_feat = self.image_embedding(input_feature)
        return batch_size, prepared_feat

    def init_state(self, input_feature, **kwargs):
        device = input_feature.device
        batch_size = input_feature.shape[0]
        h_0 = torch.zeros((batch_size, self.lstm.hidden_size)).to(device)
        return self.lstm(input_feature, (h_0, h_0))

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        device = input_feature.device
        if not isinstance(last_word_id_batch, torch.Tensor):
            last_word_id_batch = torch.LongTensor(np.array(last_word_id_batch).astype(np.int64)).to(device)
        emb = self.word_embedding(last_word_id_batch)
        h, c = self.lstm(emb, last_state)
        output = self.dropout(h)
        output = self.output_embedding(output)
        return output, (h, c), None


class LanguageGenerator(LanguageModel):
    def __init__(self, **kwargs):
        default_args = {
            'encoder_embedding_dim': 300,
            'decoder_embedding_dim': 300,
            'style_embedding_dim': 0,
            'hidden_dim': 512, 'dropout_prob': 0.5,
            'encoder_embedding': None, 'decoder_embedding': None,
            'encoder_vocab': None, 'decoder_vocab': None
        }
        default_args.update(kwargs)
        kwargs = default_args
        print('init {} using args {}'.format(self.__class__.__name__, kwargs))
        kwargs['vocab'] = kwargs['decoder_vocab']

        super().__init__(**kwargs)

        self.encoder_embedding_dim = kwargs['encoder_embedding_dim']
        self.decoder_embedding_dim = kwargs['decoder_embedding_dim']
        self.hidden_dim = kwargs['hidden_dim']

        if kwargs['encoder_embedding'] is None:
            self.encoder_embedding = nn.Embedding(num_embeddings=len(kwargs['encoder_vocab']),
                                                  embedding_dim=self.encoder_embedding_dim)
        else:
            self.encoder_embedding = kwargs['encoder_embedding']
        self.encoder_embedding = nn.Sequential(
            self.encoder_embedding,
            nn.Dropout(0.3)
        )

        if kwargs['decoder_embedding'] is None:
            self.decoder_embedding = nn.Embedding(num_embeddings=len(kwargs['decoder_vocab']),
                                                  embedding_dim=self.decoder_embedding_dim)
        else:
            self.decoder_embedding = kwargs['decoder_embedding']
        self.decoder_embedding = nn.Sequential(
            self.decoder_embedding,
            nn.Dropout(0.3)
        )

        # style embedding
        self.style_embedding_dim = kwargs['style_embedding_dim']
        self.use_style_embedding = self.style_embedding_dim > 0
        if self.use_style_embedding:
            self.style_embedding = nn.Embedding(num_embeddings=10, embedding_dim=self.style_embedding_dim)

        self.encoder = nn.GRU(input_size=self.encoder_embedding_dim, hidden_size=self.hidden_dim // 2,
                              batch_first=True, bidirectional=True)

        self.fc_attn = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.decoder = nn.GRUCell(input_size=self.encoder_embedding_dim, hidden_size=self.hidden_dim)
        self.fc_output = nn.Linear(in_features=self.hidden_dim * 2, out_features=len(kwargs['decoder_vocab']))

    def prepare_feat(self, input_feature, **kwargs):
        """

                :param input_feature: tokens, lengths, style_label
                :param kwargs:
                :return:
                """
        tokens, lengths = input_feature[:2]     # torch.Tensor, torch.LongTensor
        if self.use_style_embedding:
            style_labels = input_feature[2]     # torch.LongTensor (batch_size)
        else:
            style_labels = None

        device = tokens.device
        assert len(tokens) == len(lengths)
        batch_size, max_length = tokens.shape

        input_emb = self.encoder_embedding(tokens)
        outputs, _ = self.encoder(input_emb)
        # outputs: (batch_size, max_length, hidden_dim)

        index = lengths.view(batch_size, 1, 1).expand(batch_size, 1, outputs.shape[-1]) - 1
        index = index.to(device)  # (batch_size, 1, hidden_dim)
        h_enc = torch.gather(outputs, dim=1, index=index)  # (batch_size, 1, hidden_dim)
        p_outputs = self.fc_attn(outputs)  # p_outputs: (batch_size, max_len, hidden_dim)
        return batch_size, (p_outputs, h_enc, lengths, style_labels)

    def init_state(self, input_feature, **kwargs):
        p_outputs, h_enc, lengths, style_labels = input_feature
        h_enc = h_enc.squeeze(1)
        return h_enc

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        p_outputs, h_enc, lengths, style_labels = input_feature
        device = p_outputs.device
        # p_outputs: (batch_size, max_len, hidden_dim)
        batch_size, max_len = p_outputs.shape[0], p_outputs.shape[1]

        if not torch.is_tensor(last_word_id_batch):
            last_word_id_batch = torch.LongTensor(last_word_id_batch).to(device)

        x = self.decoder_embedding(last_word_id_batch)
        if self.use_style_embedding:
            se = self.style_embedding(style_labels)     # (batch_size, style_emb_dim)
            decoder_input = torch.cat([x, se], dim=1)
        else:
            decoder_input = x
        h_dec = self.decoder(decoder_input, last_state)     # (batch_size, hidden_dim)

        attn_weight = torch.bmm(p_outputs, h_dec.unsqueeze(2))      # (batch_size, max_len, 1)
        attn_weight = attn_weight.squeeze(2)    # (batch_size, max_len)
        attn_weight = seq_softmax(attn_weight, lengths)
        attn_weight = attn_weight.unsqueeze(2).expand(*p_outputs.shape)

        c = (attn_weight * p_outputs).sum(dim=1, keepdim=False)     # (batch_size, hidden_dim)

        c = torch.cat([h_dec, c], dim=1)
        logits = self.fc_output(c)

        return logits, h_dec


class MultiAttnModel(LanguageModel):
    def __init__(self, **kwargs):
        default_args = {
            'encoder_embedding_dim': 300,
            'decoder_embedding_dim': 300,
            'hidden_dim': 512, 'dropout_prob': 0.5,
            'encoder_embedding': None, 'decoder_embedding': None,
            'encoder_vocab': None, 'decoder_vocab': None,
        }
        default_args.update(kwargs)
        kwargs = default_args
        print('init {} using args {}'.format(self.__class__.__name__, kwargs))
        kwargs['vocab'] = kwargs['decoder_vocab']

        super().__init__(**kwargs)

        self.encoder_embedding_dim = kwargs['encoder_embedding_dim']
        self.decoder_embedding_dim = kwargs['decoder_embedding_dim']
        self.hidden_dim = kwargs['hidden_dim']

        self.triplet_dim = 3 * self.decoder_embedding_dim

        if kwargs['encoder_embedding'] is None:
            self.encoder_embedding = nn.Embedding(num_embeddings=len(kwargs['encoder_vocab']),
                                                  embedding_dim=self.encoder_embedding_dim)
        else:
            self.encoder_embedding = kwargs['encoder_embedding']
        self.encoder_embedding = nn.Sequential(
            self.encoder_embedding,
            nn.Dropout(0.3)
        )

        if kwargs['decoder_embedding'] is None:
            self.decoder_embedding = nn.Embedding(num_embeddings=len(kwargs['decoder_vocab']),
                                                  embedding_dim=self.decoder_embedding_dim)
        else:
            self.decoder_embedding = kwargs['decoder_embedding']
        self.decoder_embedding = nn.Sequential(
            self.decoder_embedding,
            nn.Dropout(0.3)
        )

        self.encoder = nn.GRU(input_size=self.encoder_embedding_dim, hidden_size=self.hidden_dim // 2,
                              batch_first=True, bidirectional=True)

        self.fc_attn = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.decoder = nn.GRUCell(input_size=self.encoder_embedding_dim, hidden_size=self.hidden_dim)
        self.fc_output = nn.Linear(in_features=self.hidden_dim * 3, out_features=len(kwargs['decoder_vocab']))

        self.fc_attn_1 = nn.Linear(in_features=self.decoder_embedding_dim * 3, out_features=self.hidden_dim)

    def prepare_feat(self, input_feature, **kwargs):
        tokens, lengths, triplets = input_feature
        n_triplets = triplets.shape[1]
        # triplets: [batch_size, 10, 3]
        device = tokens.device
        assert len(tokens) == len(lengths)
        assert len(tokens) == len(triplets)
        batch_size, max_length = tokens.shape

        triplet_emb = self.decoder_embedding(triplets)   # (batch_size, 10, 3, 300)
        triplet_emb = triplet_emb.reshape(batch_size, triplet_emb.shape[1], triplet_emb.shape[2] * triplet_emb.shape[3])
        p_triplet = self.fc_attn_1(triplet_emb)

        input_emb = self.encoder_embedding(tokens)
        outputs, _ = self.encoder(input_emb)
        # outputs: (batch_size, max_length, hidden_dim)

        index = lengths.view(batch_size, 1, 1).expand(batch_size, 1, outputs.shape[-1]) - 1
        index = index.to(device)  # (batch_size, 1, hidden_dim)
        h_enc = torch.gather(outputs, dim=1, index=index)  # (batch_size, 1, hidden_dim)
        p_outputs = self.fc_attn(outputs)  # p_outputs: (batch_size, max_len, hidden_dim)
        return batch_size, (p_outputs, h_enc, lengths, p_triplet)

    def init_state(self, input_feature, **kwargs):
        p_outputs, h_enc, lengths, p_triplet = input_feature
        h_enc = h_enc.squeeze(1)
        return h_enc

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        p_outputs, _, lengths, p_triplet = input_feature
        device = p_outputs.device
        # p_outputs: (batch_size, max_len, hidden_dim)
        batch_size, max_len = p_outputs.shape[0], p_outputs.shape[1]

        if not torch.is_tensor(last_word_id_batch):
            last_word_id_batch = torch.LongTensor(last_word_id_batch).to(device)

        x = self.decoder_embedding(last_word_id_batch)
        h_dec = self.decoder(x, last_state)     # (batch_size, hidden_dim)

        attn_weight = torch.bmm(p_outputs, h_dec.unsqueeze(2))      # (batch_size, max_len, 1)
        attn_weight = attn_weight.squeeze(2)    # (batch_size, max_len)
        attn_weight = seq_softmax(attn_weight, lengths)
        attn_weight = attn_weight.unsqueeze(2).expand(*p_outputs.shape)

        attn_weight_1 = torch.bmm(p_triplet, h_dec.unsqueeze(2))
        attn_weight_1 = attn_weight_1.squeeze(2)
        attn_weight_1 = F.softmax(attn_weight_1, dim=-1)
        attn_weight_1 = attn_weight_1.unsqueeze(2).expand(*p_triplet.shape)
        tp = (attn_weight_1 * p_triplet).sum(dim=1, keepdim=False)  # (batch_size, hidden_dim)

        c = (attn_weight * p_outputs).sum(dim=1, keepdim=False)  # (batch_size, hidden_dim)

        c = torch.cat([h_dec, c, tp], dim=1)
        logits = self.fc_output(c)

        return logits, h_dec


def main():
    pass


if __name__ == '__main__':
    main()

