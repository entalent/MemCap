import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util
from util.model import LanguageModel
from main.caption.model import seq_softmax

from .model2 import TopDownAttnModel

class Dictionary(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.enabled = True
        self.norm = False

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_s = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.weight_c = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # self.weight = torch.autograd.Variable(torch.Tensor(num_embeddings, embedding_dim)) #.cuda()
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.normal_(0, 1)
        util.truncated_normal_(self.weight_s, mean=0, std=1)
        if self.norm:
            self.weight_s.data = F.normalize(self.weight_s.data, dim=1, p=2)
        util.truncated_normal_(self.weight_c, mean=0, std=1)
        if self.norm:
            self.weight_c.data = F.normalize(self.weight_c.data, dim=1, p=2)

    def _attn(self, e_c):   # e_c: (batch_size, emb_dim)
        if self.norm:
            e_c = F.normalize(e_c, p=2, dim=-1)
            m = F.normalize(self.weight_c, p=2, dim=-1)
        else:
            m = self.weight_c   # (num_emb, emb_dim)

        w = torch.matmul(e_c, m.transpose(0, 1))
        w = F.softmax(w, dim=1)     # (batch_size, num_emb)

        return w

    def update(self, e_content, e_style):
        x = e_content

        assert x.dim() == 2 and x.shape[1] == self.embedding_dim  # x: (batch_size, self.embedding_dim)
        batch_size = x.shape[0]

        w = self._attn(x)       # (batch_size, num_emb)

        w = w.unsqueeze(2).expand(batch_size, self.num_embeddings, self.embedding_dim)
        w = w.transpose(2, 1)   # (batch_size, embedding_dim, num_embeddings)

        o = w * e_style.unsqueeze(-1)   # (batch_size, embedding_dim, num_embeddings)
        o = torch.sum(o, dim=0, keepdim=False)  # (embedding_dim, num_embeddings)
        o = o.transpose(0, 1)

        o1 = w * e_content.unsqueeze(-1)
        o1 = torch.sum(o1, dim=0, keepdim=False)
        o1 = o1.transpose(0, 1)

        if (torch.isnan(self.weight_s).any()):
            print('s --')

        if (torch.isnan(self.weight_c).any()):
            print('c --')

        if self.enabled:
            self.weight_s.data += o
            self.weight_c.data += o1

            if self.norm:
                self.weight_s.data = F.normalize(self.weight_s.data, dim=1, p=2)
                self.weight_c.data = F.normalize(self.weight_c.data, dim=1, p=2)

    def forward(self, x):
        """
        :param x: the tensor to encode, shape: (... , embedding_dim)
        :return:
        """
        assert x.shape[-1] == self.embedding_dim
        s = x.shape

        _x = x.reshape(-1, self.embedding_dim)  # (?, embedding_dim)

        batch_size = _x.shape[0]
        w = self._attn(_x)

        w = w.unsqueeze(2).expand(batch_size, self.num_embeddings, self.embedding_dim)
        w = w.transpose(2, 1)  # (batch_size, embedding_dim, num_embeddings)

        choiced_mem = w * self.weight_s.transpose(0, 1).unsqueeze(0)   # (batch_size, embedding_dim, num_embeddings)
        choiced_mem = choiced_mem.sum(dim=2, keepdim=False)          # (batch_size, embedding_dim)

        __x = choiced_mem.reshape(*s)

        if not self.enabled:
            __x *= 0.

        return __x

class TopDownAttnModelSG(TopDownAttnModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mem_size = kwargs.get('mem_size', 100)

        self.rel_embedding = nn.Linear(self.embedding_dim * 3, self.embedding_dim)

        self.style_mem = Dictionary(embedding_dim=self.embedding_dim, num_embeddings=self.mem_size)
        self.style_w = nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim)

        self.use_mem = kwargs.get('use_mem', True)

    def load_state_dict(self, state_dict, strict=True):
        _state_dict = self.state_dict()

        keys = set(state_dict.keys())
        for k in keys:
            if k not in _state_dict:
                del state_dict[k]

        for k, v in _state_dict.items():
            if k not in state_dict:
                state_dict[k] = _state_dict[k]
            if state_dict[k].shape != _state_dict[k].shape:
                state_dict[k] = _state_dict[k]

        for k in _state_dict.keys():
            print(k, _state_dict[k].shape, state_dict[k].shape)

        return super().load_state_dict(state_dict, False)

    def _get_rel_element_emb(self, s):
        strs = s.split()
        if len(strs) == 1:
            strs = [s, s]
        index = [self.vocab.get_index(i) for i in strs][:2]
        return index

    def prepare_feat(self, input_feature, **kwargs):
        (tokens, lengths, style_mask, sg_batch) = input_feature[:4]
        batch_size, max_len = tokens.shape
        device = tokens.device

        _object_words, _rel_words = [], []
        object_size, rel_size = [], []
        for i, sg in enumerate(sg_batch):
            objects = [self.vocab.get_index(i) for i in sg['obj']]
            rels = []
            for r in sg['rel']:
                subj_id, pred_id, obj_id = [self._get_rel_element_emb(s) for s in r[:3]]
                word_ids = subj_id + pred_id + obj_id
                rels.append(word_ids)

            _object_words.extend(objects)
            object_size.append(len(objects))
            _rel_words.extend(rels)
            rel_size.append(len(rels))

        object_words = torch.LongTensor(_object_words).to(device)
        rel_words = torch.LongTensor(_rel_words).to(device)
        rel_flag = len(rel_words) > 0

        object_emb = self.input_embedding(object_words)

        if rel_flag:
            rel_emb = self.input_embedding(rel_words).reshape(rel_words.shape[0], 3, 2, self.embedding_dim).mean(dim=2)
            rel_emb = rel_emb.reshape(rel_words.shape[0], 3 * self.embedding_dim)
            rel_emb = self.rel_embedding(rel_emb)

        # print(object_words.shape, rel_words.shape)

        input_size = [object_size[i] + rel_size[i] for i in range(batch_size)]

        att_mask = self.input_embedding.weight.new_zeros(batch_size, max(input_size))
        for i in range(batch_size):
            att_mask[i, :input_size[i]] = 1

        input_emb = self.input_embedding.weight.new_zeros(batch_size, max(input_size), self.embedding_dim)
        input_emb_avg = self.input_embedding.weight.new_zeros(batch_size, self.embedding_dim)

        _i1 = 0
        _i2 = 0
        for i in range(batch_size):
            _obj_emb = object_emb[_i1 : _i1 + object_size[i]];  _i1 += object_size[i]

            if rel_flag:
                _rel_emb = rel_emb[_i2 : _i2 + rel_size[i]];        _i2 += rel_size[i]
                emb = torch.cat([_obj_emb, _rel_emb], dim=0)
            else:
                emb = _obj_emb

            input_emb[i, :emb.shape[0]] = emb
            input_emb_avg[i] = emb.mean(dim=0)

        # TODO: why use lengths[i] ?
        for i in range(batch_size):
            input_emb_avg[i, :] = torch.mean(input_emb[i, :lengths[i], :], dim=0, keepdim=False)

        if self.use_mem:
            # from tokens
            _style_mask = style_mask.unsqueeze(2).expand(batch_size, max_len, self.embedding_dim)
            word_emb = self.input_embedding(tokens)
            s_con = torch.sum(word_emb * (1 - _style_mask), dim=1, keepdim=False)  # (batch_size, embedding_dim)

            # from sg
            # s_con = input_emb_avg

            if self.training:
                _style_mask = style_mask.unsqueeze(2).expand(batch_size, max_len, self.embedding_dim)
                word_emb = self.input_embedding(tokens)  # (batch_size, max_length, embedding_dim)
                s_style = torch.sum(word_emb * _style_mask, dim=1, keepdim=False)  # (batch_size, embedding_dim)
                self.style_mem.update(s_con, s_style)
            chosen_mem = self.style_mem.forward(s_con)  # (batch_size, embedding_dim)

            c_0 = self.style_w(chosen_mem)
        else:
            c_0 = torch.zeros((batch_size, self.hidden_dim)).to(device)

        h_0 = torch.zeros((batch_size, self.hidden_dim)).to(device)

        img_feat_attn = input_emb
        img_feat_avg = input_emb_avg
        batch_size, attn_size, _ = img_feat_attn.shape
        img_feat_avg = self.image_embedding_avg(img_feat_avg)
        img_feat_attn = self.image_embedding(img_feat_attn)
        p_att_feats = self.ctx2att(img_feat_attn)
        return batch_size, (img_feat_avg, img_feat_attn, p_att_feats, att_mask, h_0, c_0)


    def init_state(self, input_feature, **kwargs):
        (img_feat_avg, img_feat_attn, p_att_feats, att_mask, h_0, c_0) = input_feature
        return (h_0, c_0), (h_0, c_0)
