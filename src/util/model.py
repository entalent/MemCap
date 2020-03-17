from abc import abstractmethod
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import arg_type


BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq', 'metadata_seq'])


class LanguageModel(nn.Module):
    """
    abstract class for RNN based captioning model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']

    @arg_type(input_sentence=torch.Tensor)
    def forward(self, input_feature, input_sentence, **kwargs):
        """

        :param input_feature: torch.Tensor or list/tuple of torch.Tensor
        :param input_sentence: torch.LongTensor
        :param kwargs: other arguments, including 'ss_prob'
        :return: output logits, shape: (batch_size, max_len, vocab_size)
        """
        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)
        assert len(input_sentence.shape) == 2 and input_sentence.shape[0] == batch_size
        max_length = input_sentence.shape[1]
        device = input_sentence.device

        ss_prob = kwargs.get('ss_prob', 0.0)

        state = self.init_state(input_feature, **kwargs)
        all_outputs = []
        for i in range(max_length):
            if self.training and i >= 1 and ss_prob > 0:    # scheduled sampling
                sample_mask = torch.zeros(batch_size).uniform_(0, 1) < ss_prob
                if sample_mask.sum() == 0:
                    word_id_batch = input_sentence[:, i]
                else:
                    sample_ind = sample_mask.nonzero().view(-1).to(device)
                    word_id_batch = input_sentence[:, i]            # (batch_size,)
                    prob_prev = torch.exp(F.log_softmax(all_outputs[-1], dim=-1))
                    word_id_batch.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                word_id_batch = input_sentence[:, i]

            _ret = self.step(input_feature, word_id_batch, state,
                             **kwargs)  # output: (batch_size, vocab_size) WITHOUT SOFTMAX
            output, state = _ret[:2]

            assert output.shape[0] == batch_size and output.shape[1] == len(self.vocab), \
                'expected output shape: {}, get shape {}'.format((batch_size, len(self.vocab)), output.shape)
            all_outputs.append(output)

        all_outputs = torch.stack(all_outputs, 1)     # (batch_size, max_len, vocab_size)
        return all_outputs

    def sample(self, input_feature, max_length, sample_max=True, **kwargs):
        """
        sample using greedy search (or using torch.multinomial)
        can be used in evaluation, or self critical training
        :param input_feature: torch.Tensor or list/tuple of torch.Tensor
        :param max_length: int, max sentence length
        :param sample_max: if specified, use greedy search
        :param kwargs:
        :return: log_prob_seq (torch.Tensor), word_id_seq (np.array), all_metadata (list, information collected during `sample`)
        """

        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)

        state = self.init_state(input_feature, **kwargs)
        log_prob_seq = []
        log_prob_sum = []

        last_word_id_batch = [start_word_id] * batch_size
        word_id_seq = []

        unfinished_flag = [1 for _ in range(batch_size)]

        all_metadata = []

        for t in range(max_length):
            _ret = self.step(input_feature=input_feature,
                                                     last_word_id_batch=last_word_id_batch,
                                                     last_state=state, **kwargs)
            output, state = _ret[:2]; step_metadata = _ret[2] if len(_ret) > 2 else None
            log_prob = F.log_softmax(output, -1)    # (batch_size, vocab_size)

            if sample_max:
                word_log_prob, word_id = torch.max(log_prob, dim=-1)    # word_id: (batch_size,)
            else:
                word_id = torch.multinomial(torch.exp(log_prob), num_samples=1)    # word_id: (batch_size, 1)
                word_log_prob = log_prob.gather(dim=1, index=word_id)              # word_log_prob: (batch_size, 1)
                word_id = word_id.squeeze(1)                    # word_id: (batch_size,)
                word_log_prob = word_log_prob.squeeze(1)        # word_log_prob: (batch_size,)

            if t == 0:
                unfinished_flag = word_id != end_word_id
            else:
                unfinished_flag = unfinished_flag * (word_id != end_word_id)

            _word_id = word_id.clone()
            _word_id[unfinished_flag == 0] = end_word_id

            word_id_seq.append(_word_id)
            last_word_id_batch = _word_id
            log_prob_seq.append(word_log_prob)

            all_metadata.append(step_metadata)

            if unfinished_flag.sum() == 0:
                break

        log_prob_seq = torch.stack(log_prob_seq, dim=1)     # (batch_size, seq_len)
        word_id_seq = torch.stack(word_id_seq, dim=1)       # (batch_size, seq_len)

        # log_prob_seq = log_prob_seq.detach().cpu().numpy()
        word_id_seq = word_id_seq.detach().cpu().numpy()

        return log_prob_seq, word_id_seq, all_metadata

    def sample_beam(self, input_feature, max_length, beam_size, **kwargs):
        """
        perform beam search
        :param input_feature: torch.Tensor or list/tuple of torch.Tensor
        :param max_length: int, max sentence length
        :param beam_size: candidates to keep at each step
        :param kwargs:
        :return: log_prob_seq (np.array), word_id_seq (np.array), all_metadata (list)
        TODO: change log_prob_seq to torch.Tensor
        """
        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)
        assert(batch_size == 1)

        start_token = kwargs.get('start_token', start_word_id)

        initial_state = self.init_state(input_feature, **kwargs)
        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq, metadata_seq
        candidates = [BeamCandidate(initial_state, 0., [], start_token, [], [])]

        for t in range(max_length):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq, step_metadata_history = candidate
                if last_word_id == end_word_id and t > 0:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    _ret = self.step(input_feature=input_feature, last_word_id_batch=[last_word_id],
                                     last_state=state, **kwargs)
                    output, state = _ret[:2]; step_metadata = _ret[2] if len(_ret) > 2 else None

                    output = F.log_softmax(output, -1).squeeze(0).detach().cpu()  # log of probability
                    output_sorted, index_sorted = torch.sort(output, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        word_id = int(word_id.numpy())
                        # log_prob = float(log_prob.numpy())
                        tmp_candidates.append(BeamCandidate(state,
                                               log_prob_sum + log_prob, log_prob_seq + [log_prob],
                                               word_id, word_id_seq + [word_id],
                                               step_metadata_history + [step_metadata]))
            candidates = sorted(tmp_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            # candidates = sorted(tmp_candidates, key=lambda x: x[1] / len(x[-1]), reverse=True)[:beam_size]
            if end_flag:
                break

        # log_prob_seq, word_id_seq, metadata_seq
        return np.array(candidates[0].log_prob_seq), np.array(candidates[0].word_id_seq), candidates[0].metadata_seq

    @abstractmethod
    def prepare_feat(self, input_feature, **kwargs):
        """
        prepare input_feature for next steps
        :param input_feature: feature from dataloader
        :return: batch_size (int), prepared_feature (torch.Tensor or list/tuple of torch.Tensor)
        """
        return 0, None

    @abstractmethod
    def init_state(self, input_feature, **kwargs):
        """
        perform steps before feeding <start> token, i.e. input image in NIC model
        :param input_feature: return value of `self.prepare_feat`
        :return: initial RNN state
        """
        pass

    @abstractmethod
    @arg_type(last_word_id_batch=[np.ndarray, list, tuple])
    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        """
        take the previous word as input and output the next word
        :param input_feature: returned by sample_prepare_feat, batched
        :param last_word_id_batch: torch.Tensor | np.array | list of int
        :param last_state: batched
        :return: output (without softmax, shape is [batch_size, vocab_size]),
                 state,
                 step_metadata (optional, other info in this step, i.e. attention weights to visualize)
        """
        return None, None, None
