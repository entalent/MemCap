import json
import os
import sys
from abc import abstractmethod

from config import *
import pickle
import util

import styled_eval.lm_kenlm as lm_kenlm
import styled_eval.clf as clf_path
from styled_eval.clf import FeatureExtractor
import styled_eval.lm_srilm as lm_srilm
from styled_eval.train_clf_nn_chn import *

class Evaluator:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, ann_file, res_file, imgscores_only=False):
        pass


class NNCLFEvaluator(Evaluator):
    def __init__(self, dataset, style, **kwargs):
        super().__init__(**kwargs)

        info_path = os.path.join(data_path, 'clf_nn', 'model_rnn_info_{}_{}.pkl'.format(dataset, style))
        model_path = os.path.join(data_path, 'clf_nn', 'model_rnn_{}_{}.h5'.format(dataset, style))

        d = pickle.load(open(info_path, 'rb'))
        w, e, tokenizer_config = d['word_index'], d['embeddings_index'], d['tokenizer_config']
        MAX_SEQUENCE_LENGTH, nclasses = d['MAX_SEQUENCE_LENGTH'], d['nclasses']
        tokenizer = tokenizer_from_json(tokenizer_config)

        model_RNN = Build_Model_RNN_Text(word_index=w, embeddings_index=e, nclasses=nclasses,
                                         MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                         EMBEDDING_DIM=d['EMBEDDING_DIM'])
        model_RNN.load_weights(model_path)

        self.tokenizer = tokenizer
        self.model_RNN = model_RNN
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def evaluate(self, ann_file, res_file, imgscores_only=False):
        with open(res_file, 'r') as f:
            d = json.load(f)
            sent_list = [i['caption'] for i in d]

        model_RNN = self.model_RNN
        tokenized = test_tokenize(self.tokenizer, sent_list, self.MAX_SEQUENCE_LENGTH)
        predicted = model_RNN.predict_classes(tokenized, verbose=0)

        if imgscores_only:
            return predicted.tolist()

        overall_score = float(sum(predicted)) / len(predicted)
        img_scores = predicted.tolist()

        return overall_score, img_scores

    def eval_sentences(self, sentences):
        sent_list = sentences
        model_RNN = self.model_RNN
        tokenized = test_tokenize(self.tokenizer, sent_list, self.MAX_SEQUENCE_LENGTH)
        predicted = model_RNN.predict_classes(tokenized, verbose=0)

        overall_score = float(sum(predicted)) / len(predicted)
        img_scores = np.array(predicted.tolist())

        return overall_score, img_scores

    def _get_sent_score(self, sents):
        return self._get_sent_class_label(sents)

    def _get_sent_class_label(self, sents):
        tokenized = test_tokenize(self.tokenizer, sents, self.MAX_SEQUENCE_LENGTH)
        predicted = self.model_RNN.predict_classes(tokenized, verbose=0)
        return predicted

    def _get_sent_prob(self, sents):
        tokenized = test_tokenize(self.tokenizer, sents, self.MAX_SEQUENCE_LENGTH)
        prob = self.model_RNN.predict(tokenized, verbose=0)
        return prob[:, 1]

    def __del__(self):
        try:
            self.session.close()
        except:
            pass


class StyledEvaluate():
    def __init__(self, styled_dataset, style_names, use_lm=True, use_clf=True, use_srilm=True, use_nnclf=False):
        self.styled_dataset = styled_dataset
        self.style_labels = style_names
        self.use_lm = use_lm
        self.use_clf = use_clf
        self.use_srilm = use_srilm
        self.use_nnclf = use_nnclf

        if self.use_lm:
            # load lm
            self.lm = {}
            for style_name in style_names:
                arpa_path = os.path.join(data_path, 'lm', '{}_{}.arpa'.format(styled_dataset, style_name))
                lm = lm_kenlm.load_lm(arpa_path)
                self.lm[style_name] = lm

        if self.use_clf:
            self.clf = {}
            for style_name in style_names:
                clf_path = os.path.join(data_path, 'clf', 'clf_lr_{}_{}.pik'.format(styled_dataset, style_name))
                with open(clf_path, 'rb') as f:
                    self.clf[style_name] = pickle.load(f)

        if self.use_nnclf:
            self.nn_clf = {}
            for style_name in style_names:
                if styled_dataset == 'chn_styled':
                    styled_dataset = 'youku_chn'
                nn_clf = NNCLFEvaluator(styled_dataset, style_name)
                self.nn_clf[style_name] = nn_clf

        if self.use_srilm:
            self.srilm = {}
            for style_name in style_names:
                lm_path = os.path.join(data_path, 'srilm', '{}_{}.srilm'.format(styled_dataset, style_name))
                self.srilm[style_name] = lm_path

    def evaluate(self, ann_file, res_file, style_name, return_img_scores=False):
        metrics, img_scores = util.eval(ann_file, res_file, True)

        with open(res_file, 'r') as f:
            d = json.load(f)
            sent_list = [i['caption'] for i in d]

        if self.use_lm:
            lm_score, _ = lm_kenlm.eval_sentences(self.lm[style_name], sent_list)
            for i, score in enumerate(img_scores):
                score['lm'] = _[i]
            metrics['lm'] = lm_score

        if self.use_clf:
            clf_score, _ = clf_path.eval_sentences(self.clf[style_name], sent_list)
            for i, score in enumerate(img_scores):
                score['clf'] = int(_[i])
            metrics['clf'] = clf_score

        if self.use_srilm:
            lm_score, _ = lm_srilm.eval_sentences(self.srilm[style_name], sent_list)
            for i, score in enumerate(img_scores):
                score['srilm'] = _[i]
            metrics['srilm'] = lm_score

        if self.use_nnclf:
            nnclf_score, _ = self.nn_clf[style_name].evaluate(ann_file, res_file)
            for i, score in enumerate(img_scores):
                score['nnclf'] = _[i]
            metrics['nnclf'] = nnclf_score

        if return_img_scores:
            return metrics, img_scores
        else:
            return metrics


