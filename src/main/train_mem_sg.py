import os
import sys
sys.path.append('.')
import csv

import numpy as np
import torch.utils.data
from torch.utils.data.sampler import WeightedRandomSampler

from config import *
import util
from styled_eval import *
from util import *
import util.reward
from main.data import *
from main.caption.model1 import *
from util.pretrained_embedding import *

from util.reward.style_reward import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def add_style_tokens(vocab, style_tokens):
    new_vocab = Vocabulary()
    words = vocab.idx2word[2:]
    for token in style_tokens:
        new_vocab._add_word(token)
    for w in words:
        new_vocab._add_word(w)
    return new_vocab


class StyledTextPipeline(SupervisedPipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-styles', default='', help='humor,romantic,positive,negative')
        parser.add_argument('-sc_after', default=0, type=int)

        parser.add_argument('-style_weight', default=0.5, type=float)

        parser.add_argument('-reward_weight_cider', default=0, type=float)
        parser.add_argument('-reward_weight_ppl', default=1, type=float)
        parser.add_argument('-reward_weight_clf', default=1, type=float)

        parser.add_argument('-use_mem', default=1, type=int)
        parser.add_argument('-mem_size', default=100, type=int)

        parser.add_argument('-random_style_mask', default=0, type=int)

    def init_data(self):
        vocab_filename = 'vocab_join_coco_senticap_flickrstyle.json'
        vocab = load_custom(os.path.join(vocab_path, vocab_filename))
        # term_vocab = load_custom(os.path.join(vocab_path, 'vocab_terms_join_coco_senticap_flickrstyle.json'))
        term_vocab = Vocabulary()
        term_vocab = add_style_tokens(term_vocab,
                                      ['<factual>', '<humorous>', '<romantic>', '<positive>', '<negative>', '<book>'])

        tag_to_style_label = {
            'factual': '<factual>',
            'humor': '<humorous>', 'romantic': '<romantic>',
            'positive': '<positive>', 'negative': '<negative>',
            'book': '<book>'
        }
        self.tag_to_style_label = tag_to_style_label

        styles = self.args.styles

        if styles in ['humor', 'romantic']:
            dataset_name = 'flickrstyle'
        elif styles in ['positive', 'negative']:
            dataset_name = 'senticap'
        elif styles == 'factual':
            dataset_name = 'coco'
        else:
            raise Exception('invalid style: ' + styles)

        random_style_mask = self.args.random_style_mask

        dataset_style = CaptionDataset(vocab=vocab, dataset_name=dataset_name, split='train', max_sent_length=25,
                                       image_mode='none', term_vocab=term_vocab, use_sg=True,
                                       use_style_label=True, style_label=tag_to_style_label,
                                       filter_style_tag=styles, random_style_mask=random_style_mask)

        train_dataset = dataset_style

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       collate_fn=get_collate_fn(dataset_style.get_sub_collate_fn()),
                                                       batch_size=60, num_workers=0, shuffle=True)

        test_dataset = CaptionDataset(vocab=vocab, dataset_name=dataset_name, split='test', max_sent_length=25,
                                      image_mode='none', term_vocab=term_vocab,
                                      # use_sg=True,
                                      use_sg=False, use_image_sg=True,
                                      use_style_label=True, style_label=tag_to_style_label,
                                      filter_style_tag=styles, random_style_mask=random_style_mask)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      collate_fn=get_collate_fn(test_dataset.get_sub_collate_fn()),
                                                      batch_size=1, num_workers=0)

        self.vocab = vocab
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # emb = read_pretrained_glove_embedding(vocab=self.vocab, cache_file=os.path.join(vocab_path, vocab_filename + '.glove'))
        # self.emb = torch.Tensor(emb)
        self.emb = None

    def init_model(self, state_dict=None):
        use_mem = self.args.use_mem
        mem_size = self.args.mem_size

        model = TopDownAttnModelSG(vocab=self.vocab, use_attn=True, word_embedding=self.emb,
                                    style_weight=self.args.style_weight,
                                    merge_mode='add',
                                    embedding_dim=128,
                                    hidden_dim=256,
                                   feat_dim=128,
                                   mem_size=mem_size,
                                   use_mem=use_mem)
        model.to(device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': 5e-4}
        ])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.8)
        if state_dict is not None:
            print('--------')
            model.load_state_dict(state_dict['model'])
            # FIXME: this should not be commented
            # optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])

        self.epoch = 1
        self.global_step = 0

        self.model, self.optimizer = model, optimizer
        self.scheduler = scheduler

    def train_epoch(self):
        model, optimizer = self.model, self.optimizer
        scheduler = self.scheduler
        style_label_counter = Counter()

        scheduler.step()
        sc_flag = self.args.sc_after > 0 and self.epoch >= self.args.sc_after
        max_sample_seq_len = 23
        self.sc_flag = sc_flag

        if sc_flag:
            init_style_scorer(self.args.styles.split(','))
            reward_weights = {'cider': self.args.reward_weight_cider, 'ppl': self.args.reward_weight_ppl,
                              'clf': self.args.reward_weight_clf}

            s = float(sum(reward_weights.values()))
            for k, v in reward_weights.items():
                reward_weights[k] = v / s
            print('weight:', reward_weights)

        for i, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), ncols=64):
            if self.args.styles == 'factual':
                if i > 0 and i % 1000 == 0:
                    self.save_model(os.path.join(self.checkpoint_folder, 'checkpoint_{}'.format(self.global_step)), state_dict=self.get_state_dict())
                    self.test()

            model.train(True)
            if self.args.styles == 'factual':
                model.enable_style = False
            else:
                model.enable_style = True

            image_id = batch_data['image_id']
            sentence_id = batch_data['sentence_id']

            _token, _length = batch_data['token'], batch_data['length']
            _style_mask = batch_data['style_mask']
            _style_label = batch_data['style_label']

            words = batch_data['words']

            style_label_counter.update(batch_data['style_label'])

            token = torch.LongTensor(_token).to(device)
            length = torch.LongTensor(_length).to(device)
            style_mask = torch.Tensor(_style_mask).to(device)
            style_label = torch.Tensor(_style_label).to(device)

            sg_batch = batch_data['sg']

            optimizer.zero_grad()

            if not sc_flag:
                token_input = token[:, :-1].contiguous()
                token_target = token[:, 1:].contiguous()

                input_feature = (token, length, style_mask, sg_batch, style_label)
                output_logits = model.forward(input_feature=input_feature, input_sentence=token_input)
                loss = masked_cross_entropy(output_logits, token_target, length - 1)
            else:
                input_feature = (token, length, style_mask, sg_batch, style_label)
                sample_logprob, sample_seq, _ = model.sample(input_feature=input_feature,
                                                             max_length=max_sample_seq_len + 1,
                                                             sample_max=False)
                model.train(False)
                with torch.no_grad():
                    greedy_logprob, greedy_seq, _ = model.sample(input_feature=input_feature,
                                                                 max_length=max_sample_seq_len + 1,
                                                                 sample_max=True)
                model.train(True)

                train_dataloader = self.train_dataloader
                gts_raw = []
                for _i, id in enumerate(image_id):
                    # g = []
                    # for s in train_dataloader.dataset.get_caption_item_by_image_id(id).sentences:
                    #     words = s.words + [util.Vocabulary.end_token]
                    #     g.append(' '.join(words[:25]))

                    _words = words[_i] + [util.Vocabulary.end_token]
                    g = [' '.join(_words)]

                    gts_raw.append(g)

                reward = get_style_reward(sample_seq, greedy_seq, gts_raw, self.vocab, style_labels=_style_label,  reward_weights=reward_weights)

                loss = util.reward.rl_criterion(log_prob=sample_logprob, generated_seq=sample_seq, reward=reward)

                avg_reward = np.mean(reward[:, 0])
                self.writer.add_scalar('reward', avg_reward, global_step=self.global_step)

            loss.backward(retain_graph=True)
            util.clip_gradient(optimizer, 2.)
            optimizer.step()

            self.global_step += 1

            if self.global_step % 10 == 0:
                loss_scalar = loss.detach().cpu().numpy()
                self.writer.add_scalar('loss', loss_scalar, global_step=self.global_step)
                print('loss: {:.6f}'.format(loss_scalar))

            # if self.global_step > 0 and self.global_step % 200 == 0:
            #     self.save_model(os.path.join(self.checkpoint_folder, 'model_step{}'.format(self.global_step)), self.get_state_dict())

    def test_epoch(self):
        is_styled = (self.args.styles != 'factual')
        if is_styled:
            if self.args.action == 'train' and (not self.sc_flag):
                if (self.epoch % 5 != 0):
                    return

        model = self.model
        vocab = self.vocab

        model.train(False)

        result_generator = COCOResultGenerator()
        style_tag = self.args.styles
        dataset_name = self.test_dataloader.dataset.dataset_name
        evaluator = StyledEvaluate(styled_dataset=dataset_name, style_names=[style_tag],
                                   use_clf=is_styled, use_lm=is_styled, use_srilm=is_styled)

        for i, batch_data in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), ncols=64):
            image_ids = batch_data['image_id']
            batch_size = len(image_ids)

            raw = batch_data['raw']

            sg_batch = batch_data['sg']

            for batch_index in range(batch_size):
                image_id = image_ids[batch_index]
                result_generator.add_annotation(image_id, raw[batch_index])

                if result_generator.has_output(image_id):
                    continue

                _factual_tokens = batch_data['factual_tokens'][batch_index : batch_index + 1]
                _factual_length = batch_data['factual_length'][batch_index : batch_index + 1]
                _style_mask = np.zeros(shape=_factual_tokens.shape, dtype=np.float64)

                factual_tokens = torch.LongTensor(_factual_tokens).to(device)
                factual_length = torch.LongTensor(_factual_length).to(device)
                style_mask = torch.Tensor(_style_mask).to(device)

                sg = [sg_batch[batch_index]]
                log_prob_seq, word_id_seq, _ = model.sample_beam(input_feature=(factual_tokens, factual_length, style_mask, sg),
                                                                 max_length=25, beam_size=1)

                words = util.trim_generated_tokens(word_id_seq)
                words = [vocab.get_word(i) for i in words]
                sent = ' '.join(words)

                sent_factual = ' '.join([vocab.get_word(i) for i in util.trim_generated_tokens(_factual_tokens[0])])
                print('factual: {}'.format(sent_factual))
                print('styled: {}'.format(sent))
                gt = raw[batch_index]

                result_generator.add_output(image_id, sent, metadata={'factual': sent_factual, 'gt': gt})

        ann_file = os.path.join(self.save_folder, 'annotation_{}_{}.json'.format(dataset_name, style_tag))
        result_file = os.path.join(self.save_folder, 'result_{}_{}_{}.json'.format(dataset_name, style_tag, self.epoch))
        result_generator.dump_annotation_and_output(ann_file, result_file)

        metrics, img_scores = evaluator.evaluate(ann_file, result_file, style_name=style_tag, return_img_scores=True)
        self.writer.add_scalars(main_tag='metric_{}/'.format(style_tag), tag_scalar_dict=metrics,
                                global_step=self.global_step)

        _, outputs = result_generator.get_annotation_and_output()
        for i, item in enumerate(outputs):
            if img_scores[i]['image_id'] == item['image_id']:
                item['meta']['scores'] = img_scores[i]
        result_generator.dump_output(result_file)

        metric_file = os.path.join(self.save_folder, 'metrics_{}_{}.csv'.format(dataset_name, style_tag))

        metrics = list(metrics.items())
        metrics.sort(key=lambda x: x[0])

        data = []
        if not os.path.exists(metric_file):
            data.append(['-'] + [m[0] for m in metrics])
        data.append(['epoch {}'.format(self.epoch)] + ['{:.6f}'.format(m[1]) for m in metrics])
        with open(metric_file, 'a') as f:
            writer = csv.writer(f)
            for line in data:
                writer.writerow(line)

    def get_state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'vocab': self.vocab,
                'args': self.args}


if __name__ == '__main__':
    p = StyledTextPipeline()
    p.run()
