import os
import sys
from tempfile import NamedTemporaryFile

import kenlm

from config import *
import util
from util import *


def train_styled_lm(dataset_name, style_tag, split='train', order=2, output_dir=os.path.join(data_path, 'lm'), lang='eng'):
    dataset = load_custom(os.path.join(annotation_path, 'dataset_{}.json'.format(dataset_name)))
    dataset = dataset['caption_item']

    sents = []
    for item in dataset:
        if split == 'all' or split == item.split:
            sents.extend(item.sentences)
    if style_tag is not None:
        sents = list(filter(lambda x: x.tag == style_tag, sents))

    print('loaded {} sentences, style = {}'.format(len(sents), style_tag))

    f_text = NamedTemporaryFile('w')

    if lang == 'eng':
        for sent in sents:
            f_text.write(sent.raw + '\n')
    elif lang == 'chn':
        for sent in sents:
            f_text.write(' '.join(sent.words) + '\n')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('output to {}'.format(output_dir))

    cmd = r'{exec} -o {order} --verbose_header --text {text} --arpa {arpa}'.format(
        exec=os.path.join(kenlm_build_path, 'bin', 'lmplz'),
        order=order, text=f_text.name,
        arpa=os.path.join(output_dir, '{}_{}.arpa'.format(dataset_name, style_tag)),
        # vocab=os.path.join(output_dir, '{}_{}.vocab'.format(dataset_name, style_tag)),
    )
    print('executing:')
    print(cmd)

    os.system(cmd)

def load_lm(arpa_path):
    print('loading language model from {}'.format(arpa_path))
    lm = kenlm.LanguageModel(arpa_path)
    return lm

def eval_sentence(lm, sent):
    full_scores = lm.full_scores(sent, bos=True, eos=True)
    lp, gram, oov = zip(*full_scores)
    # change to log base 2
    lp /= np.log10(2)
    lp *= -1
    mean_score = np.mean(lp)    # perplexity
    return mean_score

def eval_sentences(lm, sent_list):
    scores = [eval_sentence(lm, sent) for sent in sent_list]
    return np.mean(scores), scores


if __name__ == '__main__':
    # train_styled_lm('flickrstyle', 'humor')
    # train_styled_lm('flickrstyle', 'romantic')
    # train_styled_lm('senticap', 'positive')
    # train_styled_lm('senticap', 'negative')
    # train_styled_lm('bookcorpus', 'book')
    for style in ['negative', 'positive', 'romantic', 'humor']:
        train_styled_lm('chn_styled_char', style, lang='chn')
        train_styled_lm('chn_styled_word', style, lang='chn')






