import io
import re
import subprocess
from tempfile import NamedTemporaryFile

from config import *
import util
from util import *


def system(cmd):
    print("executing:", cmd)
    ret = os.system(cmd)
    ret >>= 8
    print('returned {}'.format(ret))

def popen(args):
    with open(os.devnull, 'w') as devnull:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=devnull)
    lines = []
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        lines.append(line)
    return lines

ngram_bin = os.path.join(srilm_path, 'ngram')
ngram_count_bin = os.path.join(srilm_path, 'ngram-count')

def train_styled_lm(dataset_name, style_tag, output_dir=os.path.join(data_path, 'srilm'), split='train', order=3, lang='eng'):
    dataset = load_custom(os.path.join(annotation_path, 'dataset_{}.json'.format(dataset_name)))
    dataset = dataset['caption_item']

    sents = []
    for item in dataset:
        if split == 'all' or split == item.split:
            sents.extend(item.sentences)
    sents = list(filter(lambda x: x.tag == style_tag, sents))

    print('loaded {} sentences, style = {}'.format(len(sents), style_tag))

    f_text = NamedTemporaryFile('w')
    print('writing to temporary file: {}'.format(f_text.name))

    if lang == 'eng':
        for sent in sents:
            f_text.write(' '.join(sent.words) + '\n')
    elif lang == 'chn':
        for sent in sents:
            f_text.write(' '.join(sent.words) + '\n')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('output to {}'.format(output_dir))

    cmd = r'{} -text {} -order 3 -write train.txt.count'.format(ngram_count_bin, f_text.name)
    system(cmd)

    cmd = r'{} -read train.txt.count -order 3 -lm {} -interpolate -kndiscount'.format(
        ngram_count_bin,
        os.path.join(output_dir, '{}_{}.srilm'.format(dataset_name, style_tag))
    )
    system(cmd)

    f_text.close()


def _parse_float(a: str):
    try:
        return float(a)
    except:
        return float('nan')

def eval_sentences(lm_path, sent_list):
    tmp_file_name = '.sents_{}'.format(gen_random_string(15))
    f_text = open(tmp_file_name, 'w')
    for sent in sent_list:
        f_text.write(sent + '\n')
    f_text.flush()
    f_text.close()

    cmd = [ngram_bin, '-ppl', f_text.name, '-order', '3', '-lm', lm_path, '-debug', '1', '1>/dev/null', '2>/dev/null']
    lines = popen(cmd)

    scores = []

    for line in lines:
        if 'logprob=' in line and 'ppl=' in line and 'ppl1=' in line:
            strs = line.strip().split()
            ppl = _parse_float(strs[5])
            # logprob, ppl, ppl1 = [_parse_float(s) for s in (strs[3], strs[5], strs[7])]
            scores.append(ppl)

    if len(scores) - 1 != len(sent_list):
        scores = [scores[-1]] * (len(sent_list) + 1)

    ppl = scores[-1]     # overall score

    os.remove(tmp_file_name)

    return ppl, np.array(scores[:-1])


def train_lm():
    train_styled_lm('senticap', 'positive')
    # train_styled_lm('senticap', 'negative')
    # train_styled_lm('flickrstyle', 'romantic')
    # train_styled_lm('flickrstyle', 'humor')

    # for style in ['negative', 'positive', 'romantic', 'humor']:
    #     train_styled_lm('chn_styled_char', style, lang='chn')
    #     train_styled_lm('chn_styled_word', style, lang='chn')

def _eval(result_dir):
    for file in os.listdir(result_dir):
        if not (file.startswith('metrics_') and file.endswith('.csv')):
            continue
        strs = re.split('[_|.]', file)
        dataset, style, epoch = strs[1], strs[2], strs[3]

        f = open(os.path.join(result_dir, 'result_{}_{}_{}.json'.format(dataset, style, epoch)))
        d = json.load(f)
        sents = [i['caption'] for i in d]

        lm = os.path.join(data_path, 'srilm', '{}_{}.srilm'.format(dataset, style))
        print(file)
        print(eval_sentences(lm, sents))

def _eval_file(file):
    def _eval_sentences(lm_path, sent_list):
        f_text = NamedTemporaryFile('w')
        for sent in sent_list:
            f_text.write(sent + '\n')

        cmd = [ngram_bin, '-ppl', f_text.name, '-order', '3', '-lm', lm_path,
               '-debug', '1']
        os.system(' '.join(cmd))

    f = open(file)
    strs = re.split('[_|.]', os.path.split(file)[-1])
    dataset, style, epoch = strs[1], strs[2], strs[3]
    d = json.load(f)
    d.sort(key=lambda x: x['image_id'])
    sents = [i['caption'] for i in d]
    lm = os.path.join(data_path, 'srilm', '{}_{}.srilm'.format(dataset, style))
    print(_eval_sentences(lm, sents))


if __name__ == '__main__':
    # _eval(r'/media/wentian/sdb2/work/styled_caption/save/2019-08-01_11-55-39_4style_cat_gt_term_no_factual')
    # _eval_file('/media/wentian/sdb2/work/styled_caption/save/2019-08-01_16-28-33_1style_humorous_gt_term/result_flickrstyle_humor_2.json')
    train_lm()
