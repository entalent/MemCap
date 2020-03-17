import os
import re

from config import *
from util import *
import util
from styled_eval import *
import csv

keys = ['-', 'Bleu_1', 'Bleu_3', 'METEOR', 'CIDEr', 'srilm', 'clf']

def eval(dir):
    f_result = open(os.path.join(dir, 'all_eval.csv'), 'w')
    writer = csv.writer(f_result)
    writer.writerow(keys)

    for file in os.listdir(dir):
        if not (file.startswith('result') and file.endswith('.json')):
            continue

        strs = re.split('[.|_]', file)
        dataset, style, epoch = strs[1], strs[2], strs[3]

        ann_file = os.path.join(dir, 'annotation_{}_{}.json'.format(dataset, style))

        evaluator = StyledEvaluate(styled_dataset=dataset, style_names=[style])
        metrics = evaluator.evaluate(ann_file, os.path.join(dir, file), style_name=style)

        print(file)
        print(metrics)

        data = [file]
        for key in keys[1:]:
            data.append(str(metrics[key]))
        writer.writerow(data)
        f_result.flush()
    f_result.close()

if __name__ == '__main__':
    eval(sys.argv[1])