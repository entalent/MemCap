import os
import sys

from styled_eval import *

if __name__ == '__main__':
    ann = r'/media/wentian/sdb2/work/styled_caption/save/2019-08-05_23-05-48_mem_positive/annotation_senticap_positive.json'
    res = r'/media/wentian/sdb2/work/styled_caption/save/2019-08-05_23-05-48_mem_positive/result_senticap_positive_9.json'

    d = json.load(open(ann, 'r'))
    for item in d['annotations']:
        if '\r\n' in item['caption']:
            idx = item['caption'].index('\r\n')
            item['caption'] = item['caption'][:idx]
    with open(ann, 'w') as f:
        json.dump(d, f)

    dataset = 'senticap'
    style = 'positive'

    with open(ann, 'r') as f:
        d1 = json.load(f)
        imgs = set(i['id'] for i in d1['images'])
    with open(res, 'r') as f:
        d2 = json.load(f)

    if len(imgs) != len(d2):
        d2 = list(filter(lambda x: x['image_id'] in imgs, d2))
        res += '_'
        with open(res, 'w') as f:
            json.dump(d2, f)

    e = StyledEvaluate(styled_dataset=dataset, style_names=[style])
    metrics = e.evaluate(ann, res, style)
    print(metrics)