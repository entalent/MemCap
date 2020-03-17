import os
import sys

data_path = os.path.join('..', 'data')
vocab_path = os.path.join(data_path, 'vocab')
annotation_path = os.path.join(data_path, 'preprocessed')
feat_path = os.path.join(data_path, 'feat')

video_feat_path = os.path.join(data_path, 'video_feat')

java_path = '/usr/local/lib/jdk1.8.0_211/bin/java'
os.environ['PATH'] += ':' + os.path.split(java_path)[0]

sys.path.append(os.getcwd())

sys.path.append('/media/wentian/sdb2/work/coco-caption-master')
# sys.path.append('/home/mcislab/zwt/coco-caption-master')

kenlm_build_path = '/home/wentian/library/kenlm/build'

srilm_path = '/home/wentian/library/srilm/lm/bin/i686-m64'
# srilm_path = '/home/mcislab/library/srilm/lm/bin/i686-m64'

sg_path = os.path.join('..', 'sent_sg_parse')

neo4j_uri = 'http://localhost:7474'
neo4j_username = 'neo4j'
neo4j_password = '123456'

clf_textcnn_word2idx_path = '/media/wentian/sdb2/work/styled_caption/data/clf_cnn/word2idx.json'

glove_embedding_file = '/media/wentian/sdb2/work/caption_dataset/glove/glove.6B.300d.txt'
