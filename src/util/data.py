import time
from abc import abstractmethod
from collections import defaultdict

import torch.utils.data
import torch.utils.data
from tqdm import tqdm

from .customjson import *

class ImageItem(JSONSerializable):
    def __init__(self, image_id=None, image_filename=None):
        """

        :param image_id: int
        :param image_filename: str
        """
        super().__init__()
        self.image_id, self.image_filename = image_id, image_filename

class SentenceItem(JSONSerializable):
    def __init__(self, sentence_id=None, raw=None, words=None, token_ids=None, tag=None):
        """

        :param sentence_id: int
        :param raw: str
        :param words: can be None
        :param token_ids: can be None
        """
        self.sentence_id, self.raw, self.words, self.token_ids = \
            sentence_id, raw, words, token_ids
        self.tag = tag

class ImageSentencePair(JSONSerializable):
    def __init__(self, image=None, sentence=None, split=None):
        # super().__init__()
        self.image, self.sentence, self.split = image, sentence, split

class CaptionItem(JSONSerializable):
    def __init__(self, image=None, sentences=None, split=None):
        # super().__init__()
        self.image, self.sentences, self.split = image, sentences, split

JSONSerializable.register_cls(ImageItem, 'II', {'image_id': 'id', 'image_filename': 'file'})
JSONSerializable.register_cls(SentenceItem, 'SI', {})
JSONSerializable.register_cls(ImageSentencePair, 'ISP', {'image': 'i', 'sentence': 's', 'split': 'sp'})
JSONSerializable.register_cls(CaptionItem, 'CI', {'image': 'i', 'sentences': 'ss', 'split': 'sp'})


class BaseCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        dataset_name = kwargs['dataset_name']
        split = kwargs.get('split', 'all')      # default: all
        vocab = kwargs.get('vocab', None)

        print('loading dataset {}, split {}, vocab size {}'.format(dataset_name, split, None if vocab is None else len(vocab)))

        self._kwargs = kwargs
        self.dataset_name = dataset_name
        self.split = split
        self.vocab = vocab

        self.image_list = []
        self.sentence_list = []
        self.caption_item_list = []             # list of CaptionItem instance
        self.image_sentence_pair_list = []      # list of ImageSentencePair

        self.caption_item_split = defaultdict(list)
        self.image_sentence_pair_split = defaultdict(list)

        start_time = time.time()
        self.load()

        for caption_item in self.caption_item_list:
            split = caption_item.split
            self.image_list.append(caption_item.image)
            self.caption_item_split[split].append(caption_item)
            self.caption_item_split['all'].append(caption_item)
            self.sentence_list.extend(caption_item.sentences)
            for sent in caption_item.sentences:
                pair = ImageSentencePair(image=caption_item.image, sentence=sent, split=split)
                self.image_sentence_pair_list.append(pair)
                self.image_sentence_pair_split[split].append(pair)
                self.image_sentence_pair_split['all'].append(pair)

        for sent in self.sentence_list:
            sent.token_ids = [self.vocab.get_index(w) for w in sent.words]

        print('load used {:.3f}s'.format(time.time() - start_time))

        info = []
        for split in self.caption_item_split.keys():
            info.append('{}: {}'.format(split, len(self.caption_item_split[split])))
        print('splits: {}'.format(' '.join(info)))

        # iterate on this
        self.iter_pairs = self.image_sentence_pair_split[self.split]

        self.image_id_map = dict((image_item.image_id, image_item) for image_item in self.image_list)
        self.sentence_id_map = dict((sentence_item.sentence_id, sentence_item) for sentence_item in self.sentence_list)
        self.image_id_map_2 = dict((caption_item.image.image_id, caption_item) for caption_item in self.caption_item_list)
        self.sentence_id_map_2 = {}
        for caption_item in self.caption_item_list:
            for sent in caption_item.sentences:
                self.sentence_id_map_2[sent.sentence_id] = caption_item

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def load(self):
        """
        init self.caption_item_list (of all splits)
        :return:
        """
        pass

    def get_image_item(self, image_id):
        return self.image_id_map[int(image_id)]

    def get_sentence_item(self, sentence_id):
        return self.sentence_id_map[int(sentence_id)]

    def get_caption_item_by_image_id(self, image_id):
        return self.image_id_map_2[int(image_id)]

    def get_caption_item_by_sentence_id(self, sentence_id):
        return self.sentence_id_map_2[int(sentence_id)]


def read_binary_blob(file_name):
    fid = open(file_name, 'rb')

    # s contains size of the blob e.g. num x chanel x length x height x width
    s = np.fromfile(fid, np.int32, 5)

    m = s[0] * s[1] * s[2] * s[3] * s[4]

    # data is the blob binary data in single precision (e.g float in C++)
    data = np.fromfile(fid, np.float32, m)
    data = data.reshape(s)

    fid.close()
    return data


@lru_cache(maxsize=50000)
def load_np(filename):
    return np.load(filename)

